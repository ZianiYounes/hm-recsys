"""Two-tower : entrainement, retrieval, cache par semaine cible et evaluation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from recsys.config import Paths, load_config
from recsys.data.datasets import GpuHistory, SequenceStore
from recsys.data.split import history_for, load_split, target_week_of, targets_for
from recsys.evaluation.evaluate import evaluate_predictions
from recsys.evaluation.report import print_results_table
from recsys.models.baselines import baseline_candidates, blend_by_priority
from recsys.models.two_tower import TwoTower, in_batch_softmax_loss
from recsys.models.two_tower.factory import build_two_tower
from recsys.retrieval.torch_topk import topk_inner_product
from recsys.utils.io import write_json, write_parquet
from recsys.utils.seed import seed_everything
from recsys.utils.timer import timer
from recsys.utils.tracking import MlflowRun

KS = (12, 50, 100)


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def embed_all_items(model: TwoTower, n_items: int, device, batch: int = 65536) -> torch.Tensor:
    out = [
        model.encode_items(torch.arange(s, min(s + batch, n_items), device=device))
        for s in range(0, n_items, batch)
    ]
    return torch.cat(out).float()


@torch.no_grad()
def embed_customers(
    model: TwoTower,
    hist: GpuHistory,
    user_feats: torch.Tensor,
    customers: np.ndarray,
    target_day: int,
    max_len: int,
    batch: int,
) -> torch.Tensor:
    out = []
    for s in range(0, len(customers), batch):
        c = torch.from_numpy(customers[s : s + batch]).long().to(hist.device)
        items, gaps, mask = hist.last_purchases(c, target_day, max_len)
        out.append(model.encode_users(items, gaps, mask, user_feats[c]).float())
    return torch.cat(out)


def retrieve(model, hist, user_feats, customers, target_day, allowed, cfg) -> dict[str, np.ndarray]:
    """Top-K articles par client + embeddings (reutilises comme features par le ranker)."""
    model.eval()
    use_amp = hist.device.type == "cuda"
    with torch.autocast(hist.device.type, dtype=torch.bfloat16, enabled=use_amp):
        item_emb = embed_all_items(model, len(allowed), hist.device)
        user_emb = embed_customers(
            model, hist, user_feats, customers, target_day,
            cfg["data"]["max_history"], cfg["eval"]["user_batch"],
        )
    scores, idx = topk_inner_product(
        user_emb, item_emb, cfg["eval"]["k"], allowed, cfg["eval"]["item_chunk"]
    )
    return {
        "scores": scores,
        "idx": idx,
        "user_emb": user_emb.cpu().numpy().astype(np.float32),
        "item_emb": item_emb.cpu().numpy().astype(np.float32),
    }


def to_frames(customers: np.ndarray, scores: np.ndarray, idx: np.ndarray):
    """-> (preds : client + liste ordonnee, candidats : client, article, score)."""
    preds = pl.DataFrame(
        {"customer_idx": customers.astype(np.int32), "preds": idx.astype(np.int32).tolist()},
        schema={"customer_idx": pl.Int32, "preds": pl.List(pl.Int32)},
    )
    k = idx.shape[1]
    cands = pl.DataFrame(
        {
            "customer_idx": np.repeat(customers, k).astype(np.int32),
            "article_idx": idx.reshape(-1).astype(np.int32),
            "score": scores.reshape(-1).astype(np.float32),
        }
    ).filter(pl.col("score").is_finite())
    return preds, cands


# --------------------------------------------------------------------------- entrainement
def fit_two_tower(
    paths: Paths,
    week: int,
    history: pl.DataFrame,
    targets: pl.DataFrame | None,
    cfg: dict[str, Any],
    run: MlflowRun | None = None,
) -> dict[str, Any]:
    """Entraine sur l'historique (semaines > week).

    Avec `targets` : evalue a chaque epoch et renvoie les top-K des clients cibles
    (preds, candidates, user_ids, user_emb). Sans `targets` (serving) : seulement le modele.
    Renvoie toujours : model, item_emb, allowed_ids."""
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA indisponible : entrainement sur CPU (tres lent)")

    name = cfg.get("name", "two_tower")
    dc, tc = cfg["data"], cfg["train"]
    articles = pl.read_parquet(paths.processed / "articles.parquet")
    customers = pl.read_parquet(paths.processed / "customers.parquet")
    n_items, n_users = articles.height, customers.height
    ref_day = int(history["day"].max())

    with timer("sequences + exemples"):
        store = SequenceStore.from_history(history, n_users)
        samples = store.training_samples(week, dc["train_weeks"])
    n_samples = len(samples["item"])
    logger.info(f"{len(store.items):,} achats en historique, {n_samples:,} exemples d'entrainement")

    model, user_feats_np = build_two_tower(paths, cfg, articles, customers)
    model = model.to(device)
    logger.info(f"{name} : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametres")

    hist = GpuHistory(store, device)
    S = {k: torch.from_numpy(v.astype(np.int64)).to(device) for k, v in samples.items()}
    user_feats = torch.from_numpy(user_feats_np).to(device)
    freq = np.bincount(samples["item"], minlength=n_items).astype(np.float64)
    log_q = torch.from_numpy(np.log(np.maximum(freq / freq.sum(), 1e-12))).float().to(device)

    # Articles recommandables : vendus pendant les `candidate_pool_weeks` dernieres semaines
    recent = history.filter(pl.col("week") <= week + dc["candidate_pool_weeks"])
    allowed_ids = np.sort(recent["article_idx"].unique().to_numpy().astype(np.int64))
    allowed = torch.zeros(n_items, dtype=torch.bool)
    allowed[torch.from_numpy(allowed_ids)] = True
    allowed = allowed.to(device)
    logger.info(f"Catalogue recommandable : {len(allowed_ids):,} / {n_items:,} articles")

    bs = tc["batch_size"]
    steps_per_epoch = math.ceil(n_samples / bs)
    total_steps = max(tc["epochs"] * steps_per_epoch, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(  # warmup 5 % puis decroissance cosinus
        opt, max_lr=tc["lr"], total_steps=total_steps, pct_start=max(0.05, 3 / total_steps)
    )

    eval_customers = (
        None if targets is None else targets["customer_idx"].to_numpy().astype(np.int64)
    )
    out: dict[str, Any] = {}

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total, bar = 0.0, tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", leave=False)
        for step in bar:
            ids = perm[step * bs : (step + 1) * bs]
            items, gaps, mask = hist.gather(
                S["hist_start"][ids], S["hist_end"][ids], S["target_day"][ids], dc["max_history"]
            )
            pos = S["item"][ids]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                u = model.encode_users(items, gaps, mask, user_feats[S["customer"][ids]])
                v = model.encode_items(pos)
            loss = in_batch_softmax_loss(u.float(), v.float(), pos, log_q, tc["temperature"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            total += loss.item()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = total / steps_per_epoch
        if targets is None:
            logger.info(f"[{name} | serving] epoch {epoch} | loss {train_loss:.4f}")
            continue
        with timer(f"evaluation epoch {epoch}"):
            r = retrieve(model, hist, user_feats, eval_customers, ref_day + 1, allowed, cfg)
            preds, cands = to_frames(eval_customers, r["scores"], r["idx"])
            m = evaluate_predictions(preds, targets, KS, n_items)
        logger.info(
            f"[{name} | semaine {week}] epoch {epoch} | loss {train_loss:.4f}"
            f" | MAP@12 {m['map_at_12']:.4f} | Recall@12 {m['recall_at_12']:.4f}"
            f" | Recall@100 {m['recall_at_100']:.4f}"
        )
        if run is not None:
            run.log({"train_loss": train_loss, **m}, step=epoch)
        out = {"preds": preds, "candidates": cands, "user_emb": r["user_emb"],
               "item_emb": r["item_emb"], "user_ids": eval_customers.astype(np.int32)}

    if targets is None:
        model.eval()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out["item_emb"] = embed_all_items(model, n_items, device).cpu().numpy()
    out |= {"model": model, "allowed_ids": allowed_ids}
    return out


# --------------------------------------------------------------------------- cache par semaine
def cache_dir(paths: Paths, week: int, name: str = "two_tower") -> Path:
    return paths.interim / f"{name}_w{week}"


def save_week_cache(paths: Paths, week: int, out: dict[str, Any], name: str) -> None:
    d = cache_dir(paths, week, name)
    d.mkdir(parents=True, exist_ok=True)
    write_parquet(out["candidates"], d / "candidates.parquet")
    np.save(d / "user_ids.npy", out["user_ids"])
    np.save(d / "user_emb.npy", out["user_emb"])
    np.save(d / "item_emb.npy", out["item_emb"])


def load_week_cache(paths: Paths, week: int, name: str = "two_tower") -> dict[str, Any] | None:
    d = cache_dir(paths, week, name)
    if not (d / "item_emb.npy").exists():
        return None
    return {
        "candidates": pl.read_parquet(d / "candidates.parquet"),
        "user_ids": np.load(d / "user_ids.npy"),
        "user_emb": np.load(d / "user_emb.npy"),
        "item_emb": np.load(d / "item_emb.npy"),
    }


def ensure_two_tower_cache(paths: Paths, week: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Charge les sorties du retriever pour la semaine cible, ou l'entraine si besoin."""
    name = cfg.get("name", "two_tower")
    cached = load_week_cache(paths, week, name)
    if cached is not None:
        logger.info(f"{name} semaine {week} : cache trouve")
        return cached
    logger.info(f"{name} semaine {week} : pas de cache, entrainement (quelques minutes)")
    tx = pl.scan_parquet(paths.processed / "transactions.parquet")
    history = history_for(tx, week).collect()
    targets = targets_for(tx, week).collect()
    out = fit_two_tower(paths, week, history, targets, cfg)
    save_week_cache(paths, week, out, name)
    return load_week_cache(paths, week, name)


def save_checkpoint(path: Path, model: TwoTower, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, path)


# --------------------------------------------------------------------------- script val / test
def run_two_tower(paths: Paths, split: str, cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    name = cfg.get("name", "two_tower")
    week = target_week_of(paths, split)
    history_lazy, targets = load_split(paths, split)
    with timer("chargement"):
        history = history_lazy.collect()
    n_items = pl.read_parquet(paths.processed / "articles.parquet", columns=["article_idx"]).height
    ref_day = int(history["day"].max())

    run = MlflowRun(name, f"{name}_{split}", {"split": split, **cfg})
    out = fit_two_tower(paths, week, history, targets, cfg, run)
    save_week_cache(paths, week, out, name)

    base = baseline_candidates(
        history.lazy(), week, targets.select("customer_idx"), ref_day, load_config("baselines")
    )
    k = cfg["eval"]["k"]
    results = {
        "fusion baselines": evaluate_predictions(
            blend_by_priority([base["re-achat"], base["co-achat"], base["popularite"]], k),
            targets, KS, n_items,
        ),
        name: evaluate_predictions(out["preds"], targets, KS, n_items),
        f"fusion + {name}": evaluate_predictions(
            blend_by_priority(
                [base["re-achat"], out["candidates"], base["co-achat"], base["popularite"]], k
            ),
            targets, KS, n_items,
        ),
    }
    print_results_table(results, f"{name} ({split})")
    run.log({f"blend_{key}": v for key, v in results[f"fusion + {name}"].items()})

    model_path = paths.models / f"{name}_{split}.pt"
    save_checkpoint(model_path, out["model"], cfg)
    write_json(results, paths.reports / f"{name}_{split}.json")
    logger.success(f"Modele sauvegarde : {model_path}")
    run.end()
    return results
