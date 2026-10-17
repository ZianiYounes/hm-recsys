"""Jeu de donnees factice au format H&M (quelques centaines de clients) pour les tests
d'integration : preprocessing -> split -> serving -> API, sans telecharger Kaggle."""
from __future__ import annotations

import datetime as dt
import hashlib
import random
from pathlib import Path

import pytest

N_ARTICLES, N_CUSTOMERS, N_COLD, N_TX = 200, 300, 5, 20_000

ARTICLE_COLS = (
    "article_id,product_code,prod_name,product_type_no,product_type_name,product_group_name,"
    "graphical_appearance_no,graphical_appearance_name,colour_group_code,colour_group_name,"
    "perceived_colour_value_id,perceived_colour_value_name,perceived_colour_master_id,"
    "perceived_colour_master_name,department_no,department_name,index_code,index_name,"
    "index_group_no,index_group_name,section_no,section_name,garment_group_no,"
    "garment_group_name,detail_desc"
)
TYPES = [(253, "Vest top"), (272, "Trousers"), (265, "Dress"), (255, "T-shirt")]
COLOURS = [(9, "Black"), (10, "White"), (73, "Dark Blue"), (13, "Beige")]


def customer_id(i: int) -> str:
    return hashlib.sha256(str(i).encode()).hexdigest()


COLD_CUSTOMERS = [customer_id(N_CUSTOMERS + j) for j in range(N_COLD)]


def write_fake_raw(raw: Path, seed: int = 0) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    arts = sorted({f"0{rng.randint(108775000, 959461000)}" for _ in range(N_ARTICLES)})
    with open(raw / "articles.csv", "w", encoding="utf-8") as f:
        f.write(ARTICLE_COLS + "\n")
        for i, a in enumerate(arts):
            tno, tname = TYPES[i % len(TYPES)]
            cno, cname = COLOURS[(i // 4) % len(COLOURS)]
            desc = "" if i % 7 == 0 else '"Soft jersey, relaxed fit"'
            f.write(
                f"{a},{a[1:7]},Item {i},{tno},{tname},Garment Upper body,1010016,Solid,"
                f"{cno},{cname},4,Dark,5,{cname},1676,Jersey Basic,A,Ladieswear,1,Ladieswear,"
                f"16,Womens Everyday Basics,1002,Jersey Basic,{desc}\n"
            )
    with open(raw / "customers.csv", "w", encoding="utf-8") as f:
        f.write("customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code\n")
        for i in range(N_CUSTOMERS + N_COLD):
            fn = "" if i % 2 else "1.0"
            news = rng.choice(["NONE", "None", "Regularly", "Monthly", ""])
            age = "" if i % 10 == 0 else str(rng.randint(16, 80))
            status = rng.choice(["ACTIVE", "PRE-CREATE", ""])
            f.write(f"{customer_id(i)},{fn},{fn},{status},{news},{age},p{i % 5}\n")
    start, end = dt.date(2018, 9, 20), dt.date(2020, 9, 22)
    span = (end - start).days
    with open(raw / "transactions_train.csv", "w", encoding="utf-8") as f:
        f.write("t_dat,customer_id,article_id,price,sales_channel_id\n")
        for _ in range(N_TX):
            d = start + dt.timedelta(days=rng.randint(0, span))
            c = customer_id(rng.randrange(N_CUSTOMERS))
            f.write(f"{d},{c},{rng.choice(arts)},{rng.random() * 0.05:.6f},{rng.choice([1, 2])}\n")
        f.write(f"{end},{customer_id(0)},{arts[0]},0.03,2\n")


@pytest.fixture(scope="session")
def fake_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("hm")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATA_DIR", str(root / "data"))
    mp.setenv("ARTIFACTS_DIR", str(root / "artifacts"))
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    write_fake_raw(root / "data" / "raw")

    from recsys.config import PATHS
    from recsys.data.preprocess import run_preprocess
    from recsys.data.split import build_and_save_splits

    run_preprocess(PATHS)
    build_and_save_splits(PATHS)
    yield PATHS
    mp.undo()
