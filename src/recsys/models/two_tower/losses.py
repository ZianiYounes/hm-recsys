from __future__ import annotations

import torch
import torch.nn.functional as F


def in_batch_softmax_loss(
    user_emb: torch.Tensor,   # [B, D] normalises
    item_emb: torch.Tensor,   # [B, D] normalises (article achete par chaque client)
    item_ids: torch.Tensor,   # [B]
    log_q: torch.Tensor,      # [n_items] log-frequence des articles dans les positifs
    temperature: float,
) -> torch.Tensor:
    """Softmax "in-batch" : pour chaque client, son article est le positif et les articles
    des autres clients du batch servent de negatifs.

    Correction logQ (Yi et al., 2019) : les articles populaires apparaissent plus souvent
    comme negatifs ; on soustrait log(frequence) pour ne pas trop les penaliser.
    Les doublons (meme article pour deux clients) ne sont pas comptes comme negatifs."""
    logits = (user_emb @ item_emb.T) / temperature
    logits = logits - log_q[item_ids][None, :]
    same_item = item_ids[:, None] == item_ids[None, :]
    not_self = ~torch.eye(len(item_ids), dtype=torch.bool, device=item_ids.device)
    logits = logits.masked_fill(same_item & not_self, float("-inf"))
    labels = torch.arange(len(item_ids), device=item_ids.device)
    return F.cross_entropy(logits, labels)
