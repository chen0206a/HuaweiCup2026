def masked_mean_pool(features: torch.Tensor, padding_mask:
    torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or padding_mask.shape != features.shape[:2]:
        raise ValueError("expected features [B,T,D] and padding_mask [B,T]")
    weights = padding_mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (features * weights).sum(dim=1) / denom

def masked_attention_pool(features: torch.Tensor, padding_mask: torch.Tensor,
                          scorer: nn.Module) -> tuple[torch.Tensor,
                              torch.Tensor]:
    if features.ndim != 3 or padding_mask.shape != features.shape[:2]:
        raise ValueError("expected features [B,T,D] and padding_mask [B,T]")
    if padding_mask.dtype != torch.bool or not padding_mask.any(dim=1).all():
        raise ValueError("every sample needs at least one valid timestep")
    scores = scorer(features).squeeze(-1).masked_fill(~padding_mask,
        -torch.inf)
    weights = torch.softmax(scores, dim=1)
    pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
    return pooled, weights

def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pad = batch["padding_mask"]
    mean_representations = {
        m: self.modality_projection[m](masked_mean_pool(batch[m], pad))
        for m in MODALITIES
    }
    pooled = []
    for modality in MODALITIES:
        auxiliary, _ = masked_attention_pool(
            batch[modality], pad, self.attention_scorer[modality])
        auxiliary_representation = self.modality_projection[modality](auxiliary)
        pooled.append(mean_representations[modality] +
                      self.gamma[modality] * auxiliary_representation)
    fused = self.fusion(torch.cat(pooled, dim=-1))
    return {"classification_logits": self.classification_head(fused),
            "regression": self.regression_head(fused).squeeze(-1)}
