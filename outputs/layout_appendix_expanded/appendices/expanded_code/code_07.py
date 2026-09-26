def train_variant(variant: str, seed: int, cfg: dict, datasets: dict,
                  valid_loader, source_ckpt: dict, class_weights: torch.Tensor,
                  device: torch.device, output_dir: Path) -> dict:
    source_state = source_ckpt["model_state_dict"]
    model = make_model(variant, cfg, seed, source_state, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{variant} has no trainable incremental parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    train_loader = fresh_train_loader(
        datasets["train"], int(cfg["data"]["batch_size"]), seed
    )

    best_score, best_epoch, stale = -float("inf"), 0, 0
    history = []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        keep_b0_eval(model)
        loss_total, seen = 0.0, 0
        for batch in train_loader:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: moved[k] for k in (*MODALITIES, "padding_mask")}
            output = model(inputs)
            loss = multitask_loss(
                output, moved, lambda_reg=float(cfg["training"]["lambda_reg"]),
                class_weights=class_weights,
            )["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at {variant} seed{seed} epoch{epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in trainable):
                raise FloatingPointError(f"nonfinite gradient at {variant} seed{seed} epoch{epoch}")
            optimizer.step()
            n = int(moved["cls_label"].shape[0])
            loss_total += float(loss.item()) * n
            seen += n

        valid = clean_evaluate(model, valid_loader, device)
        score = valid["selection_score"]
        row = {"epoch": epoch, "train_loss": loss_total / seen,
               "valid": valid, "selection_score": score}
        history.append(row)
