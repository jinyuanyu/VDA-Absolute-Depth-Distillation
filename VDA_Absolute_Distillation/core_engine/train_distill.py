from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_engine.dataset import ScaleLabelDataset, simple_collate, split_dataset_indices
from models.modified_vda import SingleImageVDA
from models.scale_mlp_head import ScaleMLPHead


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_feature_batch(vda_model: SingleImageVDA, batch_items: List[Dict[str, Any]]) -> torch.Tensor:
    images = [vda_model.load_rgb_image(item["image_path"]) for item in batch_items]
    batch_tensor, _ = vda_model.prepare_batch(images)
    return vda_model.extract_global_features(batch_tensor)


def build_target_batch(batch_items: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    target = torch.tensor(
        [[float(item["scale"]), float(item["shift"])] for item in batch_items],
        dtype=torch.float32,
        device=device,
    )
    return target


def run_epoch(
    loader: DataLoader,
    vda_model: SingleImageVDA,
    scale_head: ScaleMLPHead,
    optimizer: AdamW | None,
    lambda_s: float,
    lambda_t: float,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    scale_head.train(is_train)
    total_loss = 0.0
    total_items = 0

    for batch_items in loader:
        features = extract_feature_batch(vda_model, batch_items).to(device)
        target = build_target_batch(batch_items, device)
        prediction = scale_head(features)
        loss_s = nn.functional.mse_loss(prediction[:, 0], target[:, 0])
        loss_t = nn.functional.mse_loss(prediction[:, 1], target[:, 1])
        loss = lambda_s * loss_s + lambda_t * loss_t

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = len(batch_items)
        total_items += batch_size
        total_loss += float(loss.item()) * batch_size

    return {"loss": total_loss / max(total_items, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scale MLP head with frozen VDA features.")
    parser.add_argument("--config", required=True, help="Path to distill_config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default="default_run")
    parser.add_argument(
        "--resume-from-last",
        action="store_true",
        help="Resume from runs/<run-name>/last.pt when available.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))

    device = torch.device(args.device)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    scale_cfg = cfg["scale_head"]
    preprocess_cfg = cfg["preprocess"]

    labels_json = Path(paths["labels_json"])
    dataset = ScaleLabelDataset(
        labels_json=labels_json,
        allowed_cams=[cam for cam in data_cfg["allowed_cams"] if cam not in set(data_cfg.get("blacklist_cams", []))],
    )
    train_indices, val_indices = split_dataset_indices(
        len(dataset),
        train_ratio=float(train_cfg["train_split"]),
        seed=int(cfg["project"]["seed"]),
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=simple_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=simple_collate,
    )

    vda_model = SingleImageVDA(
        repo_root=paths["vda_repo_root"],
        checkpoint_path=paths["vda_checkpoint"],
        encoder=model_cfg["encoder"],
        features=int(model_cfg["features"]),
        out_channels=model_cfg["out_channels"],
        input_size=int(preprocess_cfg["input_size"]),
        max_res=int(preprocess_cfg["max_res"]),
        device=device,
    )
    if model_cfg.get("freeze_backbone", True):
        vda_model.freeze_backbone()
    if model_cfg.get("freeze_decoder", True):
        vda_model.freeze_decoder()
    vda_model.eval()

    scale_head = ScaleMLPHead(
        input_dim=int(scale_cfg["input_dim"]),
        hidden_dims=scale_cfg["hidden_dims"],
        dropout=float(scale_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = AdamW(
        scale_head.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    run_root = Path(paths["runs_root"]) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    start_epoch = 1

    if args.resume_from_last:
        last_ckpt_path = run_root / "last.pt"
        if last_ckpt_path.exists():
            last_ckpt = torch.load(last_ckpt_path, map_location=device)
            scale_head.load_state_dict(last_ckpt["scale_head_state"], strict=True)
            history = list(last_ckpt.get("history", []))
            if history:
                best_val = min(float(item["val_loss"]) for item in history)
            start_epoch = int(last_ckpt.get("epoch", 0)) + 1
            print(
                json.dumps(
                    {
                        "resume": True,
                        "from_epoch": int(last_ckpt.get("epoch", 0)),
                        "next_epoch": start_epoch,
                        "best_val_so_far": best_val,
                    }
                )
            )

    total_epochs = int(train_cfg["epochs"])
    if start_epoch > total_epochs:
        print(
            json.dumps(
                {
                    "resume": True,
                    "message": "Training already reached configured total epochs.",
                    "configured_total_epochs": total_epochs,
                    "start_epoch": start_epoch,
                }
            )
        )
        (run_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return

    for epoch in range(start_epoch, total_epochs + 1):
        train_stats = run_epoch(
            train_loader,
            vda_model,
            scale_head,
            optimizer,
            float(train_cfg["lambda_s"]),
            float(train_cfg["lambda_t"]),
            device,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                val_loader,
                vda_model,
                scale_head,
                None,
                float(train_cfg["lambda_s"]),
                float(train_cfg["lambda_t"]),
                device,
            )
        record = {"epoch": epoch, "train_loss": train_stats["loss"], "val_loss": val_stats["loss"]}
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "epoch": epoch,
            "scale_head_state": scale_head.state_dict(),
            "feature_dim": vda_model.feature_dim,
            "encoder": vda_model.encoder_name,
            "config": cfg,
            "history": history,
        }
        torch.save(checkpoint, run_root / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(checkpoint, run_root / "best.pt")

    (run_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
