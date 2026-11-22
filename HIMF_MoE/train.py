#!/usr/bin/env python3
"""Train HIMF-MoE: a standalone hierarchy-aware multimodal fusion model.

One model jointly learns three related targets from the fixed train/val/test.json splits:
  1) label:   text authenticity (text-only prediction head)
  2) mlabel:  overall real/fake news (multimodal prediction head)
  3) forgery: 0/101/102/110/111 fine-grained forgery type

The script is self-contained and does not import the baseline implementation.
It uses multi-level ResNet18 features, multi-view BERT text features, soft
interaction experts, supervised contrastive learning, and exact label-hierarchy
projections tailored to the 0/101/102/110/111 forgery taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import csv
from resume_state import atomic_save, save_training, restore_training
import json
import logging
import math
import random
import re
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("hmcf_net")
FORGERY_CODES = (0, 101, 102, 110, 111)
FORGERY_TO_INDEX = {code: index for index, code in enumerate(FORGERY_CODES)}
INDEX_TO_FORGERY = {index: code for code, index in FORGERY_TO_INDEX.items()}
IMAGE_STATE_CODES = (0, 1, 2)  # clean, locally edited, replaced
ROUTER_STATE_CODES = (0, 1, 2, 3)  # clean, image-only, text-only, joint
TEMPLATE_KEYS = (
    "auc",
    "spauc",
    "metric",
    "f1_real",
    "f1_fake",
    "recall",
    "recall_real",
    "recall_fake",
    "precision",
    "precision_real",
    "precision_fake",
    "acc",
)


def add_boolean_argument(
    parser: argparse.ArgumentParser, name: str, default: bool
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true")
    group.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--image-root", type=Path, default=ROOT / "images")
    parser.add_argument("--bert-path", type=Path, default=ROOT / "pretrained/chinese-bert-wwm-ext")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/joint/himf_moe/seed_2026")
    parser.add_argument("--text-fields", nargs="+", default=["content"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bert-lr", type=float, default=2e-5)
    parser.add_argument("--image-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--forgery-loss-weight", type=float, default=1.5)
    parser.add_argument("--image-loss-weight", type=float, default=0.5)
    parser.add_argument("--router-loss-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--balance-loss-weight", type=float, default=0.02)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--modality-dropout", type=float, default=0.10)
    parser.add_argument("--baseline-val-text", type=float, default=0.860543)
    parser.add_argument("--baseline-val-binary", type=float, default=0.686559)
    parser.add_argument("--baseline-val-forgery", type=float, default=0.536687)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--spauc-max-fpr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    add_boolean_argument(parser, "require-valid-image", default=False)
    add_boolean_argument(parser, "image-pretrained", default=True)
    add_boolean_argument(parser, "amp", default=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fresh", action="store_true", help="Require empty output; never overwrites results")
    parser.add_argument("--plan",action="store_true",help="Print configuration without training")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_root}")
    if not args.bert_path.is_dir():
        raise FileNotFoundError(f"Local BERT directory not found: {args.bert_path}")
    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Required data file not found: {path}")
    if args.hidden_dim % args.attention_heads != 0:
        raise ValueError("--hidden-dim must be divisible by --attention-heads")
    if not 0 <= args.modality_dropout < 1:
        raise ValueError("--modality-dropout must be in [0, 1)")
    if not 0 < args.spauc_max_fpr < 1:
        raise ValueError("--spauc-max-fpr must be between 0 and 1")
    if args.contrastive_temperature <= 0:
        raise ValueError("--contrastive-temperature must be positive")
    if min(
        args.baseline_val_text,
        args.baseline_val_binary,
        args.baseline_val_forgery,
    ) <= 0:
        raise ValueError("Validation baseline reference scores must be positive")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(value), stream, ensure_ascii=False, indent=2)


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return records if limit is None else records[:limit]


def build_text(record: dict[str, Any], fields: Iterable[str]) -> str:
    parts = []
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    return " [SEP] ".join(parts)


def numeric_key(value: str) -> tuple[int, str]:
    matches = re.findall(r"\d+", value)
    return (int(matches[-1]) if matches else 0, value)


def image_names(record: dict[str, Any]) -> list[str]:
    images = record.get("images")
    if isinstance(images, dict):
        pairs = sorted(images.items(), key=lambda pair: numeric_key(str(pair[0])))
        return [str(value) for _, value in pairs if value]
    if isinstance(images, list):
        return [str(value) for value in images if value]
    if isinstance(images, str) and images:
        return [images]
    return []


def has_existing_image(record: dict[str, Any], image_root: Path) -> bool:
    return any((image_root / name).is_file() for name in image_names(record))


def image_transform(image_size: int, train: bool) -> transforms.Compose:
    resize_size = int(round(image_size * 256 / 224))
    crop = transforms.RandomCrop(image_size) if train else transforms.CenterCrop(image_size)
    operations: list[Any] = [transforms.Resize(resize_size), crop]
    if train:
        operations.append(
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return transforms.Compose(operations)


def derived_image_state(forgery_code: int) -> int:
    """Map forgery code to clean/local-edit/replacement image supervision."""
    if forgery_code in (0, 110):
        return 0
    if forgery_code in (101, 111):
        return 1
    if forgery_code == 102:
        return 2
    raise ValueError(f"Unsupported forgery code: {forgery_code}")


def derived_router_state(forgery_code: int) -> int:
    """Map forgery code to clean/image-only/text-only/joint interaction."""
    if forgery_code == 0:
        return 0
    if forgery_code in (101, 102):
        return 1
    if forgery_code == 110:
        return 2
    if forgery_code == 111:
        return 3
    raise ValueError(f"Unsupported forgery code: {forgery_code}")


class MultiTaskNewsDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        text_fields: list[str],
        max_length: int,
        image_root: Path,
        max_images: int,
        image_size: int,
        train: bool,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.text_fields = text_fields
        self.max_length = max_length
        self.image_root = image_root
        self.max_images = max_images
        self.image_size = image_size
        self.transform = image_transform(image_size, train)
        bad_codes = sorted(
            {int(record["forgery"]) for record in records}
            - set(FORGERY_CODES)
        )
        if bad_codes:
            raise ValueError(f"Unsupported forgery labels: {bad_codes}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_images(self, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tensors: list[torch.Tensor] = []
        mask: list[float] = []
        existing_names = [
            name for name in image_names(record) if (self.image_root / name).is_file()
        ]
        for relative_name in existing_names[: self.max_images]:
            path = self.image_root / relative_name
            try:
                with Image.open(path) as image:
                    tensors.append(self.transform(image.convert("RGB")))
                mask.append(1.0)
            except (FileNotFoundError, OSError, ValueError):
                LOGGER.warning("Unreadable image replaced by mask: %s", path)
                tensors.append(torch.zeros(3, self.image_size, self.image_size))
                mask.append(0.0)
        while len(tensors) < self.max_images:
            tensors.append(torch.zeros(3, self.image_size, self.image_size))
            mask.append(0.0)
        return torch.stack(tensors), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        forgery_code = int(record["forgery"])
        encoded = self.tokenizer(
            build_text(record, self.text_fields),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        images, image_mask = self._load_images(record)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "images": images,
            "image_mask": image_mask,
            "text_label": torch.tensor(int(record["label"]), dtype=torch.long),
            "binary_label": torch.tensor(int(record["mlabel"]), dtype=torch.long),
            "forgery_label": torch.tensor(
                FORGERY_TO_INDEX[forgery_code], dtype=torch.long
            ),
            "image_state_label": torch.tensor(
                derived_image_state(forgery_code), dtype=torch.long
            ),
            "router_label": torch.tensor(
                derived_router_state(forgery_code), dtype=torch.long
            ),
            "sample_id": str(record.get("id", index)),
        }


def make_resnet18(pretrained: bool) -> nn.Module:
    network=models.resnet18(weights=None)
    if pretrained:
        path=ROOT / 'pretrained/resnet18/resnet18-f37072fd.pth'
        if not path.is_file():raise FileNotFoundError('Local ResNet18 checkpoint required: '+str(path))
        network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    return network


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(features.dtype).unsqueeze(-1)
    return (features * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)


class MultiScaleResNet18(nn.Module):
    """Return low-level pattern tokens and high-level semantic tokens."""

    def __init__(self, pretrained: bool, hidden_dim: int, grid_size: int = 4) -> None:
        super().__init__()
        network = make_resnet18(pretrained)
        self.stem = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.pattern_project = nn.Sequential(
            nn.Linear(128, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.semantic_project = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.spatial_count = grid_size * grid_size

    def _tokens(
        self,
        feature_map: torch.Tensor,
        projector: nn.Module,
        batch_size: int,
        image_count: int,
    ) -> torch.Tensor:
        pooled = self.pool(feature_map).flatten(2).transpose(1, 2)
        projected = projector(pooled)
        return projected.reshape(
            batch_size, image_count * self.spatial_count, projected.shape[-1]
        )

    def forward(
        self, images: torch.Tensor, image_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, image_count, channels, height, width = images.shape
        features = self.stem(images.reshape(-1, channels, height, width))
        features = self.layer1(features)
        pattern_map = self.layer2(features)
        features = self.layer3(pattern_map)
        semantic_map = self.layer4(features)
        pattern_tokens = self._tokens(
            pattern_map, self.pattern_project, batch_size, image_count
        )
        semantic_tokens = self._tokens(
            semantic_map, self.semantic_project, batch_size, image_count
        )
        token_mask = image_mask.bool().repeat_interleave(self.spatial_count, dim=1)
        has_image = token_mask.any(dim=1, keepdim=True).to(pattern_tokens.dtype)

        # MultiheadAttention cannot receive an all-masked key sequence.
        all_missing = ~token_mask.any(dim=1)
        if all_missing.any():
            pattern_tokens = pattern_tokens.clone()
            semantic_tokens = semantic_tokens.clone()
            token_mask = token_mask.clone()
            pattern_tokens[all_missing] = 0.0
            semantic_tokens[all_missing] = 0.0
            token_mask[all_missing, 0] = True
        return pattern_tokens, semantic_tokens, token_mask, has_image


def probability_js_divergence(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    midpoint = 0.5 * (first + second)
    first_term = F.kl_div(
        midpoint.clamp_min(1e-8).log(), first, reduction="none"
    ).sum(dim=-1, keepdim=True)
    second_term = F.kl_div(
        midpoint.clamp_min(1e-8).log(), second, reduction="none"
    ).sum(dim=-1, keepdim=True)
    return 0.5 * (first_term + second_term)


class HIMFMoE(nn.Module):
    """Hierarchy-aware Interaction and Multi-view Fusion Mixture-of-Experts."""

    def __init__(
        self,
        bert_path: Path,
        hidden_dim: int,
        attention_heads: int,
        dropout: float,
        modality_dropout: float,
        image_pretrained: bool,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(str(bert_path), local_files_only=True)
        bert_dim = int(self.bert.config.hidden_size)
        self.text_project = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        cnn_channels = max(32, hidden_dim // 4)
        self.text_pool_score = nn.Linear(hidden_dim, 1)
        self.text_convs = nn.ModuleList(
            nn.Conv1d(hidden_dim, cnn_channels, kernel_size, padding=kernel_size // 2)
            for kernel_size in (2, 3, 4)
        )
        self.text_view_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cnn_channels * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.image_encoder = MultiScaleResNet18(image_pretrained, hidden_dim)
        self.image_view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.text_to_image = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_cross_norm = nn.LayerNorm(hidden_dim)
        self.image_cross_norm = nn.LayerNorm(hidden_dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.text_ffn_norm = nn.LayerNorm(hidden_dim)
        self.image_ffn_norm = nn.LayerNorm(hidden_dim)
        self.modality_dropout = modality_dropout
        self.text_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2)
        )
        self.image_state_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(IMAGE_STATE_CODES)),
        )
        router_input_dim = hidden_dim * 4 + 7
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ROUTER_STATE_CODES)),
        )
        expert_input_dim = hidden_dim * 6 + 3
        self.fusion_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(expert_input_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in ROUTER_STATE_CODES
        )
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.forgery_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 14, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, len(FORGERY_CODES)),
        )

    def _text_views(
        self, text_tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        scores = self.text_pool_score(text_tokens).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        attention_feature = torch.sum(
            torch.softmax(scores, dim=1).unsqueeze(-1) * text_tokens, dim=1
        )
        sequence_length = text_tokens.shape[1]
        convolution_input = text_tokens.transpose(1, 2)
        local_features = []
        for convolution in self.text_convs:
            convolved = F.gelu(convolution(convolution_input))[..., :sequence_length]
            convolved = convolved.masked_fill(~mask.unsqueeze(1), -1e4)
            local_features.append(convolved.max(dim=-1).values)
        return self.text_view_fusion(
            torch.cat(
                (text_tokens[:, 0], attention_feature, *local_features), dim=-1
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bert_tokens = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_tokens = self.text_project(bert_tokens)
        text_padding_mask = ~attention_mask.bool()
        text_feature = self._text_views(text_tokens, attention_mask)

        pattern_tokens, semantic_tokens, image_token_mask, has_image = self.image_encoder(
            images, image_mask
        )
        pattern_feature = masked_mean(pattern_tokens, image_token_mask)
        semantic_feature = masked_mean(semantic_tokens, image_token_mask)
        image_view_input = torch.cat(
            (
                pattern_feature,
                semantic_feature,
                torch.abs(pattern_feature - semantic_feature),
                pattern_feature * semantic_feature,
            ),
            dim=-1,
        )
        image_view_weights = torch.softmax(self.image_view_gate(image_view_input), dim=-1)
        image_feature = (
            image_view_weights[:, :1] * pattern_feature
            + image_view_weights[:, 1:] * semantic_feature
        )
        image_tokens = torch.cat((pattern_tokens, semantic_tokens), dim=1)
        combined_image_mask = torch.cat((image_token_mask, image_token_mask), dim=1)
        image_padding_mask = ~combined_image_mask
        text_context, _ = self.text_to_image(
            text_tokens,
            image_tokens,
            image_tokens,
            key_padding_mask=image_padding_mask,
            need_weights=False,
        )
        image_context, _ = self.image_to_text(
            image_tokens,
            text_tokens,
            text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        text_cross = self.text_cross_norm(text_tokens + text_context)
        image_cross = self.image_cross_norm(image_tokens + image_context)
        text_cross = self.text_ffn_norm(text_cross + self.text_ffn(text_cross))
        image_cross = self.image_ffn_norm(image_cross + self.image_ffn(image_cross))
        cross_text_feature = masked_mean(text_cross, attention_mask.bool())
        cross_image_feature = masked_mean(image_cross, combined_image_mask)

        if self.training and self.modality_dropout > 0:
            keep_image = (
                torch.rand_like(has_image) >= self.modality_dropout
            ).to(has_image.dtype)
            has_image = has_image * keep_image
            image_feature = image_feature * keep_image
            pattern_feature = pattern_feature * keep_image
            semantic_feature = semantic_feature * keep_image
            cross_image_feature = cross_image_feature * keep_image
            cross_text_feature = (
                keep_image * cross_text_feature + (1.0 - keep_image) * text_feature
            )

        difference = torch.abs(text_feature - image_feature)
        product = text_feature * image_feature
        alignment = F.cosine_similarity(
            text_feature, semantic_feature, dim=-1
        ).unsqueeze(-1) * has_image
        text_logits = self.text_head(text_feature)
        image_state_logits = self.image_state_head(
            (
                torch.cat(
                    (
                        pattern_feature,
                        semantic_feature,
                        torch.abs(pattern_feature - semantic_feature),
                        pattern_feature * semantic_feature,
                        has_image,
                    ),
                    dim=-1,
                )
            )
        )
        text_probabilities = torch.softmax(text_logits.float(), dim=-1)
        image_state_probabilities = torch.softmax(
            image_state_logits.float(), dim=-1
        )
        image_binary_probabilities = torch.stack(
            (
                image_state_probabilities[:, 0],
                image_state_probabilities[:, 1:].sum(dim=-1),
            ),
            dim=-1,
        )
        unimodal_js = probability_js_divergence(
            text_probabilities, image_binary_probabilities
        ) * has_image
        router_input = torch.cat(
            (
                text_feature,
                image_feature,
                difference,
                product,
                text_probabilities,
                image_binary_probabilities,
                alignment,
                unimodal_js,
                has_image,
            ),
            dim=-1,
        )
        router_logits = self.router(router_input)
        router_probabilities = torch.softmax(router_logits, dim=-1)
        expert_input = torch.cat(
            (
                text_feature,
                image_feature,
                cross_text_feature,
                cross_image_feature,
                difference,
                product,
                alignment,
                unimodal_js,
                has_image,
            ),
            dim=-1,
        )
        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.fusion_experts], dim=1
        )
        fused = torch.sum(router_probabilities.unsqueeze(-1) * expert_outputs, dim=1)
        binary_logits = self.binary_head(
            torch.cat(
                (
                    fused,
                    text_feature,
                    image_feature,
                    alignment,
                    unimodal_js,
                    has_image,
                ),
                dim=-1,
            )
        )
        hierarchy_signal = torch.cat(
            (
                fused,
                text_feature,
                image_feature,
                text_logits,
                image_state_logits,
                binary_logits,
                router_logits,
                alignment,
                unimodal_js,
                has_image,
            ),
            dim=-1,
        )
        forgery_logits = self.forgery_head(hierarchy_signal)
        forgery_probabilities = torch.softmax(forgery_logits.float(), dim=-1)
        hierarchy_binary_probabilities = torch.stack(
            (
                forgery_probabilities[:, FORGERY_TO_INDEX[0]],
                1.0 - forgery_probabilities[:, FORGERY_TO_INDEX[0]],
            ),
            dim=-1,
        )
        binary_probabilities = 0.5 * (
            torch.softmax(binary_logits.float(), dim=-1)
            + hierarchy_binary_probabilities
        )
        return {
            "text_logits": text_logits,
            "binary_logits": binary_logits,
            "forgery_logits": forgery_logits,
            "image_state_logits": image_state_logits,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "binary_probabilities": binary_probabilities,
            "fused_feature": fused,
            "alignment": alignment.squeeze(-1),
            "unimodal_js": unimodal_js.squeeze(-1),
            "image_view_weights": image_view_weights,
        }


def filter_records(
    records: list[dict[str, Any]], image_root: Path, require_valid_image: bool
) -> list[dict[str, Any]]:
    if not require_valid_image:
        return records
    return [record for record in records if has_existing_image(record, image_root)]


def make_loaders(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[dict[str, DataLoader], list[dict[str, Any]]]:
    limits = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    loaders: dict[str, DataLoader] = {}
    train_records: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.json"
        records = load_records(path, limits[split])
        original_count = len(records)
        records = filter_records(records, args.image_root, args.require_valid_image)
        LOGGER.info(
            "%s: kept=%d removed=%d from %s",
            split,
            len(records),
            original_count - len(records),
            path,
        )
        if split == "train":
            train_records = records
        dataset = MultiTaskNewsDataset(
            records,
            tokenizer,
            args.text_fields,
            args.max_length,
            args.image_root,
            args.max_images,
            args.image_size,
            train=split == "train",
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders, train_records


def inverse_frequency_weights(
    records: list[dict[str, Any]], field: str, codes: tuple[int, ...]
) -> torch.Tensor:
    counts = Counter(int(record[field]) for record in records)
    missing = [code for code in codes if counts[code] == 0]
    if missing:
        raise ValueError(
            f"Training subset is missing {field} labels {missing}; "
            "increase --max-train-samples"
        )
    total = len(records)
    return torch.tensor(
        [total / (len(codes) * counts[code]) for code in codes],
        dtype=torch.float32,
    )


def derived_frequency_weights(
    records: list[dict[str, Any]], mapper: Any, codes: tuple[int, ...]
) -> torch.Tensor:
    counts = Counter(mapper(int(record["forgery"])) for record in records)
    missing = [code for code in codes if counts[code] == 0]
    if missing:
        raise ValueError(f"Training subset is missing derived labels {missing}")
    total = len(records)
    return torch.tensor(
        [total / (len(codes) * counts[code]) for code in codes],
        dtype=torch.float32,
    )


def make_criteria(
    records: list[dict[str, Any]], args: argparse.Namespace, device: torch.device
) -> dict[str, nn.Module]:
    return {
        "text": nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(records, "label", (0, 1)).to(device),
            label_smoothing=args.label_smoothing,
        ),
        "binary": nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(records, "mlabel", (0, 1)).to(device),
            label_smoothing=args.label_smoothing,
        ),
        "forgery": nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(records, "forgery", FORGERY_CODES).to(device),
            label_smoothing=args.label_smoothing,
        ),
        "image": nn.CrossEntropyLoss(
            weight=derived_frequency_weights(
                records, derived_image_state, IMAGE_STATE_CODES
            ).to(device),
            label_smoothing=args.label_smoothing,
        ),
        "router": nn.CrossEntropyLoss(
            weight=derived_frequency_weights(
                records, derived_router_state, ROUTER_STATE_CODES
            ).to(device),
            label_smoothing=args.label_smoothing,
        ),
    }


def symmetric_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    return 0.5 * (
        F.kl_div(first.log(), second, reduction="batchmean")
        + F.kl_div(second.log(), first, reduction="batchmean")
    )


def hierarchy_consistency_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    text_probability = torch.softmax(outputs["text_logits"].float(), dim=-1)
    binary_probability = torch.softmax(outputs["binary_logits"].float(), dim=-1)
    image_probability = torch.softmax(outputs["image_state_logits"].float(), dim=-1)
    forgery_probability = torch.softmax(outputs["forgery_logits"].float(), dim=-1)
    projected_text = torch.stack(
        (
            forgery_probability[:, FORGERY_TO_INDEX[0]]
            + forgery_probability[:, FORGERY_TO_INDEX[101]]
            + forgery_probability[:, FORGERY_TO_INDEX[102]],
            forgery_probability[:, FORGERY_TO_INDEX[110]]
            + forgery_probability[:, FORGERY_TO_INDEX[111]],
        ),
        dim=-1,
    )
    projected_binary = torch.stack(
        (
            forgery_probability[:, FORGERY_TO_INDEX[0]],
            1.0 - forgery_probability[:, FORGERY_TO_INDEX[0]],
        ),
        dim=-1,
    )
    projected_image = torch.stack(
        (
            forgery_probability[:, FORGERY_TO_INDEX[0]]
            + forgery_probability[:, FORGERY_TO_INDEX[110]],
            forgery_probability[:, FORGERY_TO_INDEX[101]]
            + forgery_probability[:, FORGERY_TO_INDEX[111]],
            forgery_probability[:, FORGERY_TO_INDEX[102]],
        ),
        dim=-1,
    )
    return (
        symmetric_kl(text_probability, projected_text)
        + symmetric_kl(binary_probability, projected_binary)
        + symmetric_kl(image_probability, projected_image)
    )


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    logits = features @ features.transpose(0, 1) / temperature
    batch_size = features.shape[0]
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~diagonal
    denominator_mask = ~diagonal
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * denominator_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
    )
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_log_probability = (
        (positive_mask * log_probability).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -mean_log_probability[valid].mean()


def router_balance_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    probabilities = outputs["router_probabilities"].float()
    target = torch.full_like(probabilities.mean(dim=0), 1.0 / probabilities.shape[1])
    load_loss = F.mse_loss(probabilities.mean(dim=0), target)
    router_z = torch.logsumexp(
        outputs["router_logits"].float(), dim=-1
    ).square().mean()
    return load_loss + 0.01 * router_z


def total_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    criteria: dict[str, nn.Module],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    text_loss = criteria["text"](outputs["text_logits"], batch["text_label"])
    binary_loss = criteria["binary"](
        outputs["binary_logits"], batch["binary_label"]
    )
    forgery_loss = criteria["forgery"](
        outputs["forgery_logits"], batch["forgery_label"]
    )
    image_loss = criteria["image"](
        outputs["image_state_logits"], batch["image_state_label"]
    )
    router_loss = criteria["router"](
        outputs["router_logits"], batch["router_label"]
    )
    contrastive_loss = supervised_contrastive_loss(
        outputs["fused_feature"],
        batch["forgery_label"],
        args.contrastive_temperature,
    )
    consistency_loss = hierarchy_consistency_loss(outputs)
    balance_loss = router_balance_loss(outputs)
    loss = (
        args.text_loss_weight * text_loss
        + args.binary_loss_weight * binary_loss
        + args.forgery_loss_weight * forgery_loss
        + args.image_loss_weight * image_loss
        + args.router_loss_weight * router_loss
        + args.contrastive_loss_weight * contrastive_loss
        + args.consistency_loss_weight * consistency_loss
        + args.balance_loss_weight * balance_loss
    )
    parts = {
        "total": float(loss.detach()),
        "text": float(text_loss.detach()),
        "binary": float(binary_loss.detach()),
        "forgery": float(forgery_loss.detach()),
        "image": float(image_loss.detach()),
        "router": float(router_loss.detach()),
        "contrastive": float(contrastive_loss.detach()),
        "consistency": float(consistency_loss.detach()),
        "balance": float(balance_loss.detach()),
    }
    return loss, parts


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in ("input_ids", "attention_mask", "images", "image_mask")
    }


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = {"bert": [], "image": [], "head": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("bert."):
            groups["bert"].append(parameter)
        elif name.startswith("image_encoder."):
            groups["image"].append(parameter)
        else:
            groups["head"].append(parameter)
    parameter_groups = []
    for key, learning_rate in (
        ("bert", args.bert_lr),
        ("image", args.image_lr),
        ("head", args.head_lr),
    ):
        parameter_groups.append(
            {"params": groups[key], "lr": learning_rate, "name": key}
        )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def make_scaler(enabled: bool) -> Any:
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    amp_enabled: bool,
    grad_clip: float,
) -> None:
    if amp_enabled:
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer_ran = scaler.get_scale() >= scale_before
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer_ran = True
    # Do not advance the schedule when AMP skipped an overflowing optimizer step.
    if optimizer_ran:
        scheduler.step()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criteria: dict[str, nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    totals = {
        key: 0.0
        for key in (
            "total",
            "text",
            "binary",
            "forgery",
            "image",
            "router",
            "contrastive",
            "consistency",
            "balance",
        )
    }
    sample_count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            outputs = model(**forward_inputs(batch))
            loss, parts = total_loss(outputs, batch, criteria, args)
        optimizer_step(
            loss,
            model,
            optimizer,
            scheduler,
            scaler,
            amp_enabled,
            args.grad_clip,
        )
        batch_size = batch["text_label"].shape[0]
        sample_count += batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def safe_auc(
    labels: np.ndarray, scores: np.ndarray, max_fpr: float | None = None
) -> float:
    try:
        kwargs = {} if max_fpr is None else {"max_fpr": max_fpr}
        return float(roc_auc_score(labels, scores, **kwargs))
    except ValueError:
        return float("nan")


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, max_fpr: float
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    result = {
        "auc": safe_auc(labels, probabilities[:, 1]),
        "spauc": safe_auc(labels, probabilities[:, 1], max_fpr=max_fpr),
        "metric": float(np.mean(f1)),
        "f1_real": float(f1[0]),
        "f1_fake": float(f1[1]),
        "recall": float(np.mean(recall)),
        "recall_real": float(recall[0]),
        "recall_fake": float(recall[1]),
        "precision": float(np.mean(precision)),
        "precision_real": float(precision[0]),
        "precision_fake": float(precision[1]),
        "acc": float(accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return result


def forgery_metrics(
    labels: np.ndarray, probabilities: np.ndarray, max_fpr: float
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    binary_labels = (labels != FORGERY_TO_INDEX[0]).astype(np.int64)
    binary_predictions = (predictions != FORGERY_TO_INDEX[0]).astype(np.int64)
    binary_probability = 1.0 - probabilities[:, FORGERY_TO_INDEX[0]]
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels,
        binary_predictions,
        labels=[0, 1],
        zero_division=0,
    )
    result: dict[str, Any] = {
        "auc": safe_auc(binary_labels, binary_probability),
        "spauc": safe_auc(binary_labels, binary_probability, max_fpr=max_fpr),
        "metric": float(np.mean(f1)),
        "f1_real": float(f1[0]),
        "f1_fake": float(f1[1]),
        "recall": float(np.mean(recall)),
        "recall_real": float(recall[0]),
        "recall_fake": float(recall[1]),
        "precision": float(np.mean(precision)),
        "precision_real": float(precision[0]),
        "precision_fake": float(precision[1]),
        "acc": float(accuracy_score(binary_labels, binary_predictions)),
        "task_accuracy": float(accuracy_score(labels, predictions)),
        "task_macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "task_weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "label_codes": list(FORGERY_CODES),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(FORGERY_CODES)))
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(FORGERY_CODES))),
            target_names=[str(code) for code in FORGERY_CODES],
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        result["task_macro_auc_ovr"] = float(
            roc_auc_score(labels, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        result["task_macro_auc_ovr"] = float("nan")
    return result


def auxiliary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, names: list[str]
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    indices = list(range(len(names)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=indices
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=indices,
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_fpr: float,
    amp_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    sample_ids: list[str] = []
    labels = {
        key: [] for key in ("text", "binary", "forgery", "image", "router")
    }
    probabilities = {
        key: [] for key in ("text", "binary", "forgery", "image", "router")
    }
    alignments: list[np.ndarray] = []
    divergences: list[np.ndarray] = []
    image_view_weights: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            sample_ids.extend(batch["sample_id"])
            batch = batch_to_device(batch, device)
            with autocast_context(amp_enabled):
                outputs = model(**forward_inputs(batch))
            labels["text"].append(batch["text_label"].cpu().numpy())
            labels["binary"].append(batch["binary_label"].cpu().numpy())
            labels["forgery"].append(batch["forgery_label"].cpu().numpy())
            labels["image"].append(batch["image_state_label"].cpu().numpy())
            labels["router"].append(batch["router_label"].cpu().numpy())
            probabilities["text"].append(
                torch.softmax(outputs["text_logits"].float(), dim=-1).cpu().numpy()
            )
            probabilities["binary"].append(
                outputs["binary_probabilities"].float().cpu().numpy()
            )
            probabilities["forgery"].append(
                torch.softmax(outputs["forgery_logits"].float(), dim=-1).cpu().numpy()
            )
            probabilities["image"].append(
                torch.softmax(
                    outputs["image_state_logits"].float(), dim=-1
                ).cpu().numpy()
            )
            probabilities["router"].append(
                outputs["router_probabilities"].float().cpu().numpy()
            )
            alignments.append(outputs["alignment"].float().cpu().numpy())
            divergences.append(outputs["unimodal_js"].float().cpu().numpy())
            image_view_weights.append(
                outputs["image_view_weights"].float().cpu().numpy()
            )
    labels_np = {key: np.concatenate(value) for key, value in labels.items()}
    probabilities_np = {
        key: np.concatenate(value) for key, value in probabilities.items()
    }
    metrics = {
        "text_label": binary_metrics(
            labels_np["text"], probabilities_np["text"], max_fpr
        ),
        "multimodal_binary": binary_metrics(
            labels_np["binary"], probabilities_np["binary"], max_fpr
        ),
        "forgery5": forgery_metrics(
            labels_np["forgery"], probabilities_np["forgery"], max_fpr
        ),
        "image_state_aux": auxiliary_metrics(
            labels_np["image"],
            probabilities_np["image"],
            ["clean", "local_edit", "replacement"],
        ),
        "interaction_router_aux": auxiliary_metrics(
            labels_np["router"],
            probabilities_np["router"],
            ["clean", "image_only", "text_only", "joint"],
        ),
    }
    metrics["diagnostics"] = {
        "mean_alignment": float(np.concatenate(alignments).mean()),
        "mean_unimodal_js": float(np.concatenate(divergences).mean()),
        "mean_image_pattern_weight": float(
            np.concatenate(image_view_weights)[:, 0].mean()
        ),
        "mean_image_semantic_weight": float(
            np.concatenate(image_view_weights)[:, 1].mean()
        ),
        "mean_router_probabilities": np.concatenate(
            probabilities["router"]
        ).mean(axis=0).tolist(),
    }
    raw = {
        "sample_ids": sample_ids,
        "labels": labels_np,
        "probabilities": probabilities_np,
        "alignment": np.concatenate(alignments),
        "unimodal_js": np.concatenate(divergences),
        "image_view_weights": np.concatenate(image_view_weights),
    }
    return metrics, raw


def validation_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    ratios = np.asarray(
        [
            metrics["text_label"]["metric"] / args.baseline_val_text,
            metrics["multimodal_binary"]["metric"] / args.baseline_val_binary,
            metrics["forgery5"]["task_macro_f1"] / args.baseline_val_forgery,
        ],
        dtype=np.float64,
    )
    # The worst relative task is primary; mean ratio only breaks close ties.
    return float(ratios.min() + 0.01 * ratios.mean())


def write_predictions(path: Path, raw: dict[str, Any]) -> None:
    text_labels = raw["labels"]["text"]
    binary_labels = raw["labels"]["binary"]
    forgery_labels = raw["labels"]["forgery"]
    text_probabilities = raw["probabilities"]["text"]
    binary_probabilities = raw["probabilities"]["binary"]
    forgery_probabilities = raw["probabilities"]["forgery"]
    image_probabilities = raw["probabilities"]["image"]
    router_probabilities = raw["probabilities"]["router"]
    with path.open("w", encoding="utf-8") as stream:
        for index, sample_id in enumerate(raw["sample_ids"]):
            forgery_prediction = int(forgery_probabilities[index].argmax())
            record = {
                "id": sample_id,
                "label": int(text_labels[index]),
                "label_prediction": int(text_probabilities[index].argmax()),
                "label_probability": float(text_probabilities[index, 1]),
                "mlabel": int(binary_labels[index]),
                "mlabel_prediction": int(binary_probabilities[index].argmax()),
                "mlabel_probability": float(binary_probabilities[index, 1]),
                "forgery": INDEX_TO_FORGERY[int(forgery_labels[index])],
                "forgery_prediction": INDEX_TO_FORGERY[forgery_prediction],
                "forgery_probabilities": {
                    str(code): float(forgery_probabilities[index, code_index])
                    for code_index, code in enumerate(FORGERY_CODES)
                },
                "image_state": int(raw["labels"]["image"][index]),
                "image_state_prediction": int(
                    image_probabilities[index].argmax()
                ),
                "router_state": int(raw["labels"]["router"][index]),
                "router_prediction": int(
                    router_probabilities[index].argmax()
                ),
                "router_probabilities": router_probabilities[index].tolist(),
                "alignment": float(raw["alignment"][index]),
                "unimodal_js": float(raw["unimodal_js"][index]),
                "image_pattern_weight": float(
                    raw["image_view_weights"][index, 0]
                ),
                "image_semantic_weight": float(
                    raw["image_view_weights"][index, 1]
                ),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    score: float,
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": "HIMF-MoE",
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "validation_score": score,
        "validation_metrics": val_metrics,
        "forgery_codes": list(FORGERY_CODES),
        "config": json_ready(vars(args)),
    }


def main() -> None:
    args = parse_args()
    if args.plan:
        print(json.dumps(json_ready(vars(args)),indent=2));return
    validate_args(args)
    if args.num_workers!=0:raise ValueError('Exact epoch resume currently requires --num-workers 0')
    contract={k:v for k,v in json_ready(vars(args)).items() if k not in ['device','output_dir','fresh','eval_only','checkpoint']}
    contract['data_sha256']={split:hashlib.sha256((args.data_dir/f'{split}.json').read_bytes()).hexdigest() for split in ['train','val','test']}
    status_path=args.output_dir/'status.json'
    if args.fresh and args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError('Use a new output directory; existing files preserved')
    if not args.eval_only and status_path.is_file():
        status=json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('contract')!=contract:raise ValueError('Output belongs to different data/settings; use another --output-dir')
        if status.get('status')=='complete' and all((args.output_dir/name).is_file() for name in ['best_model.pt','test_metrics.json','test_predictions.jsonl','test_predictions.csv']):
            print('HIMF-MoE already complete: skipped',args.output_dir);return
    configure_logging(args.output_dir)
    save_json(status_path,{'status':'running','contract':contract})
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    LOGGER.info("model=HIMF-MoE device=%s amp=%s", device, amp_enabled)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "model_name": "HIMF-MoE",
            "tasks": {
                "text_label": "label",
                "multimodal_binary": "mlabel",
                "forgery5": list(FORGERY_CODES),
            },
            "validation_score": (
                "min(task_f1 / baseline_val_f1) + "
                "0.01*mean(task_f1 / baseline_val_f1)"
            ),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.bert_path), local_files_only=True
    )
    loaders, train_records = make_loaders(args, tokenizer)
    model = HIMFMoE(
        args.bert_path,
        args.hidden_dim,
        args.attention_heads,
        args.dropout,
        args.modality_dropout,
        args.image_pretrained,
    ).to(device)
    criteria = make_criteria(train_records, args, device)
    best_path = args.output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []

    if args.eval_only:
        checkpoint_path = args.checkpoint or best_path
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        LOGGER.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        optimizer = make_optimizer(model, args)
        total_steps = max(1, len(loaders["train"]) * args.epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        scaler = make_scaler(amp_enabled)
        best_score = -math.inf
        stale_epochs = 0
        last_path=args.output_dir/'last.pt'
        start_epoch=1
        if last_path.is_file() and not args.eval_only:
            resumed=restore_training(last_path,model,optimizer,scheduler,scaler,loaders['train'].generator,contract)
            start_epoch=resumed['epoch']+1;best_score=resumed['best_score'];stale_epochs=resumed['stale_epochs'];history=resumed['history']
            LOGGER.info('Resuming from completed epoch %d',resumed['epoch'])
            if resumed['training_finished']:start_epoch=args.epochs+1
        elif best_path.exists() and not args.eval_only:
            raise RuntimeError('Best checkpoint exists without last.pt; use --eval-only or a separate output directory')
        for epoch in range(start_epoch, args.epochs + 1):
            losses = train_epoch(
                model,
                loaders["train"],
                optimizer,
                scheduler,
                criteria,
                args,
                device,
                scaler,
                amp_enabled,
            )
            val_metrics, _ = evaluate(
                model,
                loaders["val"],
                device,
                args.spauc_max_fpr,
                amp_enabled,
            )
            score = validation_score(val_metrics, args)
            epoch_record = {
                "epoch": epoch,
                "train_loss": losses,
                "validation_score": score,
                "val": val_metrics,
            }
            history.append(epoch_record)
            save_json(args.output_dir / "history.json", history)
            save_json(
                args.output_dir / f"val_metrics_epoch_{epoch:02d}.json",
                val_metrics,
            )
            LOGGER.info(
                "epoch=%d loss=%.6f val_score=%.6f text_f1=%.6f "
                "binary_f1=%.6f forgery5_f1=%.6f",
                epoch,
                losses["total"],
                score,
                val_metrics["text_label"]["metric"],
                val_metrics["multimodal_binary"]["metric"],
                val_metrics["forgery5"]["task_macro_f1"],
            )
            if score > best_score:
                best_score = score
                stale_epochs = 0
                atomic_save(
                    checkpoint_payload(model, args, epoch, score, val_metrics),
                    best_path,
                )
                LOGGER.info("Saved new best checkpoint: %s", best_path)
            else:
                stale_epochs += 1
            warmup_epochs=math.ceil(args.epochs*args.warmup_ratio)
            if epoch<=warmup_epochs:stale_epochs=0
            finished=(args.patience>0 and stale_epochs>=args.patience) or epoch>=args.epochs
            save_training(last_path,model,optimizer,scheduler,scaler,loaders['train'].generator,epoch=epoch,best_score=best_score,stale_epochs=stale_epochs,history=history,contract=contract,training_finished=finished)
            if finished:break
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])

    test_metrics, raw = evaluate(
        model,
        loaders["test"],
        device,
        args.spauc_max_fpr,
        amp_enabled,
    )
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    with (args.output_dir / "test_metrics.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        for task_name in ("text_label", "multimodal_binary", "forgery5"):
            summary = {
                key: test_metrics[task_name][key] for key in TEMPLATE_KEYS
            }
            stream.write(f"{task_name}: {summary!r}\n")
        stream.write(
            "forgery5_native: "
            + repr(
                {
                    "task_macro_f1": test_metrics["forgery5"]["task_macro_f1"],
                    "task_accuracy": test_metrics["forgery5"]["task_accuracy"],
                    "task_macro_auc_ovr": test_metrics["forgery5"].get(
                        "task_macro_auc_ovr"
                    ),
                }
            )
            + "\n"
        )
    write_predictions(args.output_dir / "test_predictions.jsonl", raw)
    prediction_rows=[json.loads(line) for line in (args.output_dir/'test_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    flat_rows=[]
    for row in prediction_rows:
        flat={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        flat.update({f'forgery_probability_{code}':prob for code,prob in row['forgery_probabilities'].items()})
        flat_rows.append(flat)
    with (args.output_dir/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat_rows[0]));writer.writeheader();writer.writerows(flat_rows)
    save_json(status_path,{'status':'complete','contract':contract,'best_epoch':payload['epoch']})
    LOGGER.info(
        "test text_f1=%.6f binary_f1=%.6f forgery5_f1=%.6f",
        test_metrics["text_label"]["metric"],
        test_metrics["multimodal_binary"]["metric"],
        test_metrics["forgery5"]["task_macro_f1"],
    )


if __name__ == "__main__":
    main()
