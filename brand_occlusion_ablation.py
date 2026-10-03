import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
from tqdm import tqdm

from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Occlusion augmentations
# ============================================================

def clip_bbox(bbox, w, h):
    """
    bbox = [xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return x1, y1, x2, y2


def mask_rectangle(img, box, mode="black"):
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box

    if mode == "black":
        fill = (0, 0, 0)

    elif mode == "gray":
        fill = (127, 127, 127)

    elif mode == "mean":
        arr = np.asarray(img)
        mean = tuple(arr.reshape(-1, 3).mean(axis=0).astype(np.uint8))
        fill = mean

    else:
        raise ValueError(mode)

    draw.rectangle([x1, y1, x2, y2], fill=fill)

    return img


def expand_bbox(box, w, h, scale=1.0):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = int(cx - bw / 2)
    ny1 = int(cy - bh / 2)
    nx2 = int(cx + bw / 2)
    ny2 = int(cy + bh / 2)

    return clip_bbox([nx1, ny1, nx2, ny2], w, h)


def logo_mask(img, bbox, scale=1.25):
    """
    실제 logo bbox보다 약간 넓게 가린다.
    로고 주변 texture까지 shortcut으로 사용하는 것을 방지.
    """
    w, h = img.size

    box = clip_bbox(bbox, w, h)
    box = expand_bbox(box, w, h, scale)

    return mask_rectangle(img, box, mode="gray")


def random_erasing(img, min_area=0.05, max_area=0.25):
    img = img.copy().convert("RGB")

    w, h = img.size

    area = w * h
    target = random.uniform(min_area, max_area) * area

    aspect = random.uniform(0.5, 2.0)

    ew = int(np.sqrt(target * aspect))
    eh = int(np.sqrt(target / aspect))

    ew = min(ew, w - 1)
    eh = min(eh, h - 1)

    if ew <= 0 or eh <= 0:
        return img

    x1 = random.randint(0, max(0, w - ew))
    y1 = random.randint(0, max(0, h - eh))

    return mask_rectangle(
        img,
        (x1, y1, x1 + ew, y1 + eh),
        mode="gray"
    )


def cutout(img, ratio=0.3):
    img = img.copy().convert("RGB")

    w, h = img.size

    cw = max(1, int(w * ratio))
    ch = max(1, int(h * ratio))

    x1 = random.randint(0, max(0, w - cw))
    y1 = random.randint(0, max(0, h - ch))

    return mask_rectangle(
        img,
        (x1, y1, x1 + cw, y1 + ch),
        mode="gray"
    )


def gridmask(img, grid=4, ratio=0.45):
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size

    gw = max(1, w // grid)
    gh = max(1, h // grid)

    mw = max(1, int(gw * ratio))
    mh = max(1, int(gh * ratio))

    offset_x = random.randint(0, gw - 1)
    offset_y = random.randint(0, gh - 1)

    for x in range(-gw, w + gw, gw):
        for y in range(-gh, h + gh, gh):

            x1 = x + offset_x
            y1 = y + offset_y

            draw.rectangle(
                [x1, y1, x1 + mw, y1 + mh],
                fill=(127, 127, 127)
            )

    return img


def apply_augmentation(img, bbox, regime):
    if regime == "none":
        return img

    if regime == "random_erasing":
        return random_erasing(img)

    if regime == "cutout":
        return cutout(img)

    if regime == "gridmask":
        return gridmask(img)

    if regime == "logo_mask":
        # 항상 가리면 clean 성능이 떨어질 수 있으므로
        # 50% 확률로 적용
        if random.random() < 0.5:
            return logo_mask(img, bbox)

        return img

    if regime == "mixed":

        r = random.random()

        if r < 0.40:
            return img

        elif r < 0.60:
            return logo_mask(img, bbox)

        elif r < 0.75:
            return random_erasing(img)

        elif r < 0.90:
            return cutout(img)

        else:
            return gridmask(img)

    raise ValueError(f"Unknown regime: {regime}")


# ============================================================
# Dataset loading
# ============================================================

def get_field(row, candidates):
    for c in candidates:
        if c in row:
            return row[c]

    return None


def prepare_data(
    max_brands=100,
    max_per_brand=100,
    min_per_brand=10,
    seed=42,
):
    """
    Hugging Face LogoDet-3K.

    데이터 card / mirror에 따라 field spelling이 조금 다를 수 있어
    후보 field name을 허용한다.
    """

    print("[data] downloading/loading LogoDet-3K...")

    # 현재 HF mirror
    candidates = [
        "axonstan/LogoDet-3K",
        "PodYapolsky/LogoDet-3K",
    ]

    ds = None

    for name in candidates:
        try:
            print(f"[data] trying {name}")
            ds = load_dataset(name, split="train")
            print(f"[data] loaded {name}")
            break
        except Exception as e:
            print(f"[data] failed {name}: {e}")

    if ds is None:
        raise RuntimeError(
            "Could not download LogoDet-3K from Hugging Face."
        )

    print("[data] columns:", ds.column_names)

    groups = defaultdict(list)

    for row in tqdm(ds, desc="Filtering clothes"):

        industry = get_field(
            row,
            [
                "industry_name",
                "industy_name",      # dataset card typo 대응
                "industry",
            ]
        )

        # clothes subset
        if industry is not None:
            industry_str = str(industry).lower()

            if (
                "cloth" not in industry_str
                and "fashion" not in industry_str
                and "apparel" not in industry_str
            ):
                continue

        company = get_field(
            row,
            [
                "company_name",
                "company",
                "label",
                "class",
            ]
        )

        bbox = get_field(
            row,
            [
                "bbox",
                "bounding_box",
                "boxes",
            ]
        )

        image = get_field(
            row,
            [
                "image",
                "image_path",
            ]
        )

        if company is None or bbox is None or image is None:
            continue

        # 일부 HF feature가 list 안에 bbox를 둘 수도 있음
        if isinstance(bbox, dict):
            if {"xmin", "ymin", "xmax", "ymax"}.issubset(bbox):
                bbox = [
                    bbox["xmin"],
                    bbox["ymin"],
                    bbox["xmax"],
                    bbox["ymax"],
                ]

        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) > 0
            and isinstance(bbox[0], (list, tuple))
        ):
            bbox = bbox[0]

        try:
            bbox = list(map(float, bbox))
        except Exception:
            continue

        if len(bbox) != 4:
            continue

        groups[int(company)].append(
            {
                "image": image,
                "bbox": bbox,
                "company": int(company),
            }
        )

    print(f"[data] brands before filter={len(groups)}")

    groups = {
        k: v
        for k, v in groups.items()
        if len(v) >= min_per_brand
    }

    print(f"[data] brands >= {min_per_brand} imgs={len(groups)}")

    rng = random.Random(seed)

    brands = list(groups.keys())
    rng.shuffle(brands)

    brands = brands[:max_brands]

    samples = []

    label_map = {
        brand: i
        for i, brand in enumerate(brands)
    }

    for brand in brands:

        rows = groups[brand]

        rng.shuffle(rows)

        rows = rows[:max_per_brand]

        for r in rows:

            samples.append(
                {
                    **r,
                    "label": label_map[brand],
                }
            )

    rng.shuffle(samples)

    print(
        f"[data] selected brands={len(brands)}, "
        f"samples={len(samples)}"
    )

    # brand-wise stratified split
    by_label = defaultdict(list)

    for s in samples:
        by_label[s["label"]].append(s)

    train = []
    gallery = []
    query = []

    for label, rows in by_label.items():

        rng.shuffle(rows)

        n = len(rows)

        n_train = max(1, int(n * 0.70))
        n_gallery = max(1, int(n * 0.15))

        train.extend(rows[:n_train])

        gallery.extend(
            rows[n_train:n_train + n_gallery]
        )

        query.extend(
            rows[n_train + n_gallery:]
        )

    print(
        f"[split] train={len(train)}, "
        f"gallery={len(gallery)}, "
        f"query={len(query)}"
    )

    return train, gallery, query, len(brands)


# ============================================================
# PyTorch Dataset
# ============================================================

class BrandDataset(Dataset):

    def __init__(
        self,
        samples,
        processor,
        regime="none",
        evaluation_mode=None,
    ):
        self.samples = samples
        self.processor = processor
        self.regime = regime
        self.evaluation_mode = evaluation_mode

    def __len__(self):
        return len(self.samples)

    def load_image(self, obj):

        if isinstance(obj, Image.Image):
            return obj.convert("RGB")

        # HF Image object / path
        if isinstance(obj, str):
            return Image.open(obj).convert("RGB")

        if isinstance(obj, dict):

            if "path" in obj and obj["path"]:
                return Image.open(obj["path"]).convert("RGB")

            if "bytes" in obj and obj["bytes"]:
                import io
                return Image.open(
                    io.BytesIO(obj["bytes"])
                ).convert("RGB")

        return obj.convert("RGB")

    def __getitem__(self, idx):

        row = self.samples[idx]

        img = self.load_image(row["image"])

        bbox = row["bbox"]

        if self.evaluation_mode == "clean":
            aug = img

        elif self.evaluation_mode == "logo_mask":
            aug = logo_mask(
                img,
                bbox,
                scale=1.25
            )

        elif self.evaluation_mode == "logo_mask_large":
            aug = logo_mask(
                img,
                bbox,
                scale=1.75
            )

        else:
            aug = apply_augmentation(
                img,
                bbox,
                self.regime
            )

        inputs = self.processor(
            images=aug,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(
                row["label"],
                dtype=torch.long
            ),
        }


# ============================================================
# Model
# ============================================================

class CLIPBrandModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)

        dim = self.clip.config.projection_dim

        self.classifier = nn.Linear(
            dim,
            num_classes
        )

    def encode(self, pixel_values):

        features = self.clip.get_image_features(
            pixel_values=pixel_values
        )

        features = F.normalize(
            features,
            dim=-1
        )

        return features

    def forward(self, pixel_values):

        z = self.encode(pixel_values)

        logits = self.classifier(z)

        return logits, z


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    loader,
    device,
    epochs=3,
    lr=1e-5,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{epochs}"
        )

        for batch in pbar:

            x = batch["pixel_values"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()

            logits, _ = model(x)

            loss = F.cross_entropy(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * x.size(0)

            pred = logits.argmax(dim=-1)

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

            pbar.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}"
            )


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device
):

    model.eval()

    zs = []
    labels = []

    for batch in tqdm(
        loader,
        desc="Embedding"
    ):

        x = batch["pixel_values"].to(device)

        z = model.encode(x)

        zs.append(
            z.cpu()
        )

        labels.append(
            batch["label"]
        )

    return (
        torch.cat(zs),
        torch.cat(labels)
    )


# ============================================================
# Retrieval evaluation
# ============================================================

def retrieval_metrics(
    query_z,
    query_y,
    gallery_z,
    gallery_y,
    ks=(1, 5, 10),
):

    # cosine similarity
    sim = query_z @ gallery_z.T

    max_k = max(ks)

    indices = sim.topk(
        min(max_k, gallery_z.size(0)),
        dim=1
    ).indices

    results = {}

    for k in ks:

        top_labels = gallery_y[
            indices[:, :k]
        ]

        hit = (
            top_labels
            == query_y[:, None]
        ).any(dim=1)

        results[f"R@{k}"] = (
            hit.float()
            .mean()
            .item()
        )

    # MRR
    ranks = []

    for i in range(len(query_y)):

        labels = gallery_y[
            indices[i]
        ]

        match = (
            labels == query_y[i]
        ).nonzero(
            as_tuple=False
        )

        if len(match) == 0:
            ranks.append(0.0)

        else:
            rank = match[0].item() + 1
            ranks.append(1.0 / rank)

    results["MRR@10"] = float(
        np.mean(ranks)
    )

    return results


# ============================================================
# Experiment
# ============================================================

def run(args):

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device =", device)

    processor = CLIPProcessor.from_pretrained(
        args.model
    )

    train_samples, gallery_samples, query_samples, num_classes = (
        prepare_data(
            max_brands=args.max_brands,
            max_per_brand=args.max_per_brand,
            min_per_brand=args.min_per_brand,
            seed=args.seed
        )
    )

    regimes = args.regimes.split(",")

    all_results = {}

    for regime in regimes:

        print()
        print("=" * 80)
        print("REGIME:", regime)
        print("=" * 80)

        model = CLIPBrandModel(
            args.model,
            num_classes
        ).to(device)

        train_ds = BrandDataset(
            train_samples,
            processor,
            regime=regime
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )

        train_model(
            model,
            train_loader,
            device,
            epochs=args.epochs,
            lr=args.lr
        )

        # ----------------------------------------------------
        # Gallery = 항상 clean
        # ----------------------------------------------------

        gallery_ds = BrandDataset(
            gallery_samples,
            processor,
            evaluation_mode="clean"
        )

        gallery_loader = DataLoader(
            gallery_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )

        gallery_z, gallery_y = extract_embeddings(
            model,
            gallery_loader,
            device
        )

        results = {}

        # clean query / logo masked / severe logo masked
        for mode in [
            "clean",
            "logo_mask",
            "logo_mask_large",
        ]:

            query_ds = BrandDataset(
                query_samples,
                processor,
                evaluation_mode=mode
            )

            query_loader = DataLoader(
                query_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers
            )

            query_z, query_y = extract_embeddings(
                model,
                query_loader,
                device
            )

            metrics = retrieval_metrics(
                query_z,
                query_y,
                gallery_z,
                gallery_y
            )

            results[mode] = metrics

        all_results[regime] = results

        del model

        torch.cuda.empty_cache()

    # ========================================================
    # Print final table
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    header = (
        f"{'Regime':<18}"
        f"{'Clean R@1':>12}"
        f"{'Mask R@1':>12}"
        f"{'Heavy R@1':>12}"
        f"{'Mask R@5':>12}"
        f"{'Heavy R@5':>12}"
        f"{'Robust Drop':>14}"
    )

    print(header)
    print("-" * len(header))

    for regime, r in all_results.items():

        clean1 = r["clean"]["R@1"]
        mask1 = r["logo_mask"]["R@1"]
        heavy1 = r["logo_mask_large"]["R@1"]

        mask5 = r["logo_mask"]["R@5"]
        heavy5 = r["logo_mask_large"]["R@5"]

        drop = clean1 - mask1

        print(
            f"{regime:<18}"
            f"{clean1:>12.4f}"
            f"{mask1:>12.4f}"
            f"{heavy1:>12.4f}"
            f"{mask5:>12.4f}"
            f"{heavy5:>12.4f}"
            f"{drop:>14.4f}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32"
    )

    parser.add_argument(
        "--regimes",
        default=(
            "none,"
            "random_erasing,"
            "cutout,"
            "gridmask,"
            "logo_mask,"
            "mixed"
        )
    )

    parser.add_argument(
        "--max-brands",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max-per-brand",
        type=int,
        default=80
    )

    parser.add_argument(
        "--min-per-brand",
        type=int,
        default=10
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    run(args)
