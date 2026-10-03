# brand_occlusion_ablation_9way.py

import argparse
import io
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# Seed
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Image helpers
# ============================================================

def load_pil_image(obj):
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")

    if isinstance(obj, str):
        return Image.open(obj).convert("RGB")

    if isinstance(obj, dict):
        if obj.get("path"):
            return Image.open(obj["path"]).convert("RGB")

        if obj.get("bytes"):
            return Image.open(
                io.BytesIO(obj["bytes"])
            ).convert("RGB")

    if hasattr(obj, "convert"):
        return obj.convert("RGB")

    raise ValueError(
        f"Unsupported image type: {type(obj)}"
    )


def mean_color(img):
    arr = np.asarray(
        img.resize((32, 32))
    ).astype(np.float32)

    c = arr.reshape(
        -1, 3
    ).mean(axis=0)

    return tuple(
        c.astype(np.uint8)
    )


# ============================================================
# Bounding box
# LogoDet-3K:
# [xmin, ymin, xmax, ymax]
# ============================================================

def normalize_bbox(bbox, w, h):

    if isinstance(bbox, dict):
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

    x1, y1, x2, y2 = map(
        float,
        bbox
    )

    x1 = int(max(
        0,
        min(x1, w - 1)
    ))

    y1 = int(max(
        0,
        min(y1, h - 1)
    ))

    x2 = int(max(
        x1 + 1,
        min(x2, w)
    ))

    y2 = int(max(
        y1 + 1,
        min(y2, h)
    ))

    return x1, y1, x2, y2


def expand_bbox(
    box,
    w,
    h,
    scale=1.25
):
    x1, y1, x2, y2 = box

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    bw = (
        x2 - x1
    ) * scale

    bh = (
        y2 - y1
    ) * scale

    nx1 = max(
        0,
        int(cx - bw / 2)
    )

    ny1 = max(
        0,
        int(cy - bh / 2)
    )

    nx2 = min(
        w,
        int(cx + bw / 2)
    )

    ny2 = min(
        h,
        int(cy + bh / 2)
    )

    return (
        nx1,
        ny1,
        max(nx1 + 1, nx2),
        max(ny1 + 1, ny2)
    )


# ============================================================
# Rectangle mask
# ============================================================

def mask_rectangle(
    img,
    box
):
    img = img.copy()

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        box,
        fill=mean_color(img)
    )

    return img


# ============================================================
# 1. Random Erasing
# ============================================================

def random_erasing(
    img,
    min_area=0.05,
    max_area=0.25
):
    img = img.copy()

    w, h = img.size
    area = w * h

    for _ in range(10):

        target = (
            random.uniform(
                min_area,
                max_area
            )
            * area
        )

        aspect = random.uniform(
            0.5,
            2.0
        )

        ew = int(
            np.sqrt(
                target * aspect
            )
        )

        eh = int(
            np.sqrt(
                target / aspect
            )
        )

        if (
            0 < ew < w
            and
            0 < eh < h
        ):

            x = random.randint(
                0,
                w - ew
            )

            y = random.randint(
                0,
                h - eh
            )

            return mask_rectangle(
                img,
                (
                    x,
                    y,
                    x + ew,
                    y + eh
                )
            )

    return img


# ============================================================
# 2. Cutout
# ============================================================

def cutout(
    img,
    ratio=0.30
):
    img = img.copy()

    w, h = img.size

    cw = max(
        1,
        int(w * ratio)
    )

    ch = max(
        1,
        int(h * ratio)
    )

    x = random.randint(
        0,
        max(0, w - cw)
    )

    y = random.randint(
        0,
        max(0, h - ch)
    )

    return mask_rectangle(
        img,
        (
            x,
            y,
            x + cw,
            y + ch
        )
    )


# ============================================================
# 3. GridMask
# ============================================================

def gridmask(
    img,
    grid=4,
    ratio=0.45
):
    img = img.copy()

    draw = ImageDraw.Draw(img)

    w, h = img.size

    gw = max(
        1,
        w // grid
    )

    gh = max(
        1,
        h // grid
    )

    mw = max(
        1,
        int(
            gw * ratio
        )
    )

    mh = max(
        1,
        int(
            gh * ratio
        )
    )

    ox = random.randint(
        0,
        gw - 1
    )

    oy = random.randint(
        0,
        gh - 1
    )

    fill = mean_color(img)

    for x in range(
        -gw,
        w + gw,
        gw
    ):
        for y in range(
            -gh,
            h + gh,
            gh
        ):

            draw.rectangle(
                (
                    x + ox,
                    y + oy,
                    x + ox + mw,
                    y + oy + mh
                ),
                fill=fill
            )

    return img


# ============================================================
# 4. Hide-and-Seek
# ============================================================

def hide_and_seek(
    img,
    grid=4,
    hide_prob=0.35
):
    img = img.copy()

    draw = ImageDraw.Draw(img)

    w, h = img.size

    cell_w = max(
        1,
        w // grid
    )

    cell_h = max(
        1,
        h // grid
    )

    fill = mean_color(img)

    for gy in range(grid):
        for gx in range(grid):

            if (
                random.random()
                < hide_prob
            ):

                x1 = gx * cell_w
                y1 = gy * cell_h

                x2 = min(
                    w,
                    (gx + 1)
                    * cell_w
                )

                y2 = min(
                    h,
                    (gy + 1)
                    * cell_h
                )

                draw.rectangle(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    fill=fill
                )

    return img


# ============================================================
# 5. Logo-aware masking
# ============================================================

def logo_mask(
    img,
    bbox,
    scale=1.25
):
    w, h = img.size

    box = normalize_bbox(
        bbox,
        w,
        h
    )

    box = expand_bbox(
        box,
        w,
        h,
        scale=scale
    )

    return mask_rectangle(
        img,
        box
    )


# ============================================================
# Dataset-level augmentation
# ============================================================

def apply_augmentation(
    img,
    bbox,
    regime
):

    if regime in [
        "baseline",
        "cutmix"
    ]:
        return img

    if regime == "random_erasing":

        return random_erasing(
            img
        )

    if regime == "cutout":

        return cutout(
            img
        )

    if regime == "gridmask":

        return gridmask(
            img
        )

    if regime == "hide_and_seek":

        return hide_and_seek(
            img
        )

    if regime == "logo_mask":

        # clean / masked 혼합
        if random.random() < 0.5:

            return logo_mask(
                img,
                bbox,
                scale=1.25
            )

        return img

    if regime in [
        "mixed",
        "part_based"
    ]:

        r = random.random()

        # 40% clean
        if r < 0.40:
            return img

        # 20% logo targeted
        if r < 0.60:
            return logo_mask(
                img,
                bbox,
                scale=1.25
            )

        # 20% random erasing
        if r < 0.80:
            return random_erasing(
                img
            )

        # 20% Hide-and-Seek
        return hide_and_seek(
            img
        )

    raise ValueError(
        regime
    )


# ============================================================
# Data
# ============================================================

def prepare_data(
    max_brands,
    max_per_brand,
    min_per_brand,
    seed
):

    print(
        "[data] loading LogoDet-3K"
    )

    ds = load_dataset(
        "axonstan/LogoDet-3K",
        split="train"
    )

    print(
        "[data] rows =",
        len(ds)
    )

    print(
        "[data] columns =",
        ds.column_names
    )

    groups = defaultdict(
        list
    )

    for row in tqdm(
        ds,
        desc="Filtering Clothes"
    ):

        industry = str(
            row[
                "industry_name"
            ]
        ).lower()

        if "cloth" not in industry:
            continue

        brand = int(
            row[
                "company_name"
            ]
        )

        groups[
            brand
        ].append(
            {
                "image": (
                    row[
                        "image_path"
                    ]
                ),
                "bbox": row["bbox"],
                "brand": brand,
            }
        )

    print(
        "[data] clothing brands =",
        len(groups)
    )

    groups = {
        k: v
        for k, v
        in groups.items()
        if len(v)
        >= min_per_brand
    }

    rng = random.Random(
        seed
    )

    brands = list(
        groups.keys()
    )

    rng.shuffle(
        brands
    )

    brands = brands[
        :max_brands
    ]

    label_map = {
        b: i
        for i, b
        in enumerate(brands)
    }

    samples = []

    for brand in brands:

        rows = list(
            groups[
                brand
            ]
        )

        rng.shuffle(
            rows
        )

        rows = rows[
            :max_per_brand
        ]

        for row in rows:

            samples.append(
                {
                    **row,
                    "label": (
                        label_map[
                            brand
                        ]
                    )
                }
            )

    by_label = defaultdict(
        list
    )

    for sample in samples:

        by_label[
            sample[
                "label"
            ]
        ].append(
            sample
        )

    train = []
    gallery = []
    query = []

    for _, rows in (
        by_label.items()
    ):

        rng.shuffle(
            rows
        )

        n = len(rows)

        if n < 3:
            continue

        n_train = max(
            1,
            int(
                n * 0.70
            )
        )

        n_gallery = max(
            1,
            int(
                n * 0.15
            )
        )

        if (
            n_train
            + n_gallery
            >= n
        ):

            n_train = (
                n - 2
            )

            n_gallery = 1

        train.extend(
            rows[
                :n_train
            ]
        )

        gallery.extend(
            rows[
                n_train:
                n_train
                + n_gallery
            ]
        )

        query.extend(
            rows[
                n_train
                + n_gallery:
            ]
        )

    print(
        f"[split] "
        f"train={len(train)} "
        f"gallery={len(gallery)} "
        f"query={len(query)}"
    )

    return (
        train,
        gallery,
        query,
        len(brands)
    )


# ============================================================
# Dataset
# ============================================================

class BrandDataset(
    Dataset
):

    def __init__(
        self,
        samples,
        processor,
        regime="baseline",
        eval_mode=None
    ):

        self.samples = samples
        self.processor = processor

        self.regime = regime
        self.eval_mode = (
            eval_mode
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        row = (
            self.samples[
                idx
            ]
        )

        img = load_pil_image(
            row[
                "image"
            ]
        )

        bbox = row[
            "bbox"
        ]

        if (
            self.eval_mode
            == "clean"
        ):

            out = img

        elif (
            self.eval_mode
            == "logo25"
        ):

            out = logo_mask(
                img,
                bbox,
                scale=0.70
            )

        elif (
            self.eval_mode
            == "logo50"
        ):

            out = logo_mask(
                img,
                bbox,
                scale=1.00
            )

        elif (
            self.eval_mode
            == "logo100"
        ):

            out = logo_mask(
                img,
                bbox,
                scale=1.35
            )

        elif (
            self.eval_mode
            == "heavy"
        ):

            out = logo_mask(
                img,
                bbox,
                scale=1.75
            )

        else:

            out = apply_augmentation(
                img,
                bbox,
                self.regime
            )

        processed = (
            self.processor(
                images=out,
                return_tensors="pt"
            )
        )

        return {
            "pixel_values":
                processed[
                    "pixel_values"
                ][0],

            "label":
                torch.tensor(
                    row[
                        "label"
                    ],
                    dtype=torch.long
                )
        }


# ============================================================
# CLIP model
# ============================================================

class CLIPBrandModel(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_classes
    ):

        super().__init__()

        self.clip = (
            CLIPModel
            .from_pretrained(
                model_name
            )
        )

        dim = (
            self.clip
            .config
            .projection_dim
        )

        self.classifier = (
            nn.Linear(
                dim,
                num_classes
            )
        )

    def encode(
        self,
        x
    ):

        outputs = (
            self.clip
            .vision_model(
                pixel_values=x,
                return_dict=True
            )
        )

        pooled = (
            outputs
            .pooler_output
        )

        features = (
            self.clip
            .visual_projection(
                pooled
            )
        )

        return F.normalize(
            features,
            dim=-1
        )

    def forward(
        self,
        x
    ):

        z = self.encode(
            x
        )

        logits = (
            self.classifier(
                z
            )
        )

        return (
            logits,
            z
        )


# ============================================================
# CutMix
# ============================================================

def rand_bbox(
    size,
    lam
):

    H = size[2]
    W = size[3]

    cut_ratio = np.sqrt(
        1.0 - lam
    )

    cut_w = int(
        W * cut_ratio
    )

    cut_h = int(
        H * cut_ratio
    )

    cx = np.random.randint(
        W
    )

    cy = np.random.randint(
        H
    )

    x1 = np.clip(
        cx - cut_w // 2,
        0,
        W
    )

    y1 = np.clip(
        cy - cut_h // 2,
        0,
        H
    )

    x2 = np.clip(
        cx + cut_w // 2,
        0,
        W
    )

    y2 = np.clip(
        cy + cut_h // 2,
        0,
        H
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def apply_cutmix(
    x,
    y,
    alpha=1.0
):

    lam = np.random.beta(
        alpha,
        alpha
    )

    index = torch.randperm(
        x.size(0),
        device=x.device
    )

    y_a = y
    y_b = y[
        index
    ]

    x1, y1, x2, y2 = (
        rand_bbox(
            x.size(),
            lam
        )
    )

    mixed = (
        x.clone()
    )

    mixed[
        :,
        :,
        y1:y2,
        x1:x2
    ] = x[
        index,
        :,
        y1:y2,
        x1:x2
    ]

    box_area = (
        (x2 - x1)
        * (y2 - y1)
    )

    total_area = (
        x.size(2)
        * x.size(3)
    )

    lam = (
        1.0
        - box_area
        / total_area
    )

    return (
        mixed,
        y_a,
        y_b,
        float(lam)
    )


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    loader,
    device,
    regime,
    epochs,
    lr
):

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
    )

    for epoch in range(
        epochs
    ):

        model.train()

        running_loss = 0.0
        total = 0

        pbar = tqdm(
            loader,
            desc=(
                f"{regime} "
                f"epoch "
                f"{epoch+1}/{epochs}"
            )
        )

        for batch in pbar:

            x = (
                batch[
                    "pixel_values"
                ].to(
                    device,
                    non_blocking=True
                )
            )

            y = (
                batch[
                    "label"
                ].to(
                    device,
                    non_blocking=True
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ================================================
            # CutMix regime
            # ================================================

            use_cutmix = False

            if regime == "cutmix":

                use_cutmix = True

            # mixed / part_based에도 일부 CutMix
            elif regime in [
                "mixed",
                "part_based"
            ]:

                if random.random() < 0.25:
                    use_cutmix = True

            if (
                use_cutmix
                and
                x.size(0) > 1
            ):

                (
                    x_mix,
                    y_a,
                    y_b,
                    lam
                ) = apply_cutmix(
                    x,
                    y
                )

                logits, _ = (
                    model(
                        x_mix
                    )
                )

                loss = (
                    lam
                    * F.cross_entropy(
                        logits,
                        y_a
                    )
                    +
                    (1.0 - lam)
                    * F.cross_entropy(
                        logits,
                        y_b
                    )
                )

            else:

                logits, _ = (
                    model(
                        x
                    )
                )

                loss = (
                    F.cross_entropy(
                        logits,
                        y
                    )
                )

            loss.backward()

            optimizer.step()

            n = y.size(0)

            running_loss += (
                loss.item()
                * n
            )

            total += n

            pbar.set_postfix(
                loss=(
                    f"{running_loss / total:.4f}"
                )
            )


# ============================================================
# Normal embedding extraction
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device
):

    model.eval()

    embeddings = []
    labels = []

    for batch in tqdm(
        loader,
        desc="Embedding"
    ):

        x = (
            batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True
            )
        )

        z = model.encode(
            x
        )

        embeddings.append(
            z.cpu()
        )

        labels.append(
            batch[
                "label"
            ]
        )

    return (
        torch.cat(
            embeddings
        ),
        torch.cat(
            labels
        )
    )


# ============================================================
# Part crops
# ============================================================

def make_part_crops(
    x
):
    """
    x:
      [B,C,H,W]

    parts:
      top-left
      top-right
      bottom-left
      bottom-right
      center

    모든 crop을 원래 CLIP input size로 resize.
    """

    B, C, H, W = (
        x.shape
    )

    h2 = H // 2
    w2 = W // 2

    crops = []

    # TL
    crops.append(
        x[
            :,
            :,
            0:h2,
            0:w2
        ]
    )

    # TR
    crops.append(
        x[
            :,
            :,
            0:h2,
            w2:W
        ]
    )

    # BL
    crops.append(
        x[
            :,
            :,
            h2:H,
            0:w2
        ]
    )

    # BR
    crops.append(
        x[
            :,
            :,
            h2:H,
            w2:W
        ]
    )

    # center crop
    y1 = H // 4
    y2 = 3 * H // 4

    x1 = W // 4
    x2 = 3 * W // 4

    crops.append(
        x[
            :,
            :,
            y1:y2,
            x1:x2
        ]
    )

    resized = []

    for crop in crops:

        crop = F.interpolate(
            crop,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        resized.append(
            crop
        )

    return torch.stack(
        resized,
        dim=1
    )


# ============================================================
# Part embeddings
# ============================================================

@torch.no_grad()
def extract_part_embeddings(
    model,
    loader,
    device
):

    model.eval()

    global_all = []
    part_all = []
    label_all = []

    for batch in tqdm(
        loader,
        desc="Part embedding"
    ):

        x = (
            batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True
            )
        )

        global_z = (
            model.encode(
                x
            )
        )

        crops = (
            make_part_crops(
                x
            )
        )

        B, P, C, H, W = (
            crops.shape
        )

        crops = crops.reshape(
            B * P,
            C,
            H,
            W
        )

        part_z = (
            model.encode(
                crops
            )
        )

        part_z = part_z.reshape(
            B,
            P,
            -1
        )

        global_all.append(
            global_z.cpu()
        )

        part_all.append(
            part_z.cpu()
        )

        label_all.append(
            batch[
                "label"
            ]
        )

    return (
        torch.cat(
            global_all
        ),
        torch.cat(
            part_all
        ),
        torch.cat(
            label_all
        )
    )


# ============================================================
# Normal retrieval
# ============================================================

def normal_similarity(
    q,
    g
):
    return (
        q
        @ g.T
    )


# ============================================================
# Part-based retrieval
# ============================================================

def part_similarity(
    q_global,
    q_parts,
    g_global,
    g_parts,
    global_weight=0.60
):

    global_sim = (
        q_global
        @ g_global.T
    )

    # q_parts:
    # [Q,P,D]
    #
    # g_parts:
    # [G,P,D]
    #
    # result:
    # [Q,G,P,P]

    part_matrix = torch.einsum(
        "qpd,gkd->qgpk",
        q_parts,
        g_parts
    )

    # strongest visible-part match
    best_part = (
        part_matrix
        .amax(
            dim=(-1, -2)
        )
    )

    return (
        global_weight
        * global_sim
        +
        (1.0 - global_weight)
        * best_part
    )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    similarity,
    query_labels,
    gallery_labels,
    ks=(1, 5, 10)
):

    max_k = min(
        max(ks),
        similarity.shape[1]
    )

    indices = (
        similarity
        .topk(
            max_k,
            dim=1
        )
        .indices
    )

    result = {}

    for k in ks:

        k = min(
            k,
            max_k
        )

        preds = (
            gallery_labels[
                indices[
                    :, :k
                ]
            ]
        )

        hits = (
            preds
            == query_labels[
                :, None
            ]
        ).any(
            dim=1
        )

        result[
            f"R@{k}"
        ] = (
            hits.float()
            .mean()
            .item()
        )

    reciprocal = []

    for i in range(
        len(query_labels)
    ):

        ranked_labels = (
            gallery_labels[
                indices[i]
            ]
        )

        match = (
            ranked_labels
            == query_labels[i]
        ).nonzero(
            as_tuple=False
        )

        if len(match) == 0:

            reciprocal.append(
                0.0
            )

        else:

            reciprocal.append(
                1.0
                /
                (
                    match[0].item()
                    + 1
                )
            )

    result[
        "MRR@10"
    ] = float(
        np.mean(
            reciprocal
        )
    )

    return result


# ============================================================
# Loader
# ============================================================

def make_loader(
    dataset,
    batch_size,
    workers,
    shuffle=False
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        )
    )


# ============================================================
# Main
# ============================================================

def run(args):

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device
    )

    processor = (
        CLIPProcessor
        .from_pretrained(
            args.model
        )
    )

    (
        train_samples,
        gallery_samples,
        query_samples,
        num_classes
    ) = prepare_data(
        args.max_brands,
        args.max_per_brand,
        args.min_per_brand,
        args.seed
    )

    regimes = [
        "baseline",
        "random_erasing",
        "cutout",
        "gridmask",
        "hide_and_seek",
        "cutmix",
        "logo_mask",
        "mixed",
        "part_based",
    ]

    if args.regimes:

        regimes = [
            r.strip()
            for r
            in args.regimes.split(",")
        ]

    eval_modes = [
        "clean",
        "logo25",
        "logo50",
        "logo100",
        "heavy",
    ]

    all_results = {}

    for regime in regimes:

        print()
        print(
            "=" * 100
        )

        print(
            "REGIME:",
            regime
        )

        print(
            "=" * 100
        )

        seed_everything(
            args.seed
        )

        model = (
            CLIPBrandModel(
                args.model,
                num_classes
            )
            .to(device)
        )

        train_ds = (
            BrandDataset(
                train_samples,
                processor,
                regime=regime
            )
        )

        train_loader = (
            make_loader(
                train_ds,
                args.batch_size,
                args.workers,
                shuffle=True
            )
        )

        train_model(
            model,
            train_loader,
            device,
            regime,
            args.epochs,
            args.lr
        )

        # ====================================================
        # Gallery
        # ====================================================

        gallery_ds = (
            BrandDataset(
                gallery_samples,
                processor,
                eval_mode="clean"
            )
        )

        gallery_loader = (
            make_loader(
                gallery_ds,
                args.batch_size,
                args.workers
            )
        )

        # ====================================================
        # Part-based
        # ====================================================

        if regime == "part_based":

            (
                g_global,
                g_parts,
                g_labels
            ) = (
                extract_part_embeddings(
                    model,
                    gallery_loader,
                    device
                )
            )

        else:

            (
                g_emb,
                g_labels
            ) = (
                extract_embeddings(
                    model,
                    gallery_loader,
                    device
                )
            )

        regime_results = {}

        # ====================================================
        # Evaluation modes
        # ====================================================

        for mode in eval_modes:

            query_ds = (
                BrandDataset(
                    query_samples,
                    processor,
                    eval_mode=mode
                )
            )

            query_loader = (
                make_loader(
                    query_ds,
                    args.batch_size,
                    args.workers
                )
            )

            if regime == "part_based":

                (
                    q_global,
                    q_parts,
                    q_labels
                ) = (
                    extract_part_embeddings(
                        model,
                        query_loader,
                        device
                    )
                )

                sim = part_similarity(
                    q_global,
                    q_parts,
                    g_global,
                    g_parts,
                    global_weight=(
                        args.global_weight
                    )
                )

            else:

                (
                    q_emb,
                    q_labels
                ) = (
                    extract_embeddings(
                        model,
                        query_loader,
                        device
                    )
                )

                sim = normal_similarity(
                    q_emb,
                    g_emb
                )

            metrics = compute_metrics(
                sim,
                q_labels,
                g_labels
            )

            regime_results[
                mode
            ] = metrics

            print(
                f"[{regime}] "
                f"{mode}: "
                f"{metrics}"
            )

        all_results[
            regime
        ] = regime_results

        del model

        torch.cuda.empty_cache()

    # ========================================================
    # Final table
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "FINAL 9-WAY OCCLUSION ROBUSTNESS RESULTS"
    )

    print(
        "=" * 140
    )

    header = (
        f"{'Regime':<20}"
        f"{'Clean R1':>10}"
        f"{'25% R1':>10}"
        f"{'50% R1':>10}"
        f"{'100% R1':>10}"
        f"{'Heavy R1':>10}"
        f"{'Clean R5':>10}"
        f"{'100% R5':>10}"
        f"{'Clean MRR':>12}"
        f"{'100% MRR':>12}"
        f"{'Drop':>10}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for regime in regimes:

        r = all_results[
            regime
        ]

        clean = r[
            "clean"
        ]

        o25 = r[
            "logo25"
        ]

        o50 = r[
            "logo50"
        ]

        o100 = r[
            "logo100"
        ]

        heavy = r[
            "heavy"
        ]

        drop = (
            clean["R@1"]
            - o100["R@1"]
        )

        print(
            f"{regime:<20}"
            f"{clean['R@1']:>10.4f}"
            f"{o25['R@1']:>10.4f}"
            f"{o50['R@1']:>10.4f}"
            f"{o100['R@1']:>10.4f}"
            f"{heavy['R@1']:>10.4f}"
            f"{clean['R@5']:>10.4f}"
            f"{o100['R@5']:>10.4f}"
            f"{clean['MRR@10']:>12.4f}"
            f"{o100['MRR@10']:>12.4f}"
            f"{drop:>10.4f}"
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=(
            "openai/"
            "clip-vit-base-patch32"
        )
    )

    parser.add_argument(
        "--max-brands",
        type=int,
        default=50
    )

    parser.add_argument(
        "--max-per-brand",
        type=int,
        default=50
    )

    parser.add_argument(
        "--min-per-brand",
        type=int,
        default=10
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2
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

    parser.add_argument(
        "--global-weight",
        type=float,
        default=0.60
    )

    parser.add_argument(
        "--regimes",
        type=str,
        default=""
    )

    args = parser.parse_args()

    run(args)
