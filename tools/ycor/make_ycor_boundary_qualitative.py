#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YCOR traversable <-> non-traversable boundary qualitative visualization.

This script DOES NOT run model inference.

It reads:
    - frozen YCOR RGB images
    - frozen YCOR palette-index GT labels
    - formal seed123 prediction PNGs from:
        A
        D-BGA

It generates:

    RGB
    | GT Boundary
    | A Boundary Errors
    | D-BGA Boundary Errors

Boundary definition exactly follows the formal RUGD3 / YCOR BF1 evaluator:

    traversable (class 1)
        <->
    non-traversable (class 2)

Boundary tolerance:
    3 pixels

Error visualization:

    GREEN  = predicted boundary matched to GT within tolerance
    RED    = predicted boundary not matched to GT
             (false / spurious prediction boundary)
    YELLOW = GT boundary not matched by prediction
             (missed GT boundary)
    WHITE  = GT boundary in GT-only column

Important:
    Display dilation is VISUALIZATION ONLY.
    Matching itself is always computed on the original 1-pixel semantic
    boundary representation using the frozen 3-pixel tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(
    "/root/autodl-tmp/projects/BiSeNet"
)

EXT = (
    REPO_ROOT
    / "experiments"
    / "external_ycor_zero_shot"
)

DATA_ROOT = Path(
    "/root/autodl-tmp/datasets/YCOR-external/data"
)

DEFAULT_SAMPLE_FILE = (
    EXT
    / "figures"
    / "qualitative_seed123"
    / "main_figure_fixed6_samples.txt"
)

FALLBACK_SAMPLE_FILE = (
    EXT
    / "protocol"
    / "qualitative_samples.txt"
)

A_PRED_DIR = (
    EXT
    / "results"
    / "ycor_all"
    / "a_seed123"
    / "qualitative_predictions"
)

DBGA_PRED_DIR = (
    EXT
    / "results"
    / "ycor_all"
    / "dbga_seed123"
    / "qualitative_predictions"
)

DEFAULT_OUT_DIR = (
    EXT
    / "figures"
    / "boundary_qualitative_seed123"
)


# ============================================================
# Frozen ontology
# ============================================================

IGNORE_INDEX = 255

SOURCE_TO_TARGET = {
    0: 255,  # background -> ignore
    1: 1,    # smooth_trail -> traversable
    2: 1,    # traversable_grass -> traversable
    3: 1,    # rough_trail -> traversable
    4: 2,    # puddle -> non-traversable
    5: 2,    # obstacle -> non-traversable
    6: 2,    # non_traversable_low_vegetation -> non-traversable
    7: 2,    # high_vegetation -> non-traversable
    8: 0,    # sky -> sky
}

TARGET_LUT = np.full(
    256,
    IGNORE_INDEX,
    dtype=np.uint8,
)

for source_id, target_id in SOURCE_TO_TARGET.items():
    TARGET_LUT[source_id] = target_id


# ============================================================
# Frozen boundary protocol
# ============================================================

BOUNDARY_TOLERANCE = 3


# ============================================================
# Visualization settings
# ============================================================

# Original YCOR:
# W=1024, H=544, aspect ~= 1.882
DISPLAY_W = 768
DISPLAY_H = 408

LEFT_LABEL_W = 150
HEADER_H = 70
ROW_GAP = 7
LEGEND_H = 90
BOTTOM_MARGIN = 15

NUM_COLUMNS = 4

COLUMN_TITLES = [
    "RGB",
    "GT Boundary",
    "A Boundary Errors",
    "D-BGA Boundary Errors",
]

# Image background darkening for boundary visibility.
DIM_FACTOR = 0.48

# Display line thickness.
#
# IMPORTANT:
# This does NOT affect BF1 matching.
# It only thickens pixels for the exported figure.
DISPLAY_RADIUS = 2

COLOR_MATCHED = np.array(
    [30, 220, 60],
    dtype=np.uint8,
)

COLOR_MISSED = np.array(
    [255, 220, 0],
    dtype=np.uint8,
)

COLOR_FALSE = np.array(
    [240, 40, 40],
    dtype=np.uint8,
)

COLOR_GT = np.array(
    [255, 255, 255],
    dtype=np.uint8,
)

BACKGROUND = (
    255,
    255,
    255,
)

HEADER_BACKGROUND = (
    246,
    246,
    246,
)

LABEL_BACKGROUND = (
    248,
    248,
    248,
)

GRID_COLOR = (
    220,
    220,
    220,
)

TEXT_COLOR = (
    20,
    20,
    20,
)


# ============================================================
# Font helper
# ============================================================

def find_font(
    size: int,
    bold: bool = False,
) -> ImageFont.ImageFont:

    if bold:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]

    else:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for candidate in candidates:

        path = Path(candidate)

        if path.is_file():

            return ImageFont.truetype(
                str(path),
                size=size,
            )

    return ImageFont.load_default()


FONT_HEADER = find_font(
    29,
    bold=True,
)

FONT_SAMPLE = find_font(
    24,
    bold=True,
)

FONT_LEGEND = find_font(
    22,
    bold=False,
)

FONT_SMALL = find_font(
    18,
    bold=False,
)


# ============================================================
# Formal boundary implementation
# ============================================================

def semantic_boundary(
    mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the formal traversability boundary definition.

    Boundary is produced only for:
        class 1 <-> class 2

    mask:
        H x W integer array

    valid:
        H x W bool array
    """

    if mask.ndim != 2:

        raise ValueError(
            f"mask must be HxW, got {mask.shape}"
        )

    if valid.shape != mask.shape:

        raise ValueError(
            "valid/mask shape mismatch: "
            f"{valid.shape} vs {mask.shape}"
        )

    edge = np.zeros(
        mask.shape,
        dtype=np.bool_,
    )

    # --------------------------------------------------------
    # Horizontal neighbours
    # --------------------------------------------------------

    left = mask[
        :,
        :-1
    ]

    right = mask[
        :,
        1:
    ]

    pair_valid = (
        valid[:, :-1]
        &
        valid[:, 1:]
    )

    difference = (
        (
            (left == 1)
            &
            (right == 2)
        )
        |
        (
            (left == 2)
            &
            (right == 1)
        )
    ) & pair_valid

    edge[:, :-1] |= difference
    edge[:, 1:] |= difference

    # --------------------------------------------------------
    # Vertical neighbours
    # --------------------------------------------------------

    top = mask[
        :-1,
        :
    ]

    bottom = mask[
        1:,
        :
    ]

    pair_valid = (
        valid[:-1, :]
        &
        valid[1:, :]
    )

    difference = (
        (
            (top == 1)
            &
            (bottom == 2)
        )
        |
        (
            (top == 2)
            &
            (bottom == 1)
        )
    ) & pair_valid

    edge[:-1, :] |= difference
    edge[1:, :] |= difference

    edge &= valid

    return edge


def dilation(
    edge: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """
    Same elliptical dilation used by the formal BF1 evaluator.
    """

    if tolerance <= 0:
        return edge.copy()

    size = (
        2 * tolerance
        + 1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            size,
            size,
        ),
    )

    out = cv2.dilate(
        edge.astype(
            np.uint8
        ),
        kernel,
        iterations=1,
    )

    return out.astype(
        bool
    )


# ============================================================
# Boundary error decomposition
# ============================================================

def compute_boundary_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int,
) -> dict[str, np.ndarray]:
    """
    Decompose traversability boundary into:

        matched_pred:
            predicted boundary pixels that fall within
            tolerance of GT boundary.

        false_pred:
            predicted boundary pixels not matched to GT.

        matched_gt:
            GT boundary pixels that fall within tolerance
            of predicted boundary.

        missed_gt:
            GT boundary pixels not matched by prediction.

    This follows the same asymmetric precision/recall matching
    used by the formal BF1 evaluator.
    """

    if prediction.shape != target.shape:

        raise RuntimeError(
            "prediction/target shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    valid = (
        target
        != IGNORE_INDEX
    )

    pred_edge = semantic_boundary(
        prediction,
        valid,
    )

    gt_edge = semantic_boundary(
        target,
        valid,
    )

    pred_dilated = dilation(
        pred_edge,
        tolerance,
    )

    gt_dilated = dilation(
        gt_edge,
        tolerance,
    )

    matched_pred = (
        pred_edge
        &
        gt_dilated
    )

    false_pred = (
        pred_edge
        &
        ~gt_dilated
    )

    matched_gt = (
        gt_edge
        &
        pred_dilated
    )

    missed_gt = (
        gt_edge
        &
        ~pred_dilated
    )

    return {
        "pred_edge":
            pred_edge,

        "gt_edge":
            gt_edge,

        "matched_pred":
            matched_pred,

        "false_pred":
            false_pred,

        "matched_gt":
            matched_gt,

        "missed_gt":
            missed_gt,
    }


def compute_boundary_metrics(
    errors: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Convenience diagnostic only.

    It reproduces the BF1 quantities for one sample.
    """

    pred_edge = errors[
        "pred_edge"
    ]

    gt_edge = errors[
        "gt_edge"
    ]

    matched_pred = errors[
        "matched_pred"
    ]

    matched_gt = errors[
        "matched_gt"
    ]

    pred_total = int(
        pred_edge.sum()
    )

    gt_total = int(
        gt_edge.sum()
    )

    pred_matched = int(
        matched_pred.sum()
    )

    gt_matched = int(
        matched_gt.sum()
    )

    precision = (
        pred_matched
        /
        max(
            pred_total,
            1,
        )
    )

    recall = (
        gt_matched
        /
        max(
            gt_total,
            1,
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        max(
            precision
            + recall,
            1e-12,
        )
    )

    return {
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "pred_total":
            pred_total,

        "gt_total":
            gt_total,

        "pred_matched":
            pred_matched,

        "gt_matched":
            gt_matched,
    }


# ============================================================
# Visualization helper
# ============================================================

def display_dilate(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Visual thickness only.

    This must NEVER be used for matching.
    """

    if radius <= 0:
        return mask.copy()

    size = (
        2 * radius
        + 1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            size,
            size,
        ),
    )

    return cv2.dilate(
        mask.astype(
            np.uint8
        ),
        kernel,
        iterations=1,
    ).astype(
        bool
    )


def dim_rgb(
    rgb: np.ndarray,
) -> np.ndarray:

    out = (
        rgb.astype(
            np.float32
        )
        * DIM_FACTOR
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(
        np.uint8
    )


def paint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
) -> None:

    image[
        mask
    ] = color


def visualize_gt_boundary(
    rgb: np.ndarray,
    gt_edge: np.ndarray,
) -> np.ndarray:

    out = dim_rgb(
        rgb
    )

    gt_display = display_dilate(
        gt_edge,
        DISPLAY_RADIUS,
    )

    paint_mask(
        out,
        gt_display,
        COLOR_GT,
    )

    return out


def visualize_model_errors(
    rgb: np.ndarray,
    errors: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Overlay order:

        green  : matched predicted boundary
        red    : false predicted boundary
        yellow : missed GT boundary

    Missed GT is painted last so it remains visible if
    visual-display dilation overlaps another category.

    Again: overlap here is visualization only.
    """

    out = dim_rgb(
        rgb
    )

    matched_display = display_dilate(
        errors[
            "matched_pred"
        ],
        DISPLAY_RADIUS,
    )

    false_display = display_dilate(
        errors[
            "false_pred"
        ],
        DISPLAY_RADIUS,
    )

    missed_display = display_dilate(
        errors[
            "missed_gt"
        ],
        DISPLAY_RADIUS,
    )

    paint_mask(
        out,
        matched_display,
        COLOR_MATCHED,
    )

    paint_mask(
        out,
        false_display,
        COLOR_FALSE,
    )

    paint_mask(
        out,
        missed_display,
        COLOR_MISSED,
    )

    return out


# ============================================================
# Dataset loading
# ============================================================

def prediction_filename(
    sample: str,
) -> str:

    return str(
        Path(
            sample.replace(
                "/",
                "__",
            )
        ).with_suffix(
            ".png"
        )
    )


def load_rgb(
    sample: str,
) -> np.ndarray:

    path = (
        DATA_ROOT
        / sample
    )

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"Failed to read RGB: {path}"
        )

    # Match project convention:
    # OpenCV BGR -> RGB
    image = (
        image[
            :,
            :,
            ::-1
        ]
        .copy()
    )

    if image.shape != (
        544,
        1024,
        3,
    ):

        raise RuntimeError(
            f"Unexpected RGB shape "
            f"{image.shape}: {path}"
        )

    return image


def load_target(
    sample: str,
) -> np.ndarray:

    rgb_path = (
        DATA_ROOT
        / sample
    )

    label_path = (
        rgb_path.parent
        / "labels.png"
    )

    if not label_path.is_file():

        raise FileNotFoundError(
            label_path
        )

    with Image.open(
        label_path
    ) as label_image:

        if label_image.mode != "P":

            raise RuntimeError(
                f"Expected YCOR P-mode GT, "
                f"got mode={label_image.mode}: "
                f"{label_path}"
            )

        raw = np.asarray(
            label_image,
            dtype=np.uint8,
        )

    if raw.shape != (
        544,
        1024,
    ):

        raise RuntimeError(
            f"Unexpected GT shape "
            f"{raw.shape}: "
            f"{label_path}"
        )

    observed = set(
        np.unique(
            raw
        ).tolist()
    )

    unknown = (
        observed
        -
        set(
            SOURCE_TO_TARGET
        )
    )

    if unknown:

        raise RuntimeError(
            f"Unknown YCOR GT source IDs "
            f"{sorted(unknown)} "
            f"in {label_path}"
        )

    target = TARGET_LUT[
        raw
    ]

    return target


def load_prediction(
    pred_dir: Path,
    sample: str,
) -> np.ndarray:

    filename = prediction_filename(
        sample
    )

    path = (
        pred_dir
        / filename
    )

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    prediction = np.asarray(
        Image.open(
            path
        ),
        dtype=np.uint8,
    )

    if prediction.shape != (
        544,
        1024,
    ):

        raise RuntimeError(
            f"Unexpected prediction shape "
            f"{prediction.shape}: "
            f"{path}"
        )

    observed = set(
        np.unique(
            prediction
        ).tolist()
    )

    if not observed.issubset(
        {
            0,
            1,
            2,
        }
    ):

        raise RuntimeError(
            f"Unexpected prediction IDs "
            f"{sorted(observed)} "
            f"in {path}"
        )

    return prediction


# ============================================================
# Frozen sample selection
# ============================================================

def read_samples(
    sample_file: Path,
) -> list[str]:

    if not sample_file.is_file():

        raise FileNotFoundError(
            sample_file
        )

    lines = [
        line.strip()
        for line in
        sample_file.read_text()
        .splitlines()
        if line.strip()
    ]

    samples = [
        line.split()[0]
        for line in lines
    ]

    return samples


def get_fixed_six_samples() -> list[str]:
    """
    Preferred:
        main_figure_fixed6_samples.txt

    Fallback:
        reproduce the previously frozen deterministic rule:
            first four train samples
            + all valid samples

    No model outputs are used in sample selection.
    """

    if DEFAULT_SAMPLE_FILE.is_file():

        samples = read_samples(
            DEFAULT_SAMPLE_FILE
        )

        if len(samples) != 6:

            raise RuntimeError(
                "Expected exactly 6 samples in "
                f"{DEFAULT_SAMPLE_FILE}, "
                f"found {len(samples)}"
            )

        return samples

    # --------------------------------------------------------
    # Fallback only.
    # --------------------------------------------------------

    all_samples = read_samples(
        FALLBACK_SAMPLE_FILE
    )

    if len(all_samples) != 12:

        raise RuntimeError(
            "Fallback requires the original "
            "12 frozen qualitative samples; "
            f"found {len(all_samples)}"
        )

    train_samples = [
        sample
        for sample in all_samples
        if "/train/" in sample
    ]

    valid_samples = [
        sample
        for sample in all_samples
        if (
            "/valid/" in sample
            or "/val/" in sample
        )
    ]

    samples = (
        train_samples[:4]
        +
        valid_samples
    )

    if len(samples) != 6:

        raise RuntimeError(
            "Fallback fixed-six selection "
            f"produced {len(samples)} samples."
        )

    return samples


# ============================================================
# PIL helpers
# ============================================================

def np_to_display_image(
    image: np.ndarray,
) -> Image.Image:

    pil = Image.fromarray(
        image,
        mode="RGB",
    )

    return pil.resize(
        (
            DISPLAY_W,
            DISPLAY_H,
        ),
        Image.Resampling.BILINEAR,
    )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:

    x0, y0, x1, y1 = box

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
    )

    text_w = (
        bbox[2]
        -
        bbox[0]
    )

    text_h = (
        bbox[3]
        -
        bbox[1]
    )

    x = (
        x0
        +
        (
            x1
            -
            x0
            -
            text_w
        )
        // 2
    )

    y = (
        y0
        +
        (
            y1
            -
            y0
            -
            text_h
        )
        // 2
    )

    draw.text(
        (
            x,
            y,
        ),
        text,
        fill=TEXT_COLOR,
        font=font,
    )


def sample_id(
    sample: str,
) -> str:

    # yamaha_v0/train/iid000484/rgb.jpg
    return (
        Path(sample)
        .parent
        .name
    )


# ============================================================
# Single sample rendering
# ============================================================

def process_sample(
    sample: str,
    tolerance: int,
) -> dict:

    rgb = load_rgb(
        sample
    )

    target = load_target(
        sample
    )

    pred_a = load_prediction(
        A_PRED_DIR,
        sample,
    )

    pred_dbga = load_prediction(
        DBGA_PRED_DIR,
        sample,
    )

    valid = (
        target
        != IGNORE_INDEX
    )

    gt_edge = semantic_boundary(
        target,
        valid,
    )

    errors_a = compute_boundary_errors(
        pred_a,
        target,
        tolerance,
    )

    errors_dbga = compute_boundary_errors(
        pred_dbga,
        target,
        tolerance,
    )

    metrics_a = compute_boundary_metrics(
        errors_a
    )

    metrics_dbga = compute_boundary_metrics(
        errors_dbga
    )

    return {
        "sample":
            sample,

        "rgb":
            rgb,

        "target":
            target,

        "gt_edge":
            gt_edge,

        "errors_a":
            errors_a,

        "errors_dbga":
            errors_dbga,

        "metrics_a":
            metrics_a,

        "metrics_dbga":
            metrics_dbga,
    }


def render_sample_panel(
    processed: dict,
) -> Image.Image:

    rgb = processed[
        "rgb"
    ]

    gt_edge = processed[
        "gt_edge"
    ]

    errors_a = processed[
        "errors_a"
    ]

    errors_dbga = processed[
        "errors_dbga"
    ]

    views = [
        rgb,

        visualize_gt_boundary(
            rgb,
            gt_edge,
        ),

        visualize_model_errors(
            rgb,
            errors_a,
        ),

        visualize_model_errors(
            rgb,
            errors_dbga,
        ),
    ]

    views = [
        np_to_display_image(
            view
        )
        for view in views
    ]

    width = (
        NUM_COLUMNS
        * DISPLAY_W
    )

    height = DISPLAY_H

    canvas = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        BACKGROUND,
    )

    for column, view in enumerate(
        views
    ):

        x = (
            column
            * DISPLAY_W
        )

        canvas.paste(
            view,
            (
                x,
                0,
            ),
        )

    return canvas


# ============================================================
# Legend
# ============================================================

def render_legend(
    width: int,
) -> Image.Image:

    canvas = Image.new(
        "RGB",
        (
            width,
            LEGEND_H,
        ),
        BACKGROUND,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    entries = [
        (
            "Matched boundary",
            tuple(
                COLOR_MATCHED.tolist()
            ),
        ),
        (
            "Missed GT boundary",
            tuple(
                COLOR_MISSED.tolist()
            ),
        ),
        (
            "False / spurious boundary",
            tuple(
                COLOR_FALSE.tolist()
            ),
        ),
        (
            "GT boundary",
            tuple(
                COLOR_GT.tolist()
            ),
        ),
    ]

    box_size = 28
    gap_after_box = 12
    entry_gap = 55

    # Determine total width dynamically.
    widths = []

    for text, _ in entries:

        bbox = draw.textbbox(
            (
                0,
                0,
            ),
            text,
            font=FONT_LEGEND,
        )

        text_width = (
            bbox[2]
            -
            bbox[0]
        )

        widths.append(
            box_size
            +
            gap_after_box
            +
            text_width
        )

    total_width = (
        sum(widths)
        +
        entry_gap
        *
        (
            len(entries)
            -
            1
        )
    )

    x = max(
        10,
        (
            width
            -
            total_width
        )
        // 2,
    )

    y = 28

    for (
        text,
        color,
    ), entry_width in zip(
        entries,
        widths,
    ):

        outline = (
            0,
            0,
            0,
        )

        draw.rectangle(
            (
                x,
                y,
                x + box_size,
                y + box_size,
            ),
            fill=color,
            outline=outline,
            width=2,
        )

        draw.text(
            (
                x
                +
                box_size
                +
                gap_after_box,

                y - 1,
            ),
            text,
            fill=TEXT_COLOR,
            font=FONT_LEGEND,
        )

        x += (
            entry_width
            +
            entry_gap
        )

    return canvas


# ============================================================
# Complete figure
# ============================================================

def render_main_figure(
    processed_samples: list[dict],
) -> Image.Image:

    figure_width = (
        LEFT_LABEL_W
        +
        NUM_COLUMNS
        * DISPLAY_W
    )

    row_height = (
        DISPLAY_H
        +
        ROW_GAP
    )

    figure_height = (
        HEADER_H
        +
        len(
            processed_samples
        )
        * row_height
        +
        LEGEND_H
        +
        BOTTOM_MARGIN
    )

    canvas = Image.new(
        "RGB",
        (
            figure_width,
            figure_height,
        ),
        BACKGROUND,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    draw.rectangle(
        (
            0,
            0,
            figure_width,
            HEADER_H,
        ),
        fill=HEADER_BACKGROUND,
    )

    for column, title in enumerate(
        COLUMN_TITLES
    ):

        x0 = (
            LEFT_LABEL_W
            +
            column
            * DISPLAY_W
        )

        x1 = (
            x0
            +
            DISPLAY_W
        )

        draw_centered_text(
            draw,
            (
                x0,
                0,
                x1,
                HEADER_H,
            ),
            title,
            FONT_HEADER,
        )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    y = HEADER_H

    for processed in processed_samples:

        sample = processed[
            "sample"
        ]

        sid = sample_id(
            sample
        )

        panel = render_sample_panel(
            processed
        )

        # sample-label background
        draw.rectangle(
            (
                0,
                y,
                LEFT_LABEL_W,
                y + DISPLAY_H,
            ),
            fill=LABEL_BACKGROUND,
        )

        draw_centered_text(
            draw,
            (
                0,
                y,
                LEFT_LABEL_W,
                y + DISPLAY_H,
            ),
            sid,
            FONT_SAMPLE,
        )

        canvas.paste(
            panel,
            (
                LEFT_LABEL_W,
                y,
            ),
        )

        # horizontal separator
        separator_y = (
            y
            +
            DISPLAY_H
        )

        draw.line(
            (
                0,
                separator_y,
                figure_width,
                separator_y,
            ),
            fill=GRID_COLOR,
            width=2,
        )

        y += row_height

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = render_legend(
        figure_width
    )

    canvas.paste(
        legend,
        (
            0,
            y,
        ),
    )

    return canvas


# ============================================================
# Diagnostics
# ============================================================

def save_diagnostics(
    processed_samples: list[dict],
    output_path: Path,
    tolerance: int,
) -> None:

    output = {
        "boundary_mode":
            "traversable_non_traversable",

        "classes":
            {
                "traversable": 1,
                "non_traversable": 2,
                "ignore": 255,
            },

        "tolerance_pixels":
            tolerance,

        "seed":
            123,

        "models":
            [
                "A",
                "D-BGA",
            ],

        "samples":
            [],
    }

    for processed in processed_samples:

        output[
            "samples"
        ].append(
            {
                "sample":
                    processed[
                        "sample"
                    ],

                "A": (
                    processed[
                        "metrics_a"
                    ]
                ),

                "D-BGA": (
                    processed[
                        "metrics_dbga"
                    ]
                ),
            }
        )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )


def print_diagnostics(
    processed_samples: list[dict],
) -> None:

    print()
    print(
        "=" * 100
    )

    print(
        "FIXED-6 PER-SAMPLE BF1-TRAV@3 DIAGNOSTIC"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Sample':<14}"
        f"{'A BF1':>12}"
        f"{'D-BGA BF1':>14}"
        f"{'Delta':>12}"
    )

    for processed in processed_samples:

        sid = sample_id(
            processed[
                "sample"
            ]
        )

        a_f1 = (
            processed[
                "metrics_a"
            ][
                "f1"
            ]
        )

        d_f1 = (
            processed[
                "metrics_dbga"
            ][
                "f1"
            ]
        )

        delta = (
            d_f1
            -
            a_f1
        )

        print(
            f"{sid:<14}"
            f"{a_f1:>12.6f}"
            f"{d_f1:>14.6f}"
            f"{delta:>+12.6f}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-six YCOR traversability "
            "boundary-error qualitative figure."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=BOUNDARY_TOLERANCE,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.tolerance != 3:

        raise RuntimeError(
            "Formal primary BF1-Trav uses tolerance=3. "
            "This script intentionally refuses to change it."
        )

    output_dir = (
        args.output_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_dir = (
        output_dir
        / "panels"
    )

    panel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = get_fixed_six_samples()

    print(
        "=" * 100
    )

    print(
        "YCOR FIXED-6 BOUNDARY QUALITATIVE EXPORT"
    )

    print(
        "=" * 100
    )

    print(
        f"Tolerance       : {args.tolerance} px"
    )

    print(
        "Models          : A seed123, D-BGA seed123"
    )

    print(
        "Boundary        : class 1 <-> class 2"
    )

    print(
        "Selection rule  : frozen fixed-six sample list"
    )

    print()

    processed_samples = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        print(
            f"[{index}/6] {sample}"
        )

        processed = process_sample(
            sample,
            tolerance=args.tolerance,
        )

        processed_samples.append(
            processed
        )

        panel = render_sample_panel(
            processed
        )

        sid = sample_id(
            sample
        )

        panel_path = (
            panel_dir
            / f"{index:02d}_{sid}.png"
        )

        panel.save(
            panel_path,
            dpi=(
                args.dpi,
                args.dpi,
            ),
        )

    # --------------------------------------------------------
    # Main figure
    # --------------------------------------------------------

    figure = render_main_figure(
        processed_samples
    )

    main_path = (
        output_dir
        / "boundary_qualitative_fixed6.png"
    )

    figure.save(
        main_path,
        dpi=(
            args.dpi,
            args.dpi,
        ),
        optimize=True,
    )

    # Uncompressed archival version.
    archival_path = (
        output_dir
        / "boundary_qualitative_fixed6_lossless.png"
    )

    figure.save(
        archival_path,
        dpi=(
            args.dpi,
            args.dpi,
        ),
        compress_level=0,
    )

    # --------------------------------------------------------
    # Frozen sample manifest
    # --------------------------------------------------------

    sample_manifest = (
        output_dir
        / "boundary_fixed6_samples.txt"
    )

    sample_manifest.write_text(
        "\n".join(
            samples
        )
        + "\n"
    )

    # --------------------------------------------------------
    # Diagnostic JSON
    # --------------------------------------------------------

    diagnostic_path = (
        output_dir
        / "boundary_fixed6_diagnostics.json"
    )

    save_diagnostics(
        processed_samples,
        diagnostic_path,
        tolerance=args.tolerance,
    )

    print_diagnostics(
        processed_samples
    )

    print()
    print(
        "=" * 100
    )

    print(
        "YCOR BOUNDARY QUALITATIVE EXPORT: PASS"
    )

    print(
        "=" * 100
    )

    print(
        "Main figure:"
    )

    print(
        main_path
    )

    print(
        "Archival figure:"
    )

    print(
        archival_path
    )

    print(
        "Diagnostics:"
    )

    print(
        diagnostic_path
    )

    print(
        "Panels:"
    )

    print(
        panel_dir
    )


if __name__ == "__main__":
    main()
