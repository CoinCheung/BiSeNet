#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)


DATA_ROOT = Path(
    "/root/autodl-tmp/datasets/"
    "YCOR-external/data"
)

EXT = Path(
    "experiments/external_ycor_zero_shot"
)

QUAL_FILE = (
    EXT /
    "protocol/qualitative_samples.txt"
)

OUT_ROOT = (
    EXT /
    "figures/qualitative_seed123"
)

MASK_DIR = (
    OUT_ROOT /
    "panels_mask"
)

OVERLAY_DIR = (
    OUT_ROOT /
    "panels_overlay"
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OVERLAY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PRED_DIRS = {
    "A":
        EXT /
        "results/ycor_all/"
        "a_seed123/"
        "qualitative_predictions",

    "D-BGA":
        EXT /
        "results/ycor_all/"
        "dbga_seed123/"
        "qualitative_predictions",

    "PIDNet-S":
        EXT /
        "results/ycor_all/"
        "pidnets_seed123/"
        "qualitative_predictions",
}


# ------------------------------------------------------------
# Frozen target mapping
# ------------------------------------------------------------

SOURCE_TO_TARGET = {
    0: 255,  # background -> ignore
    1: 1,    # smooth trail
    2: 1,    # traversable grass
    3: 1,    # rough trail
    4: 2,    # puddle
    5: 2,    # obstacle
    6: 2,    # low vegetation
    7: 2,    # high vegetation
    8: 0,    # sky
}

LUT = np.full(
    256,
    255,
    dtype=np.uint8,
)

for source, target in (
    SOURCE_TO_TARGET.items()
):
    LUT[source] = target


# ------------------------------------------------------------
# Paper visualization palette
#
# 0 sky             blue
# 1 traversable     green
# 2 non-traversable red
# 255 ignore        dark gray
# ------------------------------------------------------------

COLORS = {
    0: (
        70,
        130,
        180,
    ),

    1: (
        65,
        170,
        85,
    ),

    2: (
        210,
        70,
        70,
    ),

    255: (
        80,
        80,
        80,
    ),
}


DISPLAY_W = 512
DISPLAY_H = 272

HEADER_H = 54
FOOTER_H = 52

COLUMN_NAMES = [
    "RGB",
    "Ground Truth",
    "BiSeNetV2 (A)",
    "D-BGA",
    "PIDNet-S",
]


def get_font(size):

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for path in candidates:

        p = Path(path)

        if p.is_file():
            return ImageFont.truetype(
                str(p),
                size,
            )

    return ImageFont.load_default()


FONT_HEADER = get_font(28)
FONT_SAMPLE = get_font(22)
FONT_LEGEND = get_font(20)


def colorize(mask):

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    out = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    unknown = (
        set(
            np.unique(
                mask
            ).tolist()
        )
        -
        set(COLORS)
    )

    if unknown:
        raise RuntimeError(
            f"Unexpected mask IDs: "
            f"{sorted(unknown)}"
        )

    for label, color in (
        COLORS.items()
    ):

        out[
            mask == label
        ] = color

    return Image.fromarray(
        out,
        mode="RGB",
    )


def overlay(
    rgb,
    mask,
    alpha=0.48,
):

    rgb_arr = np.asarray(
        rgb,
        dtype=np.float32,
    )

    color_arr = np.asarray(
        colorize(mask),
        dtype=np.float32,
    )

    valid = (
        np.asarray(mask)
        != 255
    )

    out = rgb_arr.copy()

    out[valid] = (
        (1.0 - alpha)
        * rgb_arr[valid]
        +
        alpha
        * color_arr[valid]
    )

    return Image.fromarray(
        np.clip(
            out,
            0,
            255,
        ).astype(
            np.uint8
        )
    )


def pred_filename(
    sample,
):

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


def load_sample(sample):

    rgb_path = (
        DATA_ROOT /
        sample
    )

    gt_path = (
        rgb_path.parent /
        "labels.png"
    )

    if not rgb_path.is_file():
        raise FileNotFoundError(
            rgb_path
        )

    if not gt_path.is_file():
        raise FileNotFoundError(
            gt_path
        )

    rgb = Image.open(
        rgb_path
    ).convert(
        "RGB"
    )

    if rgb.size != (
        1024,
        544,
    ):
        raise RuntimeError(
            f"Bad RGB size: "
            f"{rgb_path} "
            f"{rgb.size}"
        )

    with Image.open(
        gt_path
    ) as im:

        if im.mode != "P":
            raise RuntimeError(
                f"GT must be P mode: "
                f"{gt_path} "
                f"mode={im.mode}"
            )

        raw_gt = np.asarray(
            im,
            dtype=np.uint8,
        )

    unknown = (
        set(
            np.unique(
                raw_gt
            ).tolist()
        )
        -
        set(
            SOURCE_TO_TARGET
        )
    )

    if unknown:
        raise RuntimeError(
            f"Unknown YCOR source IDs "
            f"{sorted(unknown)} "
            f"in {gt_path}"
        )

    gt = LUT[
        raw_gt
    ]

    preds = {}

    pred_name = (
        pred_filename(
            sample
        )
    )

    for model, root in (
        PRED_DIRS.items()
    ):

        path = (
            root /
            pred_name
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        pred = np.asarray(
            Image.open(
                path
            ),
            dtype=np.uint8,
        )

        if pred.shape != (
            544,
            1024,
        ):
            raise RuntimeError(
                f"Bad prediction size: "
                f"{path} "
                f"{pred.shape}"
            )

        ids = set(
            np.unique(
                pred
            ).tolist()
        )

        if not ids.issubset(
            {
                0,
                1,
                2,
            }
        ):
            raise RuntimeError(
                f"Unexpected prediction IDs "
                f"{sorted(ids)} "
                f"in {path}"
            )

        preds[
            model
        ] = pred

    return (
        rgb,
        gt,
        preds,
    )


def resize_display(im):

    return im.resize(
        (
            DISPLAY_W,
            DISPLAY_H,
        ),
        Image.Resampling.BILINEAR,
    )


def draw_centered_text(
    draw,
    box,
    text,
    font,
):

    x0, y0, x1, y1 = box

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
    )

    tw = (
        bbox[2]
        - bbox[0]
    )

    th = (
        bbox[3]
        - bbox[1]
    )

    x = (
        x0
        +
        (
            x1
            - x0
            - tw
        ) // 2
    )

    y = (
        y0
        +
        (
            y1
            - y0
            - th
        ) // 2
    )

    draw.text(
        (
            x,
            y,
        ),
        text,
        fill=(
            0,
            0,
            0,
        ),
        font=font,
    )


def make_legend(
    width,
):

    legend = Image.new(
        "RGB",
        (
            width,
            FOOTER_H,
        ),
        (
            255,
            255,
            255,
        ),
    )

    draw = ImageDraw.Draw(
        legend
    )

    entries = [
        (
            "Sky",
            COLORS[0],
        ),

        (
            "Traversable",
            COLORS[1],
        ),

        (
            "Non-traversable",
            COLORS[2],
        ),

        (
            "Ignore",
            COLORS[255],
        ),
    ]

    total_entry_w = 250

    start_x = max(
        20,
        (
            width
            - total_entry_w
            * len(entries)
        ) // 2,
    )

    y = 14

    for i, (
        name,
        color,
    ) in enumerate(
        entries
    ):

        x = (
            start_x
            +
            i
            * total_entry_w
        )

        draw.rectangle(
            (
                x,
                y,
                x + 24,
                y + 24,
            ),
            fill=color,
            outline=(
                0,
                0,
                0,
            ),
        )

        draw.text(
            (
                x + 34,
                y,
            ),
            name,
            fill=(
                0,
                0,
                0,
            ),
            font=FONT_LEGEND,
        )

    return legend


def make_panel(
    sample,
    use_overlay,
):

    (
        rgb,
        gt,
        preds,
    ) = load_sample(
        sample
    )

    rgb_disp = (
        resize_display(
            rgb
        )
    )

    if use_overlay:

        gt_view = overlay(
            rgb,
            gt,
        )

        a_view = overlay(
            rgb,
            preds["A"],
        )

        d_view = overlay(
            rgb,
            preds["D-BGA"],
        )

        p_view = overlay(
            rgb,
            preds["PIDNet-S"],
        )

    else:

        gt_view = colorize(
            gt
        )

        a_view = colorize(
            preds["A"]
        )

        d_view = colorize(
            preds["D-BGA"]
        )

        p_view = colorize(
            preds["PIDNet-S"]
        )

    views = [
        rgb_disp,
        resize_display(
            gt_view
        ),
        resize_display(
            a_view
        ),
        resize_display(
            d_view
        ),
        resize_display(
            p_view
        ),
    ]

    width = (
        DISPLAY_W
        * len(views)
    )

    height = (
        HEADER_H
        + DISPLAY_H
        + FOOTER_H
    )

    panel = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        (
            255,
            255,
            255,
        ),
    )

    draw = ImageDraw.Draw(
        panel
    )

    for index, (
        title,
        view,
    ) in enumerate(
        zip(
            COLUMN_NAMES,
            views,
        )
    ):

        x = (
            index
            * DISPLAY_W
        )

        draw_centered_text(
            draw,
            (
                x,
                0,
                x + DISPLAY_W,
                HEADER_H,
            ),
            title,
            FONT_HEADER,
        )

        panel.paste(
            view,
            (
                x,
                HEADER_H,
            ),
        )

    legend = make_legend(
        width
    )

    panel.paste(
        legend,
        (
            0,
            HEADER_H
            + DISPLAY_H,
        ),
    )

    return panel


def make_contact_sheet(
    samples,
    use_overlay,
    out_path,
):

    panels = [
        make_panel(
            sample,
            use_overlay,
        )
        for sample in samples
    ]

    panel_w = panels[0].width
    panel_h = panels[0].height

    sample_label_h = 40

    canvas = Image.new(
        "RGB",
        (
            panel_w,
            (
                panel_h
                + sample_label_h
            )
            * len(panels),
        ),
        (
            255,
            255,
            255,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    y = 0

    for sample, panel in zip(
        samples,
        panels,
    ):

        sample_id = (
            str(
                Path(sample).parent
            )
        )

        draw.text(
            (
                12,
                y + 7,
            ),
            sample_id,
            fill=(
                0,
                0,
                0,
            ),
            font=FONT_SAMPLE,
        )

        y += sample_label_h

        canvas.paste(
            panel,
            (
                0,
                y,
            ),
        )

        y += panel_h

    canvas.save(
        out_path
    )


def main():

    lines = [
        x.strip()
        for x in
        QUAL_FILE.read_text()
        .splitlines()
        if x.strip()
    ]

    samples = [
        x.split()[0]
        for x in lines
    ]

    if len(samples) != 12:
        raise RuntimeError(
            f"Expected 12 frozen samples, "
            f"found {len(samples)}"
        )

    if len(
        set(samples)
    ) != 12:
        raise RuntimeError(
            "Duplicate qualitative samples."
        )

    print(
        "Frozen qualitative samples:"
    )

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        print(
            f"{index:02d}: {sample}"
        )

        mask_panel = (
            make_panel(
                sample,
                use_overlay=False,
            )
        )

        overlay_panel = (
            make_panel(
                sample,
                use_overlay=True,
            )
        )

        stem = (
            Path(sample)
            .parent
            .name
        )

        mask_panel.save(
            MASK_DIR /
            f"{index:02d}_{stem}.png"
        )

        overlay_panel.save(
            OVERLAY_DIR /
            f"{index:02d}_{stem}.png"
        )

    # --------------------------------------------------------
    # All 12 frozen samples.
    # --------------------------------------------------------

    make_contact_sheet(
        samples,
        use_overlay=False,
        out_path=(
            OUT_ROOT /
            "contact_sheet_all12_masks.png"
        ),
    )

    make_contact_sheet(
        samples,
        use_overlay=True,
        out_path=(
            OUT_ROOT /
            "contact_sheet_all12_overlay.png"
        ),
    )

    # --------------------------------------------------------
    # Deterministic main-paper candidate:
    #
    # first four TRAIN samples in frozen ordering
    # + all VALID samples.
    #
    # Selection is independent of model outputs.
    # --------------------------------------------------------

    train_samples = [
        s
        for s in samples
        if "/train/" in s
    ]

    valid_samples = [
        s
        for s in samples
        if (
            "/valid/" in s
            or "/val/" in s
        )
    ]

    main_samples = (
        train_samples[:4]
        + valid_samples
    )

    if len(main_samples) != 6:
        raise RuntimeError(
            "Expected deterministic "
            "main figure to contain 6 samples; "
            f"found {len(main_samples)}"
        )

    (
        OUT_ROOT /
        "main_figure_fixed6_samples.txt"
    ).write_text(
        "\n".join(
            main_samples
        )
        + "\n"
    )

    make_contact_sheet(
        main_samples,
        use_overlay=False,
        out_path=(
            OUT_ROOT /
            "main_figure_fixed6_masks.png"
        ),
    )

    make_contact_sheet(
        main_samples,
        use_overlay=True,
        out_path=(
            OUT_ROOT /
            "main_figure_fixed6_overlay.png"
        ),
    )

    print()
    print(
        "YCOR QUALITATIVE FIGURE EXPORT: PASS"
    )

    print(
        "Output:",
        OUT_ROOT,
    )


if __name__ == "__main__":
    main()
