from pathlib import Path

import cv2
import numpy as np


ROOT = Path("datasets/rugd4")
LIST_FILE = ROOT / "train.txt"

channel_sum = np.zeros(3, dtype=np.float64)
channel_sq_sum = np.zeros(3, dtype=np.float64)
pixel_count = 0

for index, line in enumerate(
    LIST_FILE.read_text().splitlines()
):
    image_rel, _ = line.split(",")

    image = cv2.imread(
        str(ROOT / image_rel)
    )

    if image is None:
        raise RuntimeError(image_rel)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # 缩小只用于统计，可明显提高速度
    image = cv2.resize(
        image,
        (344, 278),
    )

    image = image.astype(
        np.float64
    ) / 255.0

    flat = image.reshape(-1, 3)

    channel_sum += flat.sum(axis=0)
    channel_sq_sum += np.square(flat).sum(axis=0)
    pixel_count += flat.shape[0]

    if (index + 1) % 500 == 0:
        print(index + 1)

mean = channel_sum / pixel_count

variance = (
    channel_sq_sum / pixel_count
    - np.square(mean)
)

std = np.sqrt(
    np.maximum(variance, 0)
)

print("mean =", tuple(mean.tolist()))
print("std  =", tuple(std.tolist()))