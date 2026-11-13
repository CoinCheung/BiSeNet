#!/usr/bin/python
# -*- encoding: utf-8 -*-

import lib.data.transform_cv2 as T

from lib.data.rugd3_dataset import RUGD3Dataset


class RUGD3PIDNetEvalDataset(RUGD3Dataset):
    """
    PIDNet-S evaluation dataset.

    BaseDataset.get_image() already converts OpenCV BGR -> RGB.

    Therefore no additional channel reversal is required here.
    We only apply the ImageNet RGB normalization used during
    PIDNet-S training.
    """

    def __init__(
        self,
        dataroot,
        annpath,
        trans_func=None,
        mode="train",
    ):
        super().__init__(
            dataroot,
            annpath,
            trans_func,
            mode,
        )

        self.to_tensor = T.ToTensor(
            mean=(
                0.485,
                0.456,
                0.406,
            ),
            std=(
                0.229,
                0.224,
                0.225,
            ),
        )
