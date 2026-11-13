import torch.nn.functional as F

from .pidnet_external.pidnet import PIDNet


class PIDNetSExternal(PIDNet):
    """
    PIDNet-S inference adapter for the existing BiSeNet evaluator.

    Input:
        N x 3 x H x W

    Output:
        (N x 3 x H x W,)

    The tuple form matches the existing BiSeNet evaluation interface.
    """

    def __init__(
        self,
        n_classes,
        aux_mode="eval",
    ):
        super().__init__(
            m=2,
            n=3,
            num_classes=n_classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=False,
        )

        self.aux_mode = aux_mode

    def forward(self, x):
        logits = super().forward(x)

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        return (logits,)
