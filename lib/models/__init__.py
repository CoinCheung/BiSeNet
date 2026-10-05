

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .bisenetv2_boundary_sup import (
    BiSeNetV2BoundarySup,
)
from .bisenetv2_boundary_refine import (
    BiSeNetV2BoundaryRefine,
)

from .bisenetv2_boundary_full import (
    BiSeNetV2BoundaryFull,
)

from .bisenetv2_boundary_full_bga import (
    BiSeNetV2BoundaryFullBGA,
)


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,

    'bisenetv2_boundary_sup':
        BiSeNetV2BoundarySup,

    'bisenetv2_boundary_refine':
        BiSeNetV2BoundaryRefine,

    'bisenetv2_boundary_full':
        BiSeNetV2BoundaryFull,

    'bisenetv2_boundary_full_bga':
        BiSeNetV2BoundaryFullBGA,
}
