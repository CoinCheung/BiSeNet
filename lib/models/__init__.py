

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
}
