import os, sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vision.models.efficientnet import EfficientNetB0


def get_base_models(models = 'EfficientNetB0', weights="imagenet", include_top=False):
    base = {
        'EfficientNetB0': EfficientNetB0(weights=weights, include_top=include_top)
    }
    return base