import os, sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vision.models.efficientnet import preprocess_input as prep_input_efficientnet
from vision.models.resnet import preprocess_input as prep_input_resnet


def prep_input(models_name, data):
    dict_prep_input = {
        'resnet': prep_input_resnet(data),
        'efficientnet': prep_input_efficientnet(data)
    }
    return dict_prep_input.get(models_name, lambda: "Model Name not Exist")()
    