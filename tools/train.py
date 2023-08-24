import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from vision.engine.load_data import load_images
from vision.engine.trainer import train_loop
from vision.engine.build_models import building_models
from vision.models.base_models import get_base_models


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--memory", default=0, type=int, help="set gpu memory limit")
parser.add_argument("--version", default="version-0.2", help="version running")
parser.add_argument("--save_dir", default="../runs", help="path result ")
parser.add_argument("--epochs", default=1, type=int, help="epochs training")
parser.add_argument("--bath_size", default=8, type=int, help="bath size training")
parser.add_argument("--verbose", default=1, type=int, help="verbose training")
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume the most recent training')
parser.add_argument("-yaml", default="../data/dataset.yaml", help="continues train next epochs")


def main(dir_train, dir_val, save_dir, models_name="EfficientNetB0", dsize=(224, 224), mode="binary", grayscale=False, shape_input=(224, 224, 3), num_class=2, unfree=False, top_layers=20):
    #load data
    train_data, train_labels = load_images(dir_train, target_size=dsize, mode=mode, grayscale=grayscale)
    val_data, val_labels = load_images(dir_val, target_size=dsize, mode=mode, grayscale=grayscale)

    # build model
    model = building_models(get_base_models(models_name), shape_input=shape_input, num_class=num_class, unfree=unfree, top_layers=top_layers)

    # training models
    train_loop(model, train_data, train_labels, val_data, val_labels, save_dir)

    pass


def check_args(args):
    return True

if __name__ == '__main__':
    args = vars(parser.parse_args())
    if check_args(args):
        main()
    else:
        print("args error!!")
    pass
