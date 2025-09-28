import os
from pathlib import Path
from utils.config import CFG
from argparse import ArgumentParser

NN = ("ann", "snn")
MODELS = ("resnet20", "vgg16")

parser = ArgumentParser(description='PyTorch Cifar-10 Training')
parser.add_argument('--nn', default='ann', choices=NN, type=str.lower, help='network type')
parser.add_argument('--model', default='resnet20', choices=MODELS, type=str.lower, help='model name')
parser.add_argument('--cuda', default=0, type=int, help='cuda device id')
parser.add_argument('--Tencode', default=16, type=int, metavar='N',
                    help='encoding time window size')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr1', default=1e-4, type=float,
                    metavar='LR_S1', help='initial learning rate of LTL training', dest='lr1')
parser.add_argument('--lr2', default=1e-5, type=float,
                    metavar='LR_S2', help='initial learning rate of TTFS training', dest='lr2')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save the training record (default: none)')
parser.add_argument('--local_coefficient', default=1.0, type=float,
                     help='Coefficient of Local Loss')
parser.add_argument('--beta', default=1, type=float,
                    metavar='beta', help='coefficient beta')
parser.add_argument('--gamma', default=5, type=float,
                    metavar='gamma', help='Maximum number of spikes per timestep in burst coding')
parser.add_argument('--threshold', default=3, type=float,
                     help='The potential threshold of the output layer (TTFS coding)')
parser.add_argument('--ltl_resume', default=False, action='store_true',
					help='Resume from LTL finetuned model and start ttfs learning')

args = parser.parse_args()
current_dir = Path(os.getcwd())

from CIFAR10.ANN_baseline.cifar10_resNet20_base_model import main as ResNet20
from CIFAR10.ANN_baseline.cifar10_vgg16_base_model import main as VGG16
from CIFAR10.Hybrid_coding.cifar10_main_res20 import main as SNN_ResNet20
from CIFAR10.Hybrid_coding.cifar10_main_vgg16 import main as SNN_VGG16

if __name__ == '__main__':
    cfg = CFG(
        args,
        current_dir=current_dir,
        data_dir=current_dir / "data" / "data",
        batch_size=128,
        num_workers=4,
        num_epochs=300,
        num_classes=10
        )
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # cudnn_enabled=cudnn.is_available()

    if cfg.args.nn == "ann" and cfg.args.model == "resnet20":
        ResNet20()
    elif cfg.args.nn == "ann" and cfg.args.model == "vgg16":
        VGG16()
    elif cfg.args.nn == "snn" and cfg.args.model == "resnet20":
        SNN_ResNet20()
    elif cfg.args.nn == "snn" and cfg.args.model == "vgg16":
        SNN_VGG16()
    else:
        raise ValueError("Invalid model name or nn type")