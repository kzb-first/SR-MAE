import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='FSRCNN_pre_6h_10v', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=12, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pre_len', default=0, type=int)

    # model parameters
    # parser.add_argument('--img_size', default=(32,64)) 
    parser.add_argument('--model_name',default='FSRCNN',type=str)
    parser.add_argument('--optimizer',default='adamw',type=str)
    # Training parameters
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    rmse = exp.test(args)
