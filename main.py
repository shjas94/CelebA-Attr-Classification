import argparse
from models.models import *
from utils.utils import *
from trainer import *

def main(cfg):
    
    '''
    Make sure img_align_celeba.zip, list_attr_celeba.csv.zip, list_eval_partition.csv.zip files 
    are under data_path!!!
    '''
    
    model = get_model(cfg.model)
    make_dirs([cfg.save_path])
    # Unzip Image & Partition File & Attr File #
    unzip_img_files(cfg.data_path)
    unzip_files(cfg.data_path, 'list_eval_partition.zip', 'list_eval_partition.csv')
    unzip_files(cfg.data_path, 'list_attr_celeba.zip', 'list_attr_celeba.csv')
    ############################################ 
    trainer(cfg, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='Root path of data')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='Dir path to save model weights')
    parser.add_argument('--seed_num', type=int, default=42, help='Random seed number')
    parser.add_argument('--epoch', type=int, default=30, help='Total Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-06, help='Learning Rate')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=('adam', 'adamw'))
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--scheduler', type=str, default='cosinewarmup', choices=('none', 'cosinewarmup'))
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--model', type=str, default='effnetV2s', choices=('customresnet', 'customefficientnet','effnetv2s'))
    parser.add_argument('--patience', type=int, default=5, help='Patience time for early stopping')
    parser.add_argument('--wandb', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--run_name', type=str)
    cfg = parser.parse_args()
    main(cfg)