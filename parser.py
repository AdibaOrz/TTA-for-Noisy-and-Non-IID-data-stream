import argparse
import math

def parse_arguments():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--model', type=str, default='resnet18', help='Backbone model to use')
    parser.add_argument('--method', type=str, default='NOTE', help='Method')
    parser.add_argument('--corruption', type=str, default='all', help='Corruption type to test')
    parser.add_argument('--concat', action='store_true', help='Concatenate corruptions to one dataset')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use pretrained checkpoint')
    parser.add_argument('--save_dir', type=str, default='logs', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    # Source/Pretrain configurations
    parser.add_argument('--src_lr', type=float, default=0.1, help='Source learning rate')
    parser.add_argument('--src_weight_decay', type=float, default=0.0005, help='Source weight decay')
    parser.add_argument('--src_momentum', type=float, default=0.9, help='Source momentum')
    parser.add_argument('--src_batch_size', type=int, default=128, help='Source batch size')
    parser.add_argument('--src_epochs', type=int, default=200, help='Source epochs')

    # NOTE configurations
    parser.add_argument('--iabn', action='store_true', help='Use IABN or not')
    parser.add_argument('--alpha', type=int, default=4, help='Source alpha')
    parser.add_argument('--memory_type', type=str, default="PBRS", help='Memory type')
    parser.add_argument('--capacity', type=int, default=64, help='Memory capacity')
    parser.add_argument('--conf_thresh', type=float, default=0.0, help='Confidence threshold for memory')

    # Online configurations - override source configurations
    parser.add_argument('--tgt_epochs', type=int, default=1, help='Target epochs')
    parser.add_argument('--tgt_lr', type=float, default=0.0001, help='Target learning rate')
    parser.add_argument('--tgt_weight_decay', type=float, default=0, help='Target weight decay')
    parser.add_argument('--tgt_batch_size', type=int, default=128, help='Target batch size')
    parser.add_argument('--tgt_use_learned_stats', action='store_true', help='Use learned stats for target')
    parser.add_argument('--tgt_bn_momentum', type=float, default=0.01, help='Target BN momentum')

    # Online-only configurations
    parser.add_argument('--update_interval', type=int, default=64, help='Target update interval')
    parser.add_argument('--temp_factor', type=float, default=1.0, help='Target temperature factor')
    parser.add_argument('--optimize', action='store_true', help='Optimize parameters online')
    parser.add_argument('--adapt', action='store_true', help='Adapt BN online')
    parser.add_argument('--weighted_loss', action='store_true', help='Use entropy-weighted loss')
    parser.add_argument('--e_margin', type=float, default=0.4*math.log(10), help='Entropy weight threshold')

    # Data configurations
    parser.add_argument('--data_file_path', type=str, default='./dataset/CIFAR-10-C', help='Dataset file path')
    parser.add_argument('--distribution', type=str, default='dirichlet', help='Distribution type')
    parser.add_argument('--dir_beta', type=float, default=0.1, help='Dirichlet beta parameter')
    parser.add_argument('--shuffle_criterion', type=str, default='class', help='Shuffle criterion')

    args = parser.parse_args()

    if args.model != 'resnet18':
        raise NotImplementedError
    if args.method != 'NOTE':
        raise NotImplementedError

    return args
