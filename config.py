import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--validation_freq', type=int, default=1)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default='resume_state')

    # Backbone settings
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--dilation', type=bool, default=True)
    
    # Learning settings
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip_max_norm', type=float, default=0.1)

    # Transformer settings
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=4253)
    parser.add_argument('--start_token', type=int, default=1)
    parser.add_argument('--end_token', type=int, default=2)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--pre_norm', type=int, default=True)

    # Diagnosisbot settings
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--thresholds_path', type=str, default="datasets/thresholds.pkl")
    parser.add_argument('--detector_weight_path', type=str, default="weight_path/diagnosisbot.pth")
    parser.add_argument('--t_model_weight_path', type=str, default="weight_path/mimic_t_model.pth")
    parser.add_argument('--knowledge_prompt_path', type=str, default="knowledge_path/knowledge_prompt_mimic.pkl")

    # ADA settings
    parser.add_argument('--theta', type=float, default=0.6)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0)

    # Delta settings
    parser.add_argument('--delta', type=float, default=0.01)

    # Dataset settings
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--anno_path', type=str, default='dataset/mimic_cxr/annotation.json')
    parser.add_argument('--data_dir', type=str, default='dataset/mimic_cxr/images300')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--strip_path', type=str, default='datasets/strip_list.pkl')
    parser.add_argument('--vocab_path', type=str, default='datasets/iu_xray_vocabulary.pkl')

    # Mode settings
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--resume_state', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)

    # Dataset-specific lengths
    parser.add_argument('--train_len', type=int, default=-1)
    parser.add_argument('--val_len', type=int, default=-1)
    parser.add_argument('--test_len', type=int, default=-1)

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            for key, value in config_args.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    elif args.config:
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    
    return args
