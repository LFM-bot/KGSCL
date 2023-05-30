import argparse
from src.train.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='KGSCL')
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--insert_ratio', default=0.2, type=float, help='KG-insert ratio')
    parser.add_argument('--substitute_ratio', default=0.7, type=float, help='KG-substitute ratio')
    parser.add_argument('--tem1', default=1., type=float, help='view-view CL temperature')
    parser.add_argument('--tem2', default=1., type=float, help='view-target CL temperature')
    parser.add_argument('--sim', default='dot', type=str, choices=['dot', 'cos'], help='InfoNCE loss similarity type')
    parser.add_argument('--lamda1', default=0.1, type=float, help='view-view CL loss weight')
    parser.add_argument('--lamda2', default=1.0, type=float, help='view-target CL loss weight')
    # Data
    parser.add_argument('--dataset', default='toys', type=str)
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--separator', default=' ', type=str, help='separator to split item sequence')
    # Training
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--train_batch', default=512, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0., type=float, help='l2 normalization')  # 1e-6
    parser.add_argument('--patience', default=10, help='early stop patience')
    parser.add_argument('--mark', default='', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)

    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS', type=str, help='[LS, LS_R@0.x, PS]')
    parser.add_argument('--eval_mode', default='full', help='[uni100, pop100, full]')
    parser.add_argument('--k', default=[5, 10], help='rank k for each metric')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--valid_metric', default='hit@10', help='indicator to apply early stop')

    config = parser.parse_args()

    trainer = Trainer(config)
    trainer.start_training()


