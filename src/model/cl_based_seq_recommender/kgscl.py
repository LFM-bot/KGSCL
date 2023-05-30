"Knowledge-Guided Semantically Consistent Contrastive Learning for Sequential Recommendation"

import copy
import torch.nn.functional as F
from easydict import EasyDict
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from src.model.loss import InfoNCELoss
from src.model.sequential_encoder import Transformer
from src.utils.utils import HyperParamDict


class KGSCL(AbstractRecommender):
    def __init__(self, num_items, config, kg_map):
        super(KGSCL, self).__init__(num_items, config)
        self.embed_size = config.embed_size
        self.tem1 = config.tem1
        self.tem2 = config.tem2
        self.lamda1 = config.lamda1
        self.lamda2 = config.lamda2

        self.initializer_range = config.initializer_range
        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.input_layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout)
        self.trm_encoder = Transformer(embed_size=self.embed_size,
                                       ffn_hidden=config.ffn_hidden,
                                       num_blocks=config.num_blocks,
                                       num_heads=config.num_heads,
                                       attn_dropout=config.attn_dropout,
                                       hidden_dropout=config.hidden_dropout,
                                       layer_norm_eps=config.layer_norm_eps)
        self.nce_loss = InfoNCELoss(temperature=self.tem1,
                                    similarity_type=config.sim)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train_forward(self, item_seq, seq_len, target, *args, **kwargs):
        """
        Args:
            item_seq: [batch, max_len]
            seq_len: [batch]
            target: [batch]
        """
        v2v_aug_seq, v2v_aug_len = args[0], args[1]
        v2t_aug_seq, v2t_aug_len = args[2], args[3]
        aug_target, pos_item_set = args[4], args[5]

        # original seq encoding
        seq_embedding = self.preference_encoding(item_seq, seq_len)  # [B, D]
        # v2v aug seq encoding
        v2v_seq_embedding = self.preference_encoding(v2v_aug_seq, v2v_aug_len)  # [B, D]
        # v2t aug seq encoding
        v2t_seq_embedding = self.preference_encoding(v2t_aug_seq, v2t_aug_len)  # [B, D]

        # rec loss
        logits = seq_embedding @ self.item_embedding.weight.t()
        rec_loss = self.cross_entropy(logits, target)
        # view-view (v2v) CL loss
        v2v_loss = self.nce_loss(seq_embedding, v2t_seq_embedding)
        # view-target (v2t) CL loss
        v2t_logits = v2v_seq_embedding @ self.item_embedding.weight.t()
        # mask positive items
        v2t_logits = torch.scatter(v2t_logits, 1, pos_item_set, float('-inf'))
        v2t_loss = self.cross_entropy(v2t_logits / self.tem2, aug_target)

        return rec_loss + self.lamda1 * v2v_loss + self.lamda2 * v2t_loss

    def forward(self, item_seq, seq_len, *args, **kwargs):
        item_vectors = self.position_encoding(item_seq)
        seq_embeddings = self.trm_encoder(item_seq, item_vectors)
        seq_embedding = self.gather_index(seq_embeddings, seq_len - 1)

        # get prediction
        logits = seq_embedding @ self.item_embedding.weight.t()

        return logits

    def position_encoding(self, item_input):
        seq_embedding = self.item_embedding(item_input)
        position = torch.arange(self.max_len, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_dropout(self.input_layer_norm(seq_embedding))

        return seq_embedding

    def preference_encoding(self, item_seq, seq_len):
        item_vectors = self.position_encoding(item_seq)
        seq_embeddings = self.trm_encoder(item_seq, item_vectors)  # [B, L, D]
        seq_embedding = self.gather_index(seq_embeddings, seq_len - 1)  # [B, D]

        return seq_embedding


def KGSCL_config():
    parser = HyperParamDict('KGSCL default hyper-parameters')
    parser.add_argument('--model', default='KGSCL')
    parser.add_argument('--model_type', default='Knowledge', choices=['Sequential', 'Knowledge'])
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')
    parser.add_argument('--insert_ratio', default=0.2, type=float, help='KG-insert ratio')
    parser.add_argument('--substitute_ratio', default=0.7, type=float, help='KG-substitute ratio')
    parser.add_argument('--tem1', default=1., type=float, help='view-view CL temperature')
    parser.add_argument('--tem2', default=1., type=float, help='view-target CL temperature')
    parser.add_argument('--sim', default='dot', type=str, choices=['dot', 'cos'], help='InfoNCE loss similarity type')
    parser.add_argument('--lamda1', default=0.1, type=float, help='view-view CL loss weight')
    parser.add_argument('--lamda2', default=1.0, type=float, help='view-target CL loss weight')
    parser.add_argument('--kg_data_type', default='other', type=str, choices=['pretrain', 'jointly_train', 'other'])
    parser.add_argument('--loss_type', default='CUSTOM', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])

    return parser


if __name__ == '__main__':
    a = torch.randn(1, 3)
    b = torch.randn(5, 3)
    res = a.expand_as(b)
    print(res.size())
