import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractRecommender(nn.Module):
    def __init__(self, num_items=0, config=None):
        super(AbstractRecommender, self).__init__()
        self.num_items = num_items
        self.loss_type = config.loss_type
        self.max_len = config.max_len
        self.dev = config.device
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, seq_len, *args, **kwargs):
        """
        Args:
            item_seq: [batch, max_len]
            seq_len: [batch]
        """
        pass

    def train_forward(self, item_seq, seq_len, target, *args, **kwargs):
        """
        Args:
            item_seq: [batch, max_len]
            seq_len: [batch]
            target: [batch]
        """
        logits = self.forward(item_seq, seq_len, *args, **kwargs)
        return self.get_loss(item_seq, logits, target)

    def get_loss(self, item_input, logits, target):
        """
        :param item_input: [batch, max_len]
        :param logits: [batch, num_items]
        :param target: [batch]
        """

        if self.loss_type.upper() == 'BCE':
            neg_item = self.get_negative_items(item_input, target, num_samples=1)
            pos_score = torch.gather(logits, -1, target.unsqueeze(-1))
            neg_score = torch.gather(logits, -1, neg_item)
            loss = -torch.mean(F.logsigmoid(pos_score) + torch.log(1 - torch.sigmoid(neg_score)).sum(-1))
        elif self.loss_type.upper() == 'BPR':  # BPR loss
            neg_item = self.get_negative_items(item_input, target, num_samples=1)
            pos_score = torch.gather(logits, -1, target.unsqueeze(-1))
            neg_score = torch.gather(logits, -1, neg_item)
            loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
        else:  # CE loss
            # prediction = F.softmax(logits, -1)
            logits = logits.view(-1, logits.size(-1))
            loss = self.loss_fct(logits, target)
            # pos_score = torch.gather(prediction, -1, target.unsqueeze(-1))
            # loss = -torch.mean(torch.log(pos_score))
        return loss

    def gather_index(self, output, index):
        """
        :param output:  [batch, max_len, H]
        :param index: [batch]
        :return: [batch, H}
        """
        gather_index = index.view(-1, 1, 1).repeat(1, 1, output.size(-1))
        gather_output = output.gather(dim=1, index=gather_index)
        return gather_output.squeeze()

    def get_target_and_length(self, target_info):
        """
        :param target_info: target information dict
        :return:
        """
        target = target_info['target']   # [batch, ep_len]
        try:
            tar_len = target_info['target_len']
        except:
            raise Exception(f"{self.__class__.__name__} requires target sequences, set use_tar_seq to true in "
                            f"experimental settings")
        return target, tar_len

    def get_negative_items(self, input_item, target, num_samples=1):
        """
        :param input_item: [batch_size, max_len]
        :param sample_size: [batch_size, num_samples]
        :return:
        """
        sample_prob = torch.ones(input_item.size(0), self.num_items, device=target.device)
        sample_prob.scatter_(-1, input_item, 0.)
        sample_prob.scatter_(-1, target.unsqueeze(-1), 0.)
        neg_items = torch.multinomial(sample_prob, num_samples)

        return neg_items

    def pack_to_batch(self, prediction):
        if prediction.dim() < 2:
            prediction = prediction.unsqueeze(0)
        return prediction

    def calc_total_params(self):
        """
        Calculate Total Parameters
        :return: number of parameters
        """
        return sum([p.nelement() for p in self.parameters()])

    def load_pretrain_model(self, pretrain_model):
        """
        load pretraining model, default: load all parameters
        """
        self.load_state_dict(pretrain_model.state_dict())
        del pretrain_model

    def MISP_pretrain_forward(self, masked_item_sequence, pos_items, neg_items,
                              masked_segment_sequence, pos_segment, neg_segment):
        """
        :param masked_item_sequence: [batch, max_len]
        :param pos_items: [batch, max_len]
        :param neg_items: [batch, max_len]
        :return: pretraining task loss
        """
        pass

    def MIM_pretrain_forward(self, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2):
        """
        :param aug_seq_1: [batch, max_len]
        :param aug_seq_2: [batch, max_len]
        :param aug_len_1: [batch]
        :param aug_len_2: [batch
        """
        pass

    def PID_pretrain_forward(self, pseudo_seq, target):
        """
        :param pseudo_seq: [batch, max_len]
        :param target: [batch, max_len]
        :return:
        """
        pass


class AbstractRLRecommender(AbstractRecommender):
    def __init__(self, num_items, config):
        super(AbstractRLRecommender, self).__init__(num_items, config)

    def sample_neg_action(self, masked_action, neg_size):
        """
        :param masked_action: [batch, max_len]
        :return: neg_action, [batch, neg_size]
        """
        sample_prob = torch.ones(masked_action.size(0), self.num_items, device=masked_action.device)
        sample_prob = sample_prob.scatter(-1, masked_action, 0.)
        neg_action = torch.multinomial(sample_prob, neg_size)

        return neg_action

    def state_transfer(self, pre_item_seq, action, seq_len):
        """
        Parameters
        ----------
        pre_item_seq: torch.LongTensor, [batch_size, max_len]
        action: torch.LongTensor, [batch_size]
        seq_len: torch.LongTensor, [batch_size]

        Return
        ------
        next_state_seq: torch.LongTensor, [batch_size, max_len]
        """
        new_item_seq = pre_item_seq.clone().detach()
        action = action.unsqueeze(-1)
        seq_len = seq_len.unsqueeze(-1)
        max_len = pre_item_seq.size(1)

        padding_col = torch.zeros_like(action, dtype=torch.long, device=action.device)
        new_item_seq = torch.cat([new_item_seq, padding_col], -1)
        new_item_seq = new_item_seq.scatter(-1, seq_len, action)
        new_item_seq = new_item_seq[:, 1:]

        new_seq_len = seq_len.squeeze() + 1
        new_seq_len[new_seq_len > max_len] = max_len

        return new_item_seq, new_seq_len

