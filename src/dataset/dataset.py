import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.utils import neg_sample
from src.model.data_augmentation import Crop, Mask, Reorder
from src.model.data_augmentation import AUGMENTATIONS


def load_specified_dataset(model_name, config):
    if model_name in ['KGSCL']:
        return KGSCLDataset
    else:
        return SequentialDataset


class BaseSequentialDataset(Dataset):
    def __init__(self, num_item, config, data_pair, additional_data_dict=None, train=True):
        super(BaseSequentialDataset, self).__init__()
        self.num_item = num_item
        self.config = config
        self.train = train
        self.dataset = config.dataset
        self.max_len = config.max_len
        self.item_seq = data_pair[0]
        self.label = data_pair[1]

    def get_SRtask_input(self, idx):
        item_seq = self.item_seq[idx]
        target = self.label[idx]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def __getitem__(self, idx):
        return self.get_SRtask_input(idx)

    def __len__(self):
        return len(self.item_seq)

    def collate_fn(self, x):
        return [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]


class SequentialDataset(BaseSequentialDataset):
    def __init__(self, num_item, config, data_pair, additional_data_dict=None, train=True):
        super(SequentialDataset, self).__init__(num_item, config, data_pair, additional_data_dict, train)


class KGSCLDataset(BaseSequentialDataset):
    """
    Use KG relations to guide the data augmentation for contrastive learning.
    """

    def __init__(self, num_items, config, data_pair, additional_data_dict=None, train=True):
        super(KGSCLDataset, self).__init__(num_items, config, data_pair, additional_data_dict, train)
        self.max_item = config.max_item
        self.insert_ratio = config.insert_ratio
        self.substitute_ratio = config.substitute_ratio
        self.kg_relation_dict = additional_data_dict['kg_relation_dict']
        self.co_occurrence_dict = additional_data_dict['co_occurrence_dict']

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # raw data
        origin_item_seq = self.item_seq[index]
        target = self.label[index]

        # standardized data
        seq_len = len(origin_item_seq) if len(origin_item_seq) < self.max_len else self.max_len
        item_seq = origin_item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]
        assert len(item_seq) == self.max_len

        # aug seq 1
        aug_seq_1 = self.KG_guided_augmentation(origin_item_seq)
        aug_seq_len_1 = len(aug_seq_1) if len(aug_seq_1) < self.max_len else self.max_len
        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        assert len(aug_seq_1) == self.max_len

        # aug seq 2
        aug_seq_2 = self.KG_guided_augmentation(origin_item_seq)
        aug_seq_len_2 = len(aug_seq_2) if len(aug_seq_2) < self.max_len else self.max_len
        aug_seq_2 = aug_seq_2[-self.max_len:]
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_2) == self.max_len

        # augment target item
        aug_target, pos_item_set = self.target_substitution(target)
        pos_item_set = pos_item_set + [0] * (60 - len(pos_item_set))

        batch_tensors = (torch.tensor(item_seq, dtype=torch.long),
                         torch.tensor(seq_len, dtype=torch.long),
                         torch.tensor(target, dtype=torch.long),
                         torch.tensor(aug_seq_1, dtype=torch.long),
                         torch.tensor(aug_seq_len_1, dtype=torch.long),
                         torch.tensor(aug_seq_2, dtype=torch.long),
                         torch.tensor(aug_seq_len_2, dtype=torch.long),
                         torch.tensor(aug_target, dtype=torch.long),
                         torch.tensor(pos_item_set, dtype=torch.long))

        return batch_tensors

    def KG_guided_augmentation(self, item_seq):
        if random.random() < 0.5:
            return self.KG_insert(item_seq)
        return self.KG_substitute(item_seq)

    def KG_insert(self, item_seq):
        copied_item_seq = copy.deepcopy(item_seq)
        insert_num = int(self.insert_ratio * len(copied_item_seq))
        insert_index = random.sample([i for i in range(len(copied_item_seq))], k=insert_num)
        new_item_seq = []
        for index, item in enumerate(copied_item_seq):
            new_item_seq.append(item)
            if index in insert_index:
                shifted_item = item - 1  # origin item id
                insert_candidates = self.kg_relation_dict[shifted_item]['c']  # c: complement
                if len(insert_candidates) > 0:  # if complement items exist
                    insert_frequency = self.co_occurrence_dict[shifted_item]['c']
                    insert_item = np.random.choice(insert_candidates, size=1, p=insert_frequency)[0]
                    shifted_insert_item = insert_item + 1
                    new_item_seq.append(shifted_insert_item)
                else:
                    new_item_seq.append(item)  # Item-repeat
        return new_item_seq

    def KG_substitute(self, item_seq):
        copied_item_seq = copy.deepcopy(item_seq)
        substitute_num = int(self.substitute_ratio * len(copied_item_seq))
        substitute_index = random.sample([i for i in range(len(copied_item_seq))], k=substitute_num)
        new_item_seq = []
        for index, item in enumerate(copied_item_seq):
            if index in substitute_index:
                shifted_item = item - 1
                substitute_candidates = self.kg_relation_dict[shifted_item]['s']  # s: substitute
                if len(substitute_candidates) > 0:  # if substitute items exist
                    substitute_frequency = self.co_occurrence_dict[shifted_item]['s']
                    substitute_item = np.random.choice(substitute_candidates, size=1, p=substitute_frequency)[0]
                    shifted_substitute_item = substitute_item + 1
                    new_item_seq.append(shifted_substitute_item)
                else:
                    new_item_seq.append(item)
                    new_item_seq.append(item)  # Item-repeat
            else:
                new_item_seq.append(item)
        return new_item_seq

    def target_substitution(self, target_item):
        shifted_target_item = target_item - 1
        substitute_candidates = self.kg_relation_dict[shifted_target_item]['s']  # s: substitute
        if len(substitute_candidates) == 0:
            return target_item, []  # if no substitute items, don't change
        substitute_frequency = self.co_occurrence_dict[shifted_target_item]['s']
        substitute_item = np.random.choice(substitute_candidates, size=1, p=substitute_frequency)[0]
        shifted_substitute_item = substitute_item + 1
        substitute_candidates = [item + 1 for item in substitute_candidates]
        substitute_candidates.remove(shifted_substitute_item)
        return shifted_substitute_item, substitute_candidates


class MISPPretrainDataset(Dataset):
    """
    Masked Item & Segment Prediction (MISP)
    """

    def __init__(self, num_items, config, data_pair, additional_data_dict=None):
        self.mask_id = num_items
        self.mask_ratio = config.mask_ratio
        self.num_items = num_items + 1
        self.config = config
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.max_len = config.max_len
        self.long_sequence = []

        for seq in self.item_seq:
            self.long_sequence.extend(seq)

    def __len__(self):
        return len(self.item_seq)

    def __getitem__(self, index):
        sequence = self.item_seq[index]  # pos_items

        # Masked Item Prediction
        masked_item_sequence = []
        neg_items = []
        pos_items = sequence

        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.num_items))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.num_items))

        assert len(masked_item_sequence) == len(sequence)
        assert len(pos_items) == len(sequence)
        assert len(neg_items) == len(sequence)

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.mask_id] * sample_length + sequence[
                                                                                             start_id + sample_length:]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (
                    len(sequence) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (
                    len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # crop sequence
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = pos_items + [0] * pad_len
        neg_items = neg_items + [0] * pad_len
        masked_segment_sequence = masked_segment_sequence + [0] * pad_len
        pos_segment = pos_segment + [0] * pad_len
        neg_segment = neg_segment + [0] * pad_len

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long))
        return cur_tensors


class MIMPretrainDataset(Dataset):
    def __init__(self, num_items, config, data_pair, additional_data_dict=None):
        self.mask_id = num_items
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.config = config
        self.max_len = config.max_len
        self.n_views = 2
        self.augmentations = [Crop(tao=config.crop_ratio),
                              Mask(gamma=config.mask_ratio, mask_id=self.mask_id),
                              Reorder(beta=config.reorder_ratio)]

    def __getitem__(self, index):
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=False)
        item_seq = self.item_seq[index]
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        aug_seq_tensors = (torch.tensor(aug_seq_1, dtype=torch.long),
                           torch.tensor(aug_seq_2, dtype=torch.long),
                           torch.tensor(aug_len_1, dtype=torch.long),
                           torch.tensor(aug_len_2, dtype=torch.long))

        return aug_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)


class PIDPretrainDataset(Dataset):
    def __init__(self, num_items, config, data_pair, additional_data_dict=None):
        self.num_items = num_items
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.config = config
        self.max_len = config.max_len
        self.pseudo_ratio = config.pseudo_ratio

    def __getitem__(self, index):
        item_seq = self.item_seq[index]
        pseudo_seq = []
        target = []

        for item in item_seq:
            if random.random() < self.pseudo_ratio:
                pseudo_item = neg_sample(item_seq, self.num_items)
                pseudo_seq.append(pseudo_item)
                target.append(0)
            else:
                pseudo_seq.append(item)
                target.append(1)

        pseudo_seq = pseudo_seq[-self.max_len:]
        target = target[-self.max_len:]

        pseudo_seq = pseudo_seq + [0] * (self.max_len - len(pseudo_seq))
        target = target + [0] * (self.max_len - len(target))
        assert len(pseudo_seq) == self.max_len
        assert len(target) == self.max_len
        pseudo_seq_tensors = (torch.tensor(pseudo_seq, dtype=torch.long),
                              torch.tensor(target, dtype=torch.float))

        return pseudo_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)


if __name__ == '__main__':
    index = np.arange(10)
    res = np.random.choice(index, size=1)
    print(index)
    print(res)
