import numpy as np
import pandas as pd
import h5py
import base64
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
#from apex import amp
from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM #, BertAdam
from sklearn.metrics import accuracy_score
#from utils import MyTokenizer, MyBert, BasicDataset, NSDataset, MNSDataset, nDCGat5_Calculator
import copy

cfg = {}
cfg['train_fraction'] = 0.25
cfg['max_query_word'] = 9
cfg['max_box_num'] = 9
cfg['bert_model_name'] = 'bert-base-uncased'
cfg['max_class_word_num'] = 11
cfg['dataloader_cfg'] = {
    'batch_size': 32,
    'num_workers': 8,
    'pin_memory': True}
cfg['epochs'] = 20
cfg['apex_opt_level'] = 'O2'
cfg['save_name'] = 'bert-base-2fc'
cfg['num_negative_sampling'] = 5
cfg['save_RAM'] = True



class MyTokenizer:
    def __init__(self, cfg):
        self.tokenizer = BertTokenizer.from_pretrained(cfg['bert_model_name'])
        self.cfg = cfg

    def convert_str_to_ids(self, text, pad=True, tensor=False, max_len=None):
        if max_len is None:
            max_len = self.cfg['max_query_word']
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_text + ["[SEP]"])

        if pad:
            if len(indexed_tokens) > max_len:
                indexed_tokens = indexed_tokens[:max_len]  # + [tokenizer.tokenizer.vocab['[SEP]']]
            elif len(indexed_tokens) < max_len:
                indexed_tokens = [0] * (max_len - len(indexed_tokens)) + indexed_tokens
        if tensor:
            tokens_tensor = torch.tensor(indexed_tokens)
            return tokens_tensor
        else:
            return indexed_tokens

    def convert_label_to_token_id(self, class_num_to_label):
        #lens = []
        label_to_token_id = {}
        for k, v in class_num_to_label.items():
            #lens.append(len(tokenizer.tokenizer.tokenize(v)))
            label_to_token_id[k] = self.tokenizer.convert_str_to_ids(v,
                                                                tensor=False,
                                                                max_len=self.cfg['max_class_word_num'])
        #print(np.percentile(lens, [0, 25, 50, 75, 95, 99, 100]))
        return label_to_token_id

class Create_h5_all_processed:
    def __init__(self, tsv,target, mytokenizer, cfg):
        self.tsv = tsv
        self.target = target
        self.cfg = cfg
        self.mytokenizer = mytokenizer
        self.MAX_NUM_BOXES = self.cfg['max_box_num']

        self.WRITE_CHUNK = 1024
        class_num_to_label = self._read_label_info(path='../data/Kdd/multimodal_labels.txt')
        self.label_to_token_id = self.mytokenizer.convert_label_to_token_id(class_num_to_label)
    def processed(self):

        with h5py.File(self.target, 'w', libver='latest')as hf:

            hf.create_group('box_loc')
            hf.create_group('box_feat')
            hf.create_group('class_labels')
            hf.create_group('querys')
            hf.create_group('others')
            box_loc_h5ds = hf.create_dataset('box_loc/data',
                                             shape=(self.WRITE_CHUNK, self.MAX_NUM_BOXES, 5),
                                             chunks=(1, self.MAX_NUM_BOXES, 5),
                                             maxshape=(None, self.MAX_NUM_BOXES, 5),
                                             # compression="lzf",
                                             dtype='f')
            box_feat_h5ds = hf.create_dataset('box_feat/data',
                                              shape=(self.WRITE_CHUNK, self.MAX_NUM_BOXES, 2048),
                                              chunks=(1, self.MAX_NUM_BOXES, 2048),
                                              maxshape=(None, self.MAX_NUM_BOXES, 2048),
                                              compression="lzf",
                                              dtype='f')
            class_labels_h5ds = hf.create_dataset('class_labels/data',
                                                  shape=(self.WRITE_CHUNK, self.MAX_NUM_BOXES),
                                                  chunks=(1, self.MAX_NUM_BOXES),
                                                  maxshape=(None, self.MAX_NUM_BOXES),
                                                  # compression="lzf",
                                                  dtype='i')
            querys_h5ds = hf.create_dataset('querys/data',
                                            shape=(self.WRITE_CHUNK,),
                                            chunks=(1,),
                                            maxshape=(None,),
                                            # compression="lzf",
                                            dtype=h5py.string_dtype())
            others_h5ds = hf.create_dataset('others/data',
                                            shape=(self.WRITE_CHUNK, 5),
                                            chunks=(1, 5),
                                            maxshape=(None, 5),
                                            # compression="lzf",
                                            dtype='i')

            with open(self.tsv, 'r') as f:
                l = f.readline()
                l = f.readline()
                i = 0

                box_locs, box_feats, class_labels, querys, others = (
                    np.zeros((self.WRITE_CHUNK, self.MAX_NUM_BOXES, 5)),
                    np.zeros((self.WRITE_CHUNK, self.MAX_NUM_BOXES, 2048)),
                    np.zeros((self.WRITE_CHUNK, self.MAX_NUM_BOXES)),
                    ['' for i in range(self.WRITE_CHUNK)],
                    np.zeros((self.WRITE_CHUNK, 5))
                )

                while l:
                    num_boxes, box_loc, box_feat, class_label, query, other = self.process_line(l)
                    box_locs[i % self.WRITE_CHUNK, :num_boxes, :] = box_loc
                    box_feats[i % self.WRITE_CHUNK, :num_boxes, :] = box_feat
                    class_labels[i % self.WRITE_CHUNK, :num_boxes] = class_label
                    querys[i % self.WRITE_CHUNK] = query
                    others[i % self.WRITE_CHUNK, :] = other
                    if i % self.WRITE_CHUNK == self.WRITE_CHUNK - 1:
                        print('\rline {}'.format(i), end='')
                        box_locs, box_feats, class_labels, querys, others = self.flush_into_ds(hf, i, box_locs, box_feats,
                                                                                          class_labels, querys, others)
                    i += 1
                    l = f.readline()
                if i % self.WRITE_CHUNK != self.WRITE_CHUNK - 1:
                    self.flush_into_ds(hf, i - 1, box_locs, box_feats, class_labels, querys, others)
                    print('\rline {}'.format(i), end='')
            print()
        return

    def process_line(self, line):
        l = line.strip().split('\t')
        num_boxes = int(l[3])
        '''
        if num_boxes > MAX_NUM_BOXES:
            raise RuntimeError(f'num_boxes is large than {MAX_NUM_BOXES}, which is {num_boxes}')
        '''

        box_loc = np.frombuffer(base64.b64decode(l[4]), dtype=np.float32).reshape(num_boxes, 4)
        box_feat = np.frombuffer(base64.b64decode(l[5]), dtype=np.float32).reshape(num_boxes, 2048)
        class_label = np.frombuffer(base64.b64decode(l[6]), dtype=np.int64).reshape(num_boxes, )
        if num_boxes > self.MAX_NUM_BOXES:
            box_loc = box_loc[:self.MAX_NUM_BOXES, :]
            box_feat = box_feat[:self.MAX_NUM_BOXES, :]
            class_label = class_label[:self.MAX_NUM_BOXES, :]
        query = l[7]

        query = self.mytokenizer.convert_str_to_ids(query)


        box_label = [np.array(self.label_to_token_id[label]).reshape(1, -1) for label in class_label]
        box_label = np.concatenate(box_label, axis=0)
        other = [int(l[0]), int(l[1]), int(l[2]), int(l[3]), int(l[8])]

        h, w = other[1], other[2]
        y1, x1, y2, x2 = box_loc[:, 0], box_loc[:, 1], box_loc[:, 2], box_loc[:, 3]
        box_pos = [x1 / w,
                   y1 / h,
                   x2 / w,
                   y2 / h,
                   (x2 - x1) * (y2 - y1) / (w * h)]
        box_pos = [item.reshape(-1, 1) for item in box_pos]
        box_pos = np.concatenate(box_pos, axis=1)

        return num_boxes, box_pos, box_feat, box_label, query, other

    def flush_into_ds(self, hf, i, box_locs, box_feats, class_labels, querys, others):
        box_loc_h5ds = hf.get('box_loc/data')
        box_loc_h5ds.resize(i + 1, axis=0)
        box_loc_h5ds[i // self.WRITE_CHUNK * self.WRITE_CHUNK:i + 1, :, :] = box_locs[:i % self.WRITE_CHUNK + 1, :, :]

        box_feat_h5ds = hf.get('box_feat/data')
        box_feat_h5ds.resize(i + 1, axis=0)
        box_feat_h5ds[i // self.WRITE_CHUNK * self.WRITE_CHUNK:i + 1, :, :] = box_feats[:i % self.WRITE_CHUNK + 1, :, :]

        class_labels_h5ds = hf.get('class_labels/data')
        class_labels_h5ds.resize(i + 1, axis=0)
        class_labels_h5ds[i // self.WRITE_CHUNK * self.WRITE_CHUNK:i + 1, :] = class_labels[:i % self.WRITE_CHUNK + 1, :]

        querys_h5ds = hf.get('querys/data')
        querys_h5ds.resize(i + 1, axis=0)
        querys_h5ds[i // self.WRITE_CHUNK * self.WRITE_CHUNK:i + 1] = querys[:i % self.WRITE_CHUNK + 1]

        others_h5ds = hf.get('others/data')
        others_h5ds.resize(i + 1, axis=0)
        others_h5ds[i // self.WRITE_CHUNK * self.WRITE_CHUNK:i + 1, :] = others[:i % self.WRITE_CHUNK + 1, :]

        return (
            np.zeros((self.WRITE_CHUNK, self.MAX_NUM_BOXES, 5)),
            np.zeros((self.WRITE_CHUNK, self.MAX_NUM_BOXES, 2048)),
            np.zeros((self.WRITE_CHUNK, self.MAX_NUM_BOXES)),
            ['' for i in range(self.WRITE_CHUNK)],
            np.zeros((self.WRITE_CHUNK, 5))
        )
    @staticmethod
    def _read_label_info(path='../data/Kdd/multimodal_labels.txt'):
        num_to_label = {}
        with open(path, 'r') as f:
            l = f.readline()
            lines = f.readlines()
            for l in lines:
                words = l.strip().split('\t')
                num_to_label[int(words[0])] = words[1]
        return num_to_label
tokenizer = MyTokenizer(cfg=cfg)

train_sample = Create_h5_all_processed('../data/Kdd/train.sample.tsv', '../data/Kdd/train.sample_processed.h5', tokenizer, cfg)
#valid = Create_h5_all_processed('../data/Kdd/valid.tsv', '../data/Kdd/valid_processed.h5', tokenizer, cfg)
#train = Create_h5_all_processed('../data/Kdd/train.tsv', '../data/Kdd/train_processed.h5', tokenizer, cfg)
train_sample.processed()
#valid.processed()
#train.processed()

