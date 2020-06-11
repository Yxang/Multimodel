#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import h5py
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
#from apex import amp

#from tqdm import tqdm

#from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup

from utils import MyTokenizer, MyBert

import copy

np.random.seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2020)


cfg = {}
cfg['train_fraction'] = 0.1
cfg['max_query_word'] = 9
cfg['max_box_num'] = 9
cfg['bert_model_name'] = 'bert-base-uncased'
cfg['max_class_word_num'] = 11
cfg['dataloader_cfg'] = {
    'batch_size': 256,
    'num_workers': 14,
    'pin_memory': True}
cfg['epochs'] = 20
cfg['apex_opt_level'] = 'O2'
cfg['save_name'] = 'bert-base-3fc-dropout'
cfg['num_negative_sampling'] = 5
cfg['save_RAM'] = True

basic_model_cfg = {}
basic_model_cfg['pos_emb_size'] = 8
basic_model_cfg['bilstm_hidden_size'] = 768
basic_model_cfg['clf1_out'] = 512
basic_model_cfg['clf2_out'] = 128

#def LabelPassBert():

def read_label_info(path='../data/multimodal_labels.txt'):
    num_to_label = []
    with open(path, 'r') as f:
        l = f.readline()
        lines = f.readlines()
        for l in lines:
            words = l.strip().split('\t')
            num_to_label.append(words[1])
    return num_to_label

myTokenizer = MyTokenizer(cfg)


def convert_label_to_token_id(class_num_to_label, mytokenizer):
    # lens = []
    label_to_token_id = []
    for v in class_num_to_label:
        # lens.append(len(tokenizer.tokenizer.tokenize(v)))
        # class label从0开始，label_to_token_id[k]表示label k 的token
        label_to_token_id.append(mytokenizer.convert_str_to_ids(v, tensor=True, max_len=512, pad=False).view(1, -1))
    # print(np.percentile(lens, [0, 25, 50, 75, 95, 99, 100]))
    return label_to_token_id

num_to_label = read_label_info()
label_to_token_id = convert_label_to_token_id(num_to_label, myTokenizer)
#print(label_to_token_id)


mybert = MyBert(cfg['bert_model_name'])
label_to_embedding = [np.zeros((1, 768))]
for i in label_to_token_id:
    emb = mybert(i)
    emb = emb.detach().numpy()
    label_to_embedding.append(emb)
#label_to_embedding.reshape(())
print(label_to_embedding)
cat_label_to_embedding = np.concatenate(label_to_embedding, axis=0)
with open('../user_data/tmp_data/label_emb_bert_base_uncased.npy', 'wb') as f:
    np.save(f, cat_label_to_embedding)
