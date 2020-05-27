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

from tqdm import tqdm

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

import json
from collections import defaultdict
from eval_utils import read_submission, get_ndcg


class MyTokenizer:
    def __init__(self, cfg):
        self.tokenizer = BertTokenizer.from_pretrained(cfg['bert_model_name'])
        self.cfg = cfg
        
    def convert_str_to_ids(self, text, pad=True, tensor=True, max_len=None):
        if max_len is None:
            max_len = self.cfg['max_query_word']
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_text + ["[SEP]"])
        
        if pad:
            if len(indexed_tokens) > max_len:
                indexed_tokens = indexed_tokens[:max_len]# + [tokenizer.tokenizer.vocab['[SEP]']]
            elif len(indexed_tokens) < max_len:
                indexed_tokens = [0] * (max_len - len(indexed_tokens)) + indexed_tokens
        if tensor:
            tokens_tensor = torch.tensor(indexed_tokens)
            return tokens_tensor
        else:
            return indexed_tokens
        
class MyBert(nn.Module):
    """
    Take 0-th index of the sequence from the last attention encoder output
    """
    def __init__(self, pretrained_model_name_or_path, *inputs, **kwargs):
        super(MyBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    def forward(self, 
                input_ids, output_all_encoded_layers=False):
        bert_out, _ = self.bert(input_ids, output_all_encoded_layers=output_all_encoded_layers)
        out = bert_out[:, 0, :]
        return out
    
class BasicDataset(data.Dataset):
    def __init__(self, h5_path, mytokenizer, cfg, label_info_path='../data/Kdd/multimodal_labels.txt', processed=False, *args, **kwargs):
        super(BasicDataset, self).__init__(*args, **kwargs)
        self.h5_path = h5_path
        self.cfg = cfg
        with h5py.File(h5_path, 'r', libver='latest') as h5file:
            self.size = h5file.get('box_loc/data').shape[0]
        
        self.mytokenizer = mytokenizer
        
        class_num_to_label = BasicDataset._read_label_info(path=label_info_path)
        self.label_to_token_id = self._convert_label_to_token_id(class_num_to_label, mytokenizer)
        
        self.processed = processed
        
    def _read_h5_file(self, index):
        with h5py.File(self.h5_path, 'r', libver='latest') as h5file:
            queris = h5file.get('querys/data')
            box_locs = h5file.get('box_loc/data')
            box_feats = h5file.get('box_feat/data')
            class_labels = h5file.get('class_labels/data')
            otherss = h5file.get('others/data')
            
            query = queris[index]
            box_loc = box_locs[index]
            box_feature = box_feats[index]
            class_label = class_labels[index]
            others = otherss[index]
        return query, box_loc, box_feature, class_label, others
    
    
    def __getitem__(self, index):
        query, box_loc, box_feature, class_label, others = self._read_h5_file(index)
        
        query = self.mytokenizer.convert_str_to_ids(query)
        box_pos, box_feature, box_label = self._process_box(
            box_loc, box_feature, class_label, others, self.label_to_token_id)
        box_pos, box_feature, box_label = list(map(torch.tensor, (box_pos, box_feature, box_label)))
        return query.long(), box_pos.float(), box_feature.float(), box_label.float(), torch.tensor([1]).float()
    
    def __len__(self):
        return self.size
    
    def _process_box(self, box_loc, box_feature, class_label, others, label_to_token_id):
        num_boxes = others[3]
        h, w = others[1], others[2]

        box_loc = box_loc[:self.cfg['max_box_num'], :]
        box_feature = box_feature[:self.cfg['max_box_num'], :]
        class_label = class_label[:self.cfg['max_box_num']]
        
        if not self.processed:
            y1, x1, y2, x2 = box_loc[:, 0], box_loc[:, 1], box_loc[:, 2], box_loc[:, 3]

            box_pos = [x1 / w,
                       y1 / h, 
                       x2 / w,
                       y2 / h,
                       (x2 - x1) * (y2 - y1) / (w * h)]
            box_pos = np.stack(box_pos, axis=1)
        else:
            box_pos = box_loc

        box_label = [np.array(label_to_token_id[label]) for label in class_label]
        box_label = np.stack(box_label, axis=0)


        return (
            box_pos, 
            box_feature,
            box_label)
    
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
    
    def _convert_label_to_token_id(self, class_num_to_label, mytokenizer):
        #lens = []
        label_to_token_id = {}
        for k, v in class_num_to_label.items():
            #lens.append(len(tokenizer.tokenizer.tokenize(v)))
            label_to_token_id[k] = mytokenizer.convert_str_to_ids(v, 
                                                                tensor=False,
                                                                max_len=self.cfg['max_class_word_num'])
        #print(np.percentile(lens, [0, 25, 50, 75, 95, 99, 100]))
        return label_to_token_id
    
    @staticmethod
    def Collate_fn(samples):
        transposed_samples = list(zip(*samples))
        cat_samples = [torch.stack(transposed_sample) for transposed_sample in transposed_samples]
        return cat_samples
    
class NSDataset(BasicDataset):
    
    def __init__(self,
                 h5_path,
                 mytokenizer,
                 cfg,
                 label_info_path='../data/Kdd/multimodal_labels.txt',
                 processed=False, 
                 neg_sampling=True, 
                 save_RAM=False, 
                 *args, 
                 **kwargs):
        super(NSDataset, self).__init__(h5_path=h5_path,
                                           mytokenizer=mytokenizer,
                                           label_info_path=label_info_path,
                                           cfg=cfg,
                                           processed=False,
                                           *args, **kwargs)
        self.neg_sampling = neg_sampling
        self.save_RAM = save_RAM
        if self.save_RAM:
            pass
        else:
            with h5py.File(self.h5_path, 'r', libver='latest') as h5file:
                otherss = h5file.get('others/data')

                self.product_ids = np.array(otherss[:, 0], dtype=np.int32)
                self.query_ids = np.array(otherss[:, 4], dtype=np.int32)
                
    def _read_ids(self, index):
        if self.save_RAM:
            with h5py.File(self.h5_path, 'r', libver='latest') as h5file:
                otherss = h5file.get('others/data')
                others = otherss[index]
            neg_product_id, neg_query_id = others[0], others[4]
        else:
            neg_product_id, neg_query_id = self.product_ids[index], self.query_ids[index]
        return neg_product_id, neg_query_id
    
    def __getitem__(self, index):
      
        # positive
        query, box_loc, box_feature, class_label, others = self._read_h5_file(index)
        query = self.mytokenizer.convert_str_to_ids(query)
        pos_box_pos, pos_box_feature, pos_box_label = self._process_box(
            box_loc, box_feature, class_label, others, self.label_to_token_id)
        pos_box_pos, pos_box_feature, pos_box_label = list(map(torch.tensor, (pos_box_pos, pos_box_feature, pos_box_label)))
        
        pos_product_id, pos_query_id = self._read_ids(index)
        
        if not self.neg_sampling:
            return (
            (query.long(), pos_box_pos.float(), pos_box_feature.float(), pos_box_label.float(), torch.tensor(1))
            )
        else:
            # negative
            neg_index = np.random.choice(self.size, 1).item()
            neg_product_id, neg_query_id = self._read_ids(neg_index)

            while pos_query_id == neg_query_id:
                neg_index = np.random.choice(self.size, 1).item()
                neg_product_id, neg_query_id = self._read_ids(neg_index)
            
            _, box_loc, box_feature, class_label, others = self._read_h5_file(neg_index)
            neg_box_pos, neg_box_feature, neg_box_label = self._process_box(
                box_loc, box_feature, class_label, others, self.label_to_token_id)
            neg_box_pos, neg_box_feature, neg_box_label = list(map(torch.tensor, (neg_box_pos, neg_box_feature, neg_box_label)))
            return (
                (query.long(), pos_box_pos.float(), pos_box_feature.float(), pos_box_label.float(), torch.tensor([1]).float()),
                (query.long(), neg_box_pos.float(), neg_box_feature.float(), neg_box_label.float(), torch.tensor([0]).float())
            )
        
    @staticmethod
    def Collate_fn(samples): # Collate_fn的输入是 [self.dataset[i] for i in indices]，即此batch的数据(data,target)的list
        samples = list(zip(*samples))
        samples = list(samples[0]) + list(samples[1])
        transposed_samples = list(zip(*samples))
        cat_samples = [torch.stack(transposed_sample) for transposed_sample in transposed_samples]
        return cat_samples


class MNSDataset(NSDataset):

    def __init__(self,
                 h5_path,
                 mytokenizer,
                 cfg,
                 label_info_path='../data/Kdd/multimodal_labels.txt',
                 processed=False,
                 multi_neg_sampling=True,
                 save_RAM=False, 
                 neg_k=5,
                 *args,
                 **kwargs):
        super(MNSDataset, self).__init__(h5_path=h5_path,
                                        mytokenizer=mytokenizer,
                                        label_info_path=label_info_path,
                                        cfg=cfg,
                                        processed=False,
                                        neg_sampling=True,
                                        save_RAM=save_RAM,
                                        *args, **kwargs)
        self.multi_neg_sampling = multi_neg_sampling
        self.neg_k = neg_k
        

    def __getitem__(self, index):

        # positive
        p_query, box_loc, box_feature, class_label, others = self._read_h5_file(index)
        pos_query = self.mytokenizer.convert_str_to_ids(p_query)
        pos_box_pos, pos_box_feature, pos_box_label = self._process_box(
            box_loc, box_feature, class_label, others, self.label_to_token_id)
        pos_box_pos, pos_box_feature, pos_box_label = list(
            map(torch.tensor, (pos_box_pos, pos_box_feature, pos_box_label)))

        pos_product_id, pos_query_id = others[0], others[4]

        if not self.multi_neg_sampling:
            return (
                (pos_query.long(), pos_box_pos.float(), pos_box_feature.float(), pos_box_label.float(), torch.tensor(1))
            )
        else:
            # negative
            samples = [(pos_query.long(), pos_box_pos.float(), pos_box_feature.float(), pos_box_label.float(),
                 torch.tensor([1]).float())]
            multi_neg_index = np.random.choice(self.size, self.neg_k, replace=False)
            for neg_index in multi_neg_index:
                _, box_loc, box_feature, class_label, others = self._read_h5_file(neg_index)
                neg_product_id, neg_query_id = others[0], others[4]
                
                while pos_query_id == neg_query_id:
                    neg_index = np.random.choice(self.size, 1).item()
                    _, box_loc, box_feature, class_label, others = self._read_h5_file(neg_index)
                    neg_product_id, neg_query_id = others[0], others[4]

                
                neg_box_pos, neg_box_feature, neg_box_label = self._process_box(
                    box_loc, box_feature, class_label, others, self.label_to_token_id)
                neg_box_pos, neg_box_feature, neg_box_label = list(map(torch.tensor, (neg_box_pos, neg_box_feature, neg_box_label)))
                samples.append((pos_query.long(), neg_box_pos.float(), neg_box_feature.float(), neg_box_label.float(),
                 torch.tensor([0]).float()))
            return (tuple(samples))

    @staticmethod
    def Collate_fn(samples):  # Collate_fn的输入是 [self.dataset[i] for i in indices]，即此batch的数据(data,target)的list
        sample_num = len(samples[0])
        samples = list(zip(*samples))
        sam = []
        for i in range(sample_num):
            sam += list(samples[i])
        transposed_samples = list(zip(*samples))
        cat_samples = [torch.stack(transposed_sample) for transposed_sample in transposed_samples]
        return cat_samples

class TestDataset(BasicDataset):
    """
    Only to get ids
    """
    
    def __init__(self,
                 h5_path,
                 mytokenizer,
                 cfg,
                 label_info_path='../data/Kdd/multimodal_labels.txt',
                 processed=False,
                 *args, 
                 **kwargs):
        super(TestDataset, self).__init__(h5_path=h5_path,
                                           mytokenizer=mytokenizer,
                                           label_info_path=label_info_path,
                                           cfg=cfg,
                                           processed=processed,
                                           *args, **kwargs)
        self.data_others = self._read_h5_file(slice(self.size))
        
    def _read_h5_file(self, index):
        with h5py.File(self.h5_path, 'r', libver='latest') as h5file:

            otherss = h5file.get('others/data')
            
            others = otherss[index]
        return others
    
    def __getitem__(self, index):

        productID = self.data_others[0]
        queryID = self.data_others[4]
        return productID, queryID
    

class nDCGat5_Calculator:
    def __init__(self, h5path, tokenizer, cfg, k=5, processed=False, val_ans_path='../data/Kdd/valid_answer.json'):
        self.test_dataset = TestDataset(h5path, tokenizer, cfg=cfg, processed=processed)
        self.k = k
        self.val_ans_path = val_ans_path
        
    def __call__(self, true, pred):
        return self.nDCGat5(self.test_dataset, pred, self.k)
    
    def nDCGat5(self, val_dataset, preds, k=5):
        dataset_sizes = len(val_dataset)
        others = val_dataset._read_h5_file(slice(dataset_sizes)) # list的Index作为参数时，需要切片slice作为参数传入
        queryID = others[:, 4]
        productID = others[:, 0]


        pred_dict = defaultdict(list)

        for i in range(dataset_sizes):
            pred_dict[queryID[i]].append((productID[i], preds[i]))
        query_id,product1,product2,product3,product4,product5 = [],[],[],[],[],[]
        for key in pred_dict.keys():
            rlist = pred_dict[key]
            rlist.sort(key=lambda x: x[1], reverse=True) # 降序
            query_id.append(key)
            product1.append(rlist[0][0])
            product2.append(rlist[1][0])
            product3.append(rlist[2][0])
            product4.append(rlist[3][0])
            product5.append(rlist[4][0])
        sub = pd.DataFrame({'query-id':query_id,
                            'product1':product1,
                            'product2':product2,
                            'product3':product3,
                            'product4':product4,
                            'product5':product5,
                            })
        sub.to_csv('result/val_submission.csv', index=False) # 不保存行索引

        reference = json.load(open(self.val_ans_path))
        # read predictions
        predictions = read_submission('result/val_submission.csv', reference, k)

        # compute score for each query
        score_sum = 0.
        for qid in reference.keys():
            ground_truth_ids = set([str(pid) for pid in reference[qid]])
            ref_vec = [1.0] * len(ground_truth_ids)
            pred_vec = [1.0 if pid in ground_truth_ids else 0.0 for pid in predictions[qid]]
            score_sum += get_ndcg(pred_vec, ref_vec, k)
        # the higher score, the better
        score = score_sum / len(reference)

        return score
