#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import h5py
import torch
import torch.distributed as dist
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from apex import amp

import argparse
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel

from tqdm import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score

from utils import MyTokenizer, MyAlbert, BasicAlbertDataset, MNSAlbertDataset, nDCGat5_Calculator

import copy

np.random.seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2020)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

if __name__ == '__main__':

    cfg = {}
    cfg['train_fraction'] = 1
    cfg['lr'] = 3e-4
    cfg['albert_lr'] = 3e-7
    cfg['max_query_word'] = 9
    cfg['max_box_num'] = 9
    cfg['bert_model_name'] = 'albert-large-v2'
    cfg['emb_file'] = 'label_emb_albert-large-v2.npy'
    cfg['max_class_word_num'] = 11
    cfg['dataloader_cfg'] = {
        'batch_size': 32,
        'num_workers': 14,
        'pin_memory': True}
    cfg['epochs'] = 20
    cfg['apex_opt_level'] = 'O2'
    cfg['save_name'] = 'albert-v1'
    cfg['num_negative_sampling'] = 5
    cfg['save_RAM'] = True

    basic_model_cfg = {}
    basic_model_cfg['pos_emb_size'] = 8
    basic_model_cfg['bilstm_hidden_size'] = 1024
    basic_model_cfg['label_emb_fc_1_out'] = 0
    basic_model_cfg['clf1_out'] = 512
    basic_model_cfg['clf2_out'] = 128

    class ImagePart(nn.Module):
        def __init__(self, cfg, model_cfg):
            super(ImagePart, self).__init__()
            self.cfg = cfg
            self.model_cfg = model_cfg

            with open(cfg['emb_file'], 'rb') as f:
                w_mat = np.load(f)

            self.label_emb = nn.Embedding.from_pretrained(torch.tensor(w_mat))

            self.pos_emb_layer = nn.Linear(5, model_cfg['pos_emb_size'])
            self.img_bilstm = nn.LSTM(2048 + model_cfg['pos_emb_size'] + w_mat.shape[1], model_cfg['bilstm_hidden_size'],
                                      batch_first=True, bidirectional=True)

        def forward(self, box_pos, box_feature, box_label):
            batch_size = box_pos.shape[0]

            box_pos = box_pos.view(-1, 5)
            pos_emb = self.pos_emb_layer(box_pos)
            pos_emb = pos_emb.view(batch_size, -1, self.model_cfg['pos_emb_size'])

            label_emb = self.label_emb(box_label)

            image_seq_feat = torch.cat([pos_emb, box_feature, label_emb], dim=2)
            image_lstm, _ = self.img_bilstm(image_seq_feat)
            image_emb = image_lstm.view(batch_size, -1, 2, self.model_cfg['bilstm_hidden_size'])[:, 0, 1, :]
            return image_emb

    class AlbertLabel(nn.Module):
        def __init__(self, bert_name, cfg, model_cfg, **other_bert_kwargs):
            super(AlbertLabel, self).__init__()
            self.cfg = cfg
            self.model_cfg = model_cfg
            self.bert = MyAlbert(bert_name, **other_bert_kwargs)
            self.image = ImagePart(self.cfg, self.model_cfg)

            self.clf1 = nn.Linear(1024 + model_cfg['bilstm_hidden_size'], basic_model_cfg['clf1_out'])
            self.clf2 = nn.Linear(basic_model_cfg['clf1_out'], basic_model_cfg['clf2_out'])
            self.clf3 = nn.Linear(basic_model_cfg['clf2_out'], 1)

            self.dropout = nn.Dropout(0.25)

        def forward(self, query, box_pos, box_feature, box_label):

            query_emb = self.bert(query)
            image_emb = self.image(box_pos, box_feature, box_label)

            embs = torch.cat([query_emb, image_emb], dim=1)

            embs = F.relu(self.clf1(embs))
            embs = F.relu(self.clf2(embs))
            embs = self.dropout(embs)
            embs = self.clf3(embs)

            return embs


    def accuracy_score_prob(y_true, y_pred, threshold=0.5):
        y_pred = y_pred > threshold
        return accuracy_score(y_pred, y_true)

    def train_model(dataloders, model, criterion, optimizer, scheduler=None, metrics=None, device='cuda:0', num_epochs=25):
        best_ndcg = 0.
        dataset_sizes = {
            'train': len(dataloders['train'].dataset),
            'valid': len(dataloders['valid'].dataset)
        }
        model.to(device)

        for epoch in range(num_epochs):
            epoch_train_metrics = {}
            epoch_val_metrics = {}
            for phase in [
                'train',
                'valid',
            ]:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                epoch_pred = []
                epoch_true = []

                for query, box_pos, box_feature, box_label, labels in tqdm(dataloders[phase]):
                    batch_size = labels.shape[0]

                    if phase == 'train':
                        optimizer.zero_grad()

                        query, box_pos, box_feature, box_label, labels = (
                            query.to(device), box_pos.to(device), box_feature.to(device), box_label.to(device), labels.to(device)
                        )

                        outputs = model(query, box_pos, box_feature, box_label)
                        preds = outputs
                        loss = criterion(outputs, labels)
                    elif phase == 'valid':
                        with torch.no_grad():
                            query, box_pos, box_feature, box_label, labels = (
                                query.to(device), box_pos.to(device), box_feature.to(device), box_label.to(device), labels.to(device)
                            )
                            outputs = model(query, box_pos, box_feature, box_label)
                            preds = outputs
                            loss = criterion(outputs, labels)

                    if phase == 'train':
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                    running_loss += loss.data.item() * batch_size
                    epoch_pred.append(preds.detach().cpu())
                    epoch_true.append(labels.cpu())

                epoch_pred = torch.cat(epoch_pred, 0)
                epoch_true = torch.cat(epoch_true, 0)


                if phase == 'train':
                    train_epoch_loss = running_loss / dataset_sizes['train']
                    if metrics:
                        for name in metrics:
                            if name != 'nDCG@5':
                                epoch_train_metrics[name] = metrics[name](epoch_true, epoch_pred)

                else:
                    valid_epoch_loss = running_loss / dataset_sizes['valid']
                    if metrics:
                        for name in metrics:
                            epoch_val_metrics[name] = metrics[name](epoch_true, epoch_pred)

                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':  scheduler.state_dict(),
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, 'model/' + cfg['save_name'] + '{:d}.pt'.format(epoch))

                if phase == 'valid' and epoch_val_metrics['nDCG@5'] > best_ndcg:
                    best_loss = valid_epoch_loss
                    best_metrics = epoch_val_metrics
                    best_epoch = epoch

            print('Epoch [{}/{}]\n'
                  '    train loss: {:.8f}\n'
                  '    metrics: {}\n'
                  '    valid loss: {:.8f}\n'
                  '    metrics: {}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, epoch_train_metrics,
                valid_epoch_loss, epoch_val_metrics))

        print()
        print('Best val loss: {:4f}'.format(best_loss))
        print('Best val metrics: {}'.format(best_metrics))
        print('At epoch: {:d}'.format(best_epoch))

        try:
            del inputs, labels
        except:
            pass
        torch.cuda.empty_cache()
        return model, optimizer, epoch, loss


    tokenizer = MyTokenizer(cfg=cfg)


    ds = MNSAlbertDataset('../data/Kdd/train_processed_albert.h5', neg_k=cfg['num_negative_sampling'], single_thread=False)
    val_ds = BasicAlbertDataset('../data/Kdd/valid_processed_albert.h5', single_thread=True)

    train_size = len(ds)
    train_split = int(train_size * cfg['train_fraction'])
    train_indices = list(range(train_size))
    train_sampler = data.sampler.SubsetRandomSampler(train_indices[:train_split])


    # In[13]:
    val_cfg = cfg['dataloader_cfg'].copy()
    val_cfg['num_workers'] = 0

    train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)

    dl = data.DataLoader(ds, sampler=train_sampler, collate_fn=ds.Collate_fn, **cfg['dataloader_cfg'])
    valid_dl = data.DataLoader(val_ds, sampler=val_sampler, shuffle=False, collate_fn=val_ds.Collate_fn, **val_cfg)
    dataloders = {'train': dl,
                  'valid': valid_dl}
    metrics = {'acc': accuracy_score_prob,
               'nDCG@5': nDCGat5_Calculator('../data/Kdd/valid_processed_albert.h5')}


    basic_model = AlbertLabel(cfg['bert_model_name'], cfg=cfg, model_cfg=basic_model_cfg)
    model = basic_model


    bert_params = basic_model.bert.parameters()
    other_params = list(set(basic_model.parameters()) - set(bert_params))

    optimizer = AdamW([
        {'params': basic_model.bert.parameters(),'lr': cfg['albert_lr']},
        {'params': other_params, 'lr': cfg['lr']}],
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int((len(ds) // cfg['dataloader_cfg']['batch_size'] + 1) * cfg['epochs'] * 0.2),
        num_training_steps=(len(ds) // cfg['dataloader_cfg']['batch_size'] + 1) * cfg['epochs'])

    criterion = nn.BCEWithLogitsLoss()


    model.cuda()
    convert_syncbn_model(model)
    model, optimizer = amp.initialize(model, optimizer, opt_level=cfg['apex_opt_level'])
    model = DistributedDataParallel(model)


    train_model(dataloders, model, criterion, optimizer, scheduler, metrics=metrics, num_epochs=cfg['epochs'])






