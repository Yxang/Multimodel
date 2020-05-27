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
from apex import amp

from tqdm.notebook import tqdm

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

from sklearn.metrics import accuracy_score

from utils import MyTokenizer, MyBert, BasicDataset, NSDataset, MNSDataset, nDCGat5_Calculator

import copy


# In[3]:


np.random.seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2020)


# ## Config

# In[4]:


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

basic_model_cfg = {}
basic_model_cfg['pos_emb_size'] = 8
basic_model_cfg['bilstm_hidden_size'] = 768
basic_model_cfg['clf1_out'] = 512
basic_model_cfg['clf2_out'] = 128


# ## Model

# In[5]:


class BasicModel(nn.Module):
    def __init__(self, bert_name, cfg, **other_bert_kwargs):
        super(BasicModel, self).__init__()
        self.cfg = cfg
        self.bert = MyBert(bert_name, **other_bert_kwargs)
        
        self.pos_emb_layer = nn.Linear(5, cfg['pos_emb_size'])
        self.img_bilstm = nn.LSTM(2048 + cfg['pos_emb_size'], cfg['bilstm_hidden_size'], batch_first=True, bidirectional=True)
        
        self.clf1 = nn.Linear(768 + cfg['bilstm_hidden_size'], basic_model_cfg['clf2_out'])
        self.clf2 = nn.Linear(basic_model_cfg['clf2_out'], 1)
        #self.clf1 = nn.Linear(768 + cfg['bilstm_hidden_size'], basic_model_cfg['clf1_out'])
        #self.clf2 = nn.Linear(basic_model_cfg['clf1_out'], basic_model_cfg['clf2_out'])
        #self.clf3 = nn.Linear(basic_model_cfg['clf2_out'], 1)
        
    def forward(self, query, box_pos, box_feature, box_label):
        
        batch_size = query.shape[0]
        
        query_emb = self.bert(query)
        
        box_pos = box_pos.view(-1, 5)
        pos_emb = self.pos_emb_layer(box_pos)
        pos_emb = pos_emb.view(batch_size, -1, self.cfg['pos_emb_size'])
        
        image_seq_feat = torch.cat([pos_emb, box_feature], dim=2)
        image_lstm, _ = self.img_bilstm(image_seq_feat)
        image_emb = image_lstm.view(batch_size, -1, 2, self.cfg['bilstm_hidden_size'])[:, 0, 1, :]
        
        embs = torch.cat([query_emb, image_emb], dim=1)
        
        #embs = F.relu(self.clf1(embs))
        #embs = F.relu(self.clf2(embs))
        #embs = self.clf3(embs)
        
        embs = F.relu(self.clf1(embs))
        embs = self.clf2(embs)
        
        return embs


# In[6]:


def accuracy_score_prob(y_true, y_pred, threshold=0.5):
    y_pred = y_pred > threshold
    return accuracy_score(y_pred, y_true)


# In[7]:


def train_model(dataloders, model, criterion, optimizer, metrics=None, device='cuda:0', num_epochs=25, save_gpu_ram=False):
    best_loss = np.inf
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
                    optimizer.step()

                running_loss += loss.data.item() * batch_size
                epoch_pred.append(preds.detach().cpu())
                epoch_true.append(labels.cpu())
            
            epoch_pred = torch.cat(epoch_pred, 0)
            epoch_true = torch.cat(epoch_true, 0)
            
            if save_gpu_ram:
                del inputs, labels
                torch.cuda.empty_cache()
            
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
                'amp': amp.state_dict()
            }
            if epoch in [0, 1, 3, 7, 11, 15, 19]:
                torch.save(checkpoint, 'model/' + cfg['save_name'] + '{:d}.pt'.format(epoch))

            if phase == 'valid' and valid_epoch_loss < best_loss:
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


# In[9]:


tokenizer = MyTokenizer(cfg=cfg)


# In[11]:


ds = MNSDataset('../data/Kdd/train_processed.h5', tokenizer, cfg, neg_k=cfg['num_negative_sampling'], processed=True, save_RAM=cfg['save_RAM'])
val_ds = BasicDataset('../data/Kdd/valid_processed.h5', tokenizer, cfg, processed=True)


# In[12]:


train_size = len(ds)
train_split = int(train_size * cfg['train_fraction'])
train_indices = list(range(train_size))
train_sampler = data.sampler.SubsetRandomSampler(train_indices[:train_split])


# In[13]:


dl = data.DataLoader(ds, sampler=train_sampler, collate_fn=NSDataset.Collate_fn, **cfg['dataloader_cfg'])
valid_dl = data.DataLoader(val_ds, shuffle=False, collate_fn=BasicDataset.Collate_fn, **cfg['dataloader_cfg'])
dataloders = {'train': dl,
              'valid': valid_dl}
metrics = {'acc': accuracy_score_prob, 
           'nDCG@5': nDCGat5_Calculator('../data/Kdd/valid.h5', tokenizer, cfg)}


# In[ ]:


basic_model = BasicModel(cfg['bert_model_name'], cfg=basic_model_cfg)
model = basic_model


# In[ ]:


bert_params = basic_model.bert.parameters()
other_params = list(set(basic_model.parameters()) - set(bert_params))

#optimizer = BertAdam([
#    {'params': basic_model.bert.parameters() ,'lr': 3e-6, 'warmup': 0.4},
#    {'params': other_params, 'lr': 3e-4, 'warmup': 0.1}],
#    t_total=(len(ds) // cfg['dataloader_cfg']['batch_size'] + 1) * cfg['epochs']
#)
optimizer = optim.Adam([
    {'params': basic_model.bert.parameters() ,'lr': 3e-6,},
    {'params': other_params, 'lr': 3e-4, }],
)
criterion = nn.BCEWithLogitsLoss()


# In[ ]:


model.cuda()
model, optimizer = amp.initialize(model, optimizer, opt_level=cfg['apex_opt_level'])


# In[ ]:


train_model(dataloders, model, criterion, optimizer, metrics=metrics, num_epochs=cfg['epochs'])


# In[ ]:


print(torch.cuda.memory_summary())


# In[ ]:


accuracy_score_prob(true, pred)


# In[ ]:




