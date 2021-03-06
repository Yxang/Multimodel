import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
import base64

from tqdm import tqdm

from utils import MyTokenizer, BasicDataset

cfg = {}
cfg['train_fraction'] = 0.25
cfg['max_query_word'] = 9
cfg['max_box_num'] = 9
cfg['bert_model_name'] = 'bert-base-uncased'
cfg['max_class_word_num'] = 11
cfg['dataloader_cfg'] = {
    'batch_size': 256,
    'num_workers': 0,
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

tokenizer = MyTokenizer(cfg=cfg)


def create_h5_all_processed(
        source_h5, target, MAX_NUM_BOXES=cfg['max_box_num'],
        WRITE_CHUNK=cfg['dataloader_cfg']['batch_size']
):
    ds = BasicDataset(source_h5, tokenizer, cfg)
    dl = data.DataLoader(ds, shuffle=False, collate_fn=BasicDataset.Collate_fn, **cfg['dataloader_cfg'])

    with h5py.File(target, 'w', libver='latest')as hf:

        hf.create_group('querys')
        hf.create_group('box_poss')
        hf.create_group('box_features')
        hf.create_group('box_labels')

        querys_h5ds = hf.create_dataset(
            'querys/data',
            shape=(WRITE_CHUNK, cfg['max_query_word']),
            chunks=(1, cfg['max_query_word']),
            maxshape=(None, cfg['max_query_word']),
            #compression="lzf",
            dtype='i'
        )
        box_poss_h5ds = hf.create_dataset(
            'box_poss/data',
            shape=(WRITE_CHUNK, cfg['max_box_num'], 5),
            chunks=(1, cfg['max_box_num'], 5),
            maxshape=(None, cfg['max_box_num'], 5),
            #compression="lzf",
            dtype='f'
        )
        box_features_h5ds = hf.create_dataset(
            'box_feature/data',
            shape=(WRITE_CHUNK, cfg['max_box_num'], 2048),
            chunks=(1, cfg['max_box_num'], 2048),
            maxshape=(None, cfg['max_box_num'], 2048),
            #compression="lzf",
            dtype='f'
        )
        box_labels_h5ds = hf.create_dataset(
            'box_label/data',
            shape=(WRITE_CHUNK, cfg['max_box_num'], cfg['max_class_word_num']),
            chunks=(1, cfg['max_box_num'], cfg['max_class_word_num']),
            maxshape=(None, cfg['max_box_num'], cfg['max_class_word_num']),
            #compression="lzf",
            dtype='f'
        )
        others_h5ds = hf.create_dataset(
            'others/data',
            shape=(WRITE_CHUNK, 5),
            chunks=(1, 5),
            maxshape=(None, 5),
            #compression="lzf",
            dtype='i'
        )

        def flush_into_ds(hf, i, query, box_pos, box_feature, box_label):
            querys_h5ds = hf.get('querys/data')
            querys_h5ds.resize(i, axis=0)
            querys_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i, :] = query

            box_poss_h5ds = hf.get('box_poss/data')
            box_poss_h5ds.resize(i, axis=0)
            box_poss_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i, :, :] = box_pos

            box_features_h5ds = hf.get('box_feature/data')
            box_features_h5ds.resize(i, axis=0)
            box_features_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i, :] = box_feature

            box_labels_h5ds = hf.get('box_label/data')
            box_labels_h5ds.resize(i, axis=0)
            box_labels_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i + 1, :, :] = box_label

            return 0

        i = 0
        for query, box_pos, box_feature, box_label, _ in tqdm(dl):
            query, box_pos, box_feature, box_label = query.numpy(), box_pos.numpy(), box_feature.numpy(), box_label.numpy()
            i += query.shape[0]
            flush_into_ds(hf, i, query, box_pos, box_feature, box_label)
        print('reading others\r', end='')
        with h5py.File(source_h5, 'r', libver='latest') as h5file_source:
            others_h5ds_source = h5file_source.get('others/data')
            len_others = h5file_source.get('others/data').shape[0]
            others_h5ds.resize(len_others, axis=0)
            for i in range(len_others):
                others_h5ds[i] = others_h5ds_source[i]
        print('reading others finished!')

    return

create_h5_all_processed('../data/Kdd/train.sample_processed.h5', '../data/Kdd/train.sample_all_processed_me.h5')
create_h5_all_processed('../data/Kdd/train_processed.h5', '../data/Kdd/train_all_processed_me.h5')
create_h5_all_processed('../data/Kdd/valid_processed.h5', '../data/Kdd/valid_all_processed_me.h5')