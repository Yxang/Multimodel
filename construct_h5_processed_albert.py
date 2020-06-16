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
cfg['bert_model_name'] = 'albert-large-v2'
cfg['max_class_word_num'] = 11
cfg['dataloader_cfg'] = {
    'batch_size': 50,
    'num_workers': 8,
    'pin_memory': False}
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

H5_CHUNK = 500000

def chunkof(index):
    return index % H5_CHUNK, index // H5_CHUNK

def create_ds(hf, chunk):
    WRITE_CHUNK = cfg['dataloader_cfg']['batch_size']
    querys_h5ds = hf.create_dataset(
        'querys/data' + f'_{chunk}',
        shape=(WRITE_CHUNK, cfg['max_query_word']),
        chunks=(1, cfg['max_query_word']),
        maxshape=(None, cfg['max_query_word']),
        # compression="lzf",
        dtype='i'
    )
    box_poss_h5ds = hf.create_dataset(
        'box_poss/data' + f'_{chunk}',
        shape=(WRITE_CHUNK, cfg['max_box_num'], 5),
        chunks=(1, cfg['max_box_num'], 5),
        maxshape=(None, cfg['max_box_num'], 5),
        # compression="lzf",
        dtype='f'
    )
    box_features_h5ds = hf.create_dataset(
        'box_feature/data' + f'_{chunk}',
        shape=(WRITE_CHUNK, cfg['max_box_num'], 2048),
        chunks=(1, cfg['max_box_num'], 2048),
        maxshape=(None, cfg['max_box_num'], 2048),
        # compression="lzf",
        dtype='f'
    )
    box_labels_h5ds = hf.create_dataset(
        'box_label/data' + f'_{chunk}',
        shape=(WRITE_CHUNK, cfg['max_box_num']),
        chunks=(1, cfg['max_box_num']),
        maxshape=(None, cfg['max_box_num']),
        # compression="lzf",
        dtype='i'
    )
    others_h5ds = hf.create_dataset(
        'others/data' + f'_{chunk}',
        shape=(WRITE_CHUNK, 5),
        chunks=(1, 5),
        maxshape=(None, 5),
        # compression="lzf",
        dtype='i'
    )
    return querys_h5ds, box_poss_h5ds, box_features_h5ds, box_labels_h5ds, others_h5ds

def create_h5_all_processed(
        source_h5, target, tsv,
        MAX_NUM_BOXES=cfg['max_box_num'],
        WRITE_CHUNK=cfg['dataloader_cfg']['batch_size']
):
    ds = BasicDataset(source_h5, tokenizer, cfg)
    dl = data.DataLoader(ds, shuffle=False, collate_fn=BasicDataset.Collate_fn, **cfg['dataloader_cfg'])

    with h5py.File(target, 'w', libver='latest')as hf:

        create_ds(hf, 0)
        created_chunk = {0}

        def flush_into_ds(hf, i, query, box_pos, box_feature):
            i, chunk = chunkof(i - 1)
            i += 1
            if chunk not in created_chunk:
                create_ds(hf, chunk)
                created_chunk.add(chunk)
            querys_h5ds = hf.get('querys/data' + f'_{chunk}')
            querys_h5ds.resize(i, axis=0)
            querys_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i, :] = query

            box_poss_h5ds = hf.get('box_poss/data' + f'_{chunk}')
            box_poss_h5ds.resize(i, axis=0)
            box_poss_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i, :, :] = box_pos

            box_features_h5ds = hf.get('box_feature/data' + f'_{chunk}')
            box_features_h5ds.resize(i, axis=0)
            box_features_h5ds[(i - 1) // WRITE_CHUNK * WRITE_CHUNK:i, :] = box_feature

            return 0

        i = 0
        for query, box_pos, box_feature, _, _ in tqdm(dl):
            query, box_pos, box_feature = query.numpy(), box_pos.numpy(), box_feature.numpy()
            i += query.shape[0]
            flush_into_ds(hf, i, query, box_pos, box_feature)


        print('reading labels:')
        ##################


        with open(tsv, 'r') as f:
            l = f.readline()
            l = f.readline()
            i = 0

            while l:
                w_class_label = np.zeros((cfg['max_box_num'],))
                l = l.strip().split('\t')
                num_boxes = int(l[3])
                class_label = np.frombuffer(base64.b64decode(l[6]), dtype=np.int64).reshape(num_boxes, ) + 1
                if num_boxes > cfg['max_box_num']:
                    w_class_label[:] = class_label[:cfg['max_box_num']]
                else:
                    w_class_label[:num_boxes] = class_label
                    chunk_i, chunk = chunkof(i)
                box_labels_h5ds = hf.get('box_label/data' + f'_{chunk}')
                box_labels_h5ds.resize(chunk_i + 1, axis=0)
                box_labels_h5ds[chunk_i, :] = w_class_label
                if i % WRITE_CHUNK == WRITE_CHUNK - 1:
                    print('\rline {}'.format(i), end='')
                i += 1
                l = f.readline()
            if i % WRITE_CHUNK != WRITE_CHUNK - 1:
                print('\rline {}'.format(i), end='')
        print()

        ##############

        print('reading others\r', end='')
        with h5py.File(source_h5, 'r', libver='latest') as h5file_source:
            others_h5ds_source = h5file_source.get('others/data')
            len_others = h5file_source.get('others/data').shape[0]
            for i in range(len_others):
                chunk_i, chunk = chunkof(i)
                others_h5ds = hf.get('others/data' + f'_{chunk}')
                others_h5ds.resize(chunk_i + 1, axis=0)
                others_h5ds[chunk_i] = others_h5ds_source[i]
        print('reading others finished!')



    return

create_h5_all_processed('../data/Kdd/train.sample_processed.h5', '../data/Kdd/train.sample_processed_albert.h5', '../data/Kdd/train.sample.tsv')
create_h5_all_processed('../data/Kdd/valid_processed.h5', '../data/Kdd/valid_processed_albert.h5', '../data/Kdd/valid.tsv')
create_h5_all_processed('../data/Kdd/train_processed.h5', '../data/Kdd/train_processed_albert.h5', '../data/Kdd/train.tsv')
