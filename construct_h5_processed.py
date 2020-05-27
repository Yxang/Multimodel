import numpy as np
import pandas as pd
import h5py
import base64

def create_h5_processed(tsv, target, MAX_NUM_BOXES=128, WRITE_CHUNK=1024):
    with h5py.File(target, 'w', libver='latest')as hf:

        hf.create_group('box_loc')
        hf.create_group('box_feat')
        hf.create_group('class_labels')
        hf.create_group('querys')
        hf.create_group('others')
        box_loc_h5ds = hf.create_dataset('box_loc/data', 
                                         shape=(WRITE_CHUNK, MAX_NUM_BOXES, 5), 
                                         chunks=(1, MAX_NUM_BOXES, 5), 
                                         maxshape=(None, MAX_NUM_BOXES, 5), 
                                         #compression="lzf",
                                         dtype='f')
        box_feat_h5ds = hf.create_dataset('box_feat/data', 
                                          shape=(WRITE_CHUNK, MAX_NUM_BOXES, 2048), 
                                          chunks=(1, MAX_NUM_BOXES, 2048), 
                                          maxshape=(None, MAX_NUM_BOXES, 2048), 
                                          compression="lzf",
                                          dtype='f')
        class_labels_h5ds = hf.create_dataset('class_labels/data', 
                                              shape=(WRITE_CHUNK, MAX_NUM_BOXES), 
                                              chunks=(1, MAX_NUM_BOXES), 
                                              maxshape=(None, MAX_NUM_BOXES), 
                                              #compression="lzf",
                                              dtype='i')
        querys_h5ds = hf.create_dataset('querys/data', 
                                         shape=(WRITE_CHUNK,), 
                                         chunks=(1,), 
                                         maxshape=(None,), 
                                         #compression="lzf",
                                         dtype=h5py.string_dtype())
        others_h5ds = hf.create_dataset('others/data', 
                                         shape=(WRITE_CHUNK, 5), 
                                         chunks=(1, 5), 
                                         maxshape=(None, 5), 
                                         #compression="lzf",
                                         dtype='i')

        def process_line(line):
            l = line.strip().split('\t')
            num_boxes = int(l[3])
            if num_boxes > MAX_NUM_BOXES:
                raise RuntimeError(f'num_boxes is large than {MAX_NUM_BOXES}, which is {num_boxes}')
            box_loc = np.frombuffer(base64.b64decode(l[4]), dtype=np.float32).reshape(num_boxes, 4)
            box_feat = np.frombuffer(base64.b64decode(l[5]), dtype=np.float32).reshape(num_boxes, 2048)
            class_label = np.frombuffer(base64.b64decode(l[6]), dtype=np.int64).reshape(num_boxes,)
            query = l[7]
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
            
            return num_boxes, box_pos, box_feat, class_label, query, other

        def flush_into_ds(hf, i, box_locs, box_feats, class_labels, querys, others):
            box_loc_h5ds = hf.get('box_loc/data')
            box_loc_h5ds.resize(i + 1, axis=0)
            box_loc_h5ds[i // WRITE_CHUNK * WRITE_CHUNK:i+1, :, :] = box_locs[:i%WRITE_CHUNK + 1, :, :]

            box_feat_h5ds = hf.get('box_feat/data')
            box_feat_h5ds.resize(i + 1, axis=0)
            box_feat_h5ds[i // WRITE_CHUNK * WRITE_CHUNK:i+1, :, :] = box_feats[:i%WRITE_CHUNK + 1, :, :]

            class_labels_h5ds = hf.get('class_labels/data')
            class_labels_h5ds.resize(i + 1, axis=0)
            class_labels_h5ds[i // WRITE_CHUNK * WRITE_CHUNK:i+1, :] = class_labels[:i%WRITE_CHUNK + 1, :]

            querys_h5ds = hf.get('querys/data')
            querys_h5ds.resize(i + 1, axis=0)
            querys_h5ds[i // WRITE_CHUNK * WRITE_CHUNK:i+1] = querys[:i%WRITE_CHUNK + 1]

            others_h5ds = hf.get('others/data')
            others_h5ds.resize(i + 1, axis=0)
            others_h5ds[i // WRITE_CHUNK * WRITE_CHUNK:i+1, :] = others[:i%WRITE_CHUNK + 1, :]
            
            return (
                np.zeros((WRITE_CHUNK, MAX_NUM_BOXES, 5)), 
                np.zeros((WRITE_CHUNK, MAX_NUM_BOXES, 2048)), 
                np.zeros((WRITE_CHUNK, MAX_NUM_BOXES)),
                ['' for i in range(WRITE_CHUNK)],
                np.zeros((WRITE_CHUNK, 5))
            )


        with open(tsv, 'r') as f:
            l = f.readline()
            l = f.readline()
            i = 0

            box_locs, box_feats, class_labels, querys, others = (
                np.zeros((WRITE_CHUNK, MAX_NUM_BOXES, 5)), 
                np.zeros((WRITE_CHUNK, MAX_NUM_BOXES, 2048)), 
                np.zeros((WRITE_CHUNK, MAX_NUM_BOXES)),
                ['' for i in range(WRITE_CHUNK)],
                np.zeros((WRITE_CHUNK, 5))
            )

            while l:
                num_boxes, box_loc, box_feat, class_label, query, other = process_line(l)
                box_locs[i%WRITE_CHUNK, :num_boxes, :] = box_loc
                box_feats[i%WRITE_CHUNK, :num_boxes, :] = box_feat
                class_labels[i%WRITE_CHUNK, :num_boxes] = class_label
                querys[i%WRITE_CHUNK] = query
                others[i%WRITE_CHUNK, :] = other
                if i % WRITE_CHUNK == WRITE_CHUNK - 1:
                    print('\rline {}'.format(i), end='')
                    box_locs, box_feats, class_labels, querys, others = flush_into_ds(hf, i, box_locs, box_feats, class_labels, querys, others)
                i += 1
                l = f.readline()
            if i % WRITE_CHUNK != WRITE_CHUNK - 1:
                flush_into_ds(hf, i - 1, box_locs, box_feats, class_labels, querys, others)
                print('\rline {}'.format(i), end='')
        print()
    return

create_h5_processed('../data/Kdd/train.sample.tsv', '../data/Kdd/train.sample_processed.h5')
create_h5_processed('../data/Kdd/valid.tsv', '../data/Kdd/valid_processed.h5')
create_h5_processed('../data/Kdd/train.tsv', '../data/Kdd/train_processed.h5')

