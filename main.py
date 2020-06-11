from construct_h5_processed import create_h5_processed
from construct_h5_all_processed_label import create_h5_all_processed
import os
try:
    os.mkdir('../user_data/tmp_data')
except FileExistsError:
    pass

create_h5_processed('../data/valid/valid.tsv', '../user_data/tmp_data/valid_processed.h5')
create_h5_processed('../data/testA/testA.tsv', '../user_data/tmp_data/testA_processed.h5')
create_h5_processed('../data/testB/testA.tsv', '../user_data/tmp_data/testB_processed.h5')
create_h5_processed('../data/train/train.tsv', '../user_data/tmp_data/train_processed.h5')

create_h5_all_processed('../user_data/tmp_data/train_processed.h5', '../user_data/tmp_data/train_all_processed_label.h5', '../data/train/train.tsv')
create_h5_all_processed('../user_data/tmp_data/valid_processed.h5', '../user_data/tmp_data/valid_all_processed_label.h5', '../data/valid/valid.tsv')
create_h5_all_processed('../user_data/tmp_data/testA_processed.h5', '../user_data/tmp_data/testA_all_processed_label.h5', '../data/valid/testA.tsv')
create_h5_all_processed('../user_data/tmp_data/testB_processed.h5', '../user_data/tmp_data/testB_all_processed_label.h5', '../data/valid/testB.tsv')

import Label_pass_bert

from Basic_model_with_label import *

def predict_on_test(dataloader):
    scores = {}
    preds_list = []
    for i in range(21):
        print(f'\rThe {i} epoch', end='')
        checkpoint = torch.load(f'models/label-model{i}.pt')
        try:
            model.load_state_dict(checkpoint['model'])
        except RuntimeError:
            print(f'wrong model file for epoch {i}')
            continue
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        device = 'cuda:0'
        epoch_pred = []
        epoch_true = []
        for query, box_pos, box_feature, box_label, labels in tqdm(dataloader):
            batch_size = labels.shape[0]

            with torch.no_grad():
                query, box_pos, box_feature, box_label, labels = (
                    query.to(device), box_pos.to(device), box_feature.to(device), box_label.to(device),
                    labels.to(device).float()
                )
                outputs = model(query, box_pos, box_feature, box_label)
                preds = outputs
                loss = criterion(outputs, labels)

            epoch_pred.append(preds.detach().cpu())
            epoch_true.append(labels.cpu())

        epoch_pred = torch.cat(epoch_pred, 0)
        epoch_true = torch.cat(epoch_true, 0)
        preds_list.append(epoch_pred.numpy().copy())
        scores[i] = metrics['nDCG@5'](epoch_true, epoch_pred)
    return scores, preds

test_ds = BasicAllDataset('../user_data/tmp_data/testB_processed_label.h5', single_thread=True)
test_dl = data.DataLoader(test_ds, shuffle=False, collate_fn=val_ds.Collate_fn, **val_cfg)
test_ndcg5 = nDCGat5_Calculator('../user_data/tmp_data/testB_processed_label.h5')
scores, _ = predict_on_test(valid_dl)

_, preds = predict_on_test(test_dl)
final_pred = np.average(preds, axis=0, weights=list(scores.values())[9:17])
test_ndcg5.save_submission(final_pred, '../prediction_result/submission.csv')



