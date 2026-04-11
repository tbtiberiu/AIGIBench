import numpy as np
import torch
from data import create_dataloader
from sklearn.metrics import accuracy_score, average_precision_score


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for path, img, text, input_ids, attention_mask, label in data_loader:
            y_pred.extend(
                model(img.cuda(), None, None, cla=True).sigmoid().flatten().tolist()
            )
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred
