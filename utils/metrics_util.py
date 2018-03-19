import os, sys
import numpy as np

def evaluate_metrics(results):
    logits = results['logits']
    label = results['label']
    pred = np.argmax(logits, axis=1)
    gdth = np.argmax(label, axis=1)
    for p, g in zip(pred, gdth):
        print('pred: {}, gdth: {}'.format(p, g))
    acc = np.sum(pred == gdth) / len(pred)
    return acc

