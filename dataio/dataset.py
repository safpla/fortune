import h5py
import os, sys
import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from config.config_util import Parser


class Dataset(object):
    def __init__(self, config, mode):
        self.config = config
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.load_h5py(config.train_data)
        elif mode == tf.estimator.ModeKeys.EVAL:
            self.load_h5py(config.valid_data)
        else:
            self.load_h5py(config.test_data)

    def load_h5py(self, filename):
        f = h5py.File(filename, "r")
        self._dataset_img = f['img']
        self._dataset_attrKey = f['attrKey']
        self._dataset_label = f['label']
        self._num_examples = self._dataset_label.len()
        print('num of examples: ', self._num_examples)

        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        # batch_size is in the first dimension
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if shuffle:
                self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        batch_index = self._index[start:end]
        batch_index = list(np.sort(batch_index))
        label = self._dataset_label[batch_index]
        label = label[:,0:6]
        img = self._dataset_img[batch_index]
        attrKey = self._dataset_attrKey[batch_index]
        samples = {}
        samples['img'] = img
        samples['label'] = label
        samples['attrKey'] = attrKey
        return samples

    def reset(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def shuffle(self):
        np.random.shuffle(self._index)


if __name__ == '__main__':
    root_path = '/home/leo/GitHub/fashionAI/'
    config_path = os.path.join(root_path, 'config/skirt_length.cfg')
    config = Parser(config_path)
    num_classes = 6
    dataset = Dataset(config, tf.estimator.ModeKeys.TRAIN)
    img, label, attrKey = dataset.next_batch(1)
    print(label)
