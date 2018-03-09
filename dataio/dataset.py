import h5py
import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot as plt

class Dataset(object):
    def __init__(self, config):
        self.config = config

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
                np.random.shuffle(self._index)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        batch_index = self._index[start:end]
        batch_index = list(np.sort(batch_index))
        label = self._dataset_label[batch_index]
        img = self._dataset_img[batch_index]
        attrKey = self._dataset_attrKey[batch_index]

        return img, label, attrKey

    def shuffle(self):
        np.random.shuffle(self._index)
        return


if __name__ == '__main__':
    filename = '/home/remote/Data/fashionAI/train.hdf5'
    dataset = Dataset(1)
    dataset.load_h5py(filename)
    img, label, attrKey = dataset.next_batch(1)
    img = img[0,:,:,:]
    plt.imshow(img)
    plt.show()
