import tensorflow as tf
import csv
import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import socket
import numpy as np
import tqdm
from PIL import Image
import h5py

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from config.config_util import Parser
from dataio.ontology import task2index

hostName = socket.gethostname()
if hostName == 'ubuntu':
    csvfilename = '/mnt/hgfs/FashionAI/web/Annotations/skirt_length_labels.csv'
    raw_train_folder = '/mnt/hgfs/FashionAI/web/'
    datafolder = '/home/leo/Data/fashionAI/'
else:
    csvfilename = '/home/remote/Data/fashionAI/train_data/Annotations/train.csv'
    raw_train_folder = '/home/remote/Data/fashionAI/train_data/'
    datafolder = '/home/remote/Data/fashionAI/'

def string2label(string, nlabel):
    label = [0] * nlabel
    for i, c in enumerate(string):
        if c == 'y':
            label[i] = 1
    return label

def main():
    dataset_name_train = 'train'
    dataset_name_valid = 'valid'
    with open(csvfilename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        ndata = len(list(spamreader))

    f_train = h5py.File(os.path.join(datafolder, dataset_name_train + '.hdf5'), 'w')
    f_valid = h5py.File(os.path.join(datafolder, dataset_name_valid + '.hdf5'), 'w')
    config_file = os.path.join(root_dir, 'config/skirt_length.cfg')
    config = Parser(config_file)
    batch_shape = (config.batch_size, config.image_height, config.image_width, 3)
    data_shape = (ndata, config.image_height, config.image_width, 3)
    dataset_img_train = f_train.create_dataset('img', data_shape,
                                               chunks=batch_shape,
                                               dtype='f')
    dataset_img_valid = f_valid.create_dataset('img', data_shape,
                                               chunks=batch_shape,
                                               dtype='f')

    batch_shape = (config.batch_size,)
    data_shape = (ndata,)
    dataset_attrKey_train = f_train.create_dataset('attrKey', data_shape,
                                       chunks=batch_shape, dtype='i')
    dataset_attrKey_valid = f_valid.create_dataset('attrKey', data_shape,
                                       chunks=batch_shape, dtype='i')

    batch_shape = (config.batch_size, config.max_label)
    data_shape = (ndata, config.max_label)
    dataset_label_train = f_train.create_dataset('label', data_shape, chunks=batch_shape,
                                     dtype='i')
    dataset_label_valid = f_valid.create_dataset('label', data_shape, chunks=batch_shape,
                                     dtype='i')

    with open(csvfilename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        num_train = 0
        num_valid = 0
        for row in tqdm.tqdm(spamreader):
            img = row[0]
            attrKey = row[1]
            label = row[2]
            img_path = os.path.join(raw_train_folder, img)
            image = Image.open(img_path)
            image = image.resize((config.image_height, config.image_width))
            image = np.array(image)
            attrKey = task2index[attrKey]
            label = string2label(label, config.max_label)
            p = np.random.rand(1)
            if p < 0.95:
                dataset_img_train[num_train, :, :, :] = image
                dataset_attrKey_train[num_train] = attrKey
                dataset_label_train[num_train, :] = label
                num_train += 1
            else:
                dataset_img_valid[num_valid, :, :, :] = image
                dataset_attrKey_valid[num_valid] = attrKey
                dataset_label_valid[num_valid, :] = label
                num_valid += 1
        f_train.close()
        f_valid.close()
        print('{} samples in trainset, \n{} samples in validset'.format(num_train, num_valid))


if __name__ == '__main__':
    main()
