import tensorflow as tf
import csv
import os,sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import socket
import numpy as np
import cv2
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
    raw_train_folder = '/home/remote/Data/fashionAI/train_data/Images/'
    datafolder = '/home/remote/Data/fashionAI/'

def string2label(string, nlabel):
    label = [0] * nlabel
    for i, c in enumerate(string):
        if c == 'y':
            label[i] = 1
    return label

def main():
    dataset_name = 'train'
    with open(csvfilename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        ndata = len(list(spamreader))

    f = h5py.File(os.path.join(datafolder, dataset_name + '.hdf5'), w)
    config = Parser(os.path.join(root_dir, 'config/skirt_length.cfg'))
    batch_shape = [config.batch_size, config.image_height, config.image_width, 3]
    data_shape = [ndata, config.image_height, config.image_width, 3]
    dataset_img = f.create_dataset('img', data_shape, chunks=batch_shape,
                                   dtype='f')

    batch_shape = [config.batch_size]
    data_shape = [ndata]
    dataset_attrKey = f.create_dataset('attrKey', data_shape,
                                       chunks=batch_shape, dtype='i')

    batch_shape = [config.batch_size, config.max_label]
    data_shape = [ndata, config.max_label]
    dataset_label = f.create_dataset('label', data_shape, chunks=batch_shape,
                                     dtype='i')

    with open(csvfilename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        num = 0
        for row in spamreader:
            img = row[0]
            attrKey = row[1]
            label = row[2]
            img_path = os.path.join(raw_train_folder, img)
            image = mpimg.imread(img_path)
            image = cv2.resize(image, (config.image_height, config.image_width))
            attrKey = task2index[attrKey]
            label = string2label(label, config.max_label)
            dataset_img[num, :, :, :] = image
            dataset_attrKey[num] = attrKey
            dataset_label[num, :] = label
            num += 1
        print(num)
        f.close()
