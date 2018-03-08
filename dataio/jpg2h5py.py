import tensorflow as tf

path = '/mnt/hgfs/FashionAI/rank/Images/skirt_length_labels/'

with tf.gfile.Open(path) as f:
    lines = [line.strip() for line in f]
    for line in lines:
        print(line)
