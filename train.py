import tensorflow as tf
import numpy as np
import os, sys
import time

print('run into here')
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
from dataio.dataset import Dataset
from model.vgg16_fashionAI import Model
from config.config_util import Parser
from utils.metrics_util import evaluate_metrics

np.random.seed(8899)
tf.set_random_seed(8899)

def main():
    config_path = os.path.join(root_path, 'config/skirt_length.cfg')
    config = Parser(config_path)
    num_classes = 6
    dataset = Dataset(config, tf.estimator.ModeKeys.TRAIN)
    dataset_valid = Dataset(config, tf.estimator.ModeKeys.EVAL)
    dataset_test = Dataset(config, tf.estimator.ModeKeys.PREDICT)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                                allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,
                                                       allow_soft_placement=True))
    model = Model(config, num_classes)
    model.load_pretrain(sess)

    saver = tf.train.Saver(max_to_keep=5)

    best_acc = 0
    best_loss = 100
    test_acc = 0
    test_loss = 0
    valid_acc = 0
    valid_loss = 0
    train_acc = 0
    train_loss = 0

    timedelay = 0
    steps = 0
    batch_size = config.batch_size
    start_time = time.time()

    valid_samples = dataset_valid.next_batch(batch_size)

    while (timedelay < config.timedelay_num) and (steps < config.max_step):
        samples = dataset.next_batch(batch_size)
        steps += 1
        model.train_one_step(sess, samples)
        if steps % config.summary_steps == 0:
            train_loss, train_results = model.test_by_batch(sess, samples)
            train_acc = evaluate_metrics(train_results)

            valid_loss, valid_results = model.test_by_batch(sess, valid_samples)
            valid_acc = evaluate_metrics(valid_results)

            if best_loss > valid_loss or best_acc < valid_acc:
                timedelay = 0
                saver.save(sess, os.path.join(config.exp_dir, 'E01/fashionAI'),
                           global_step=steps)
            else:
                timedelay += 1

            if best_acc < valid_acc:
                best_acc = valid_acc
            if best_loss > valid_loss:
                best_loss = valid_loss

            sys.stdout.write('\nBatches: %d' % steps)
            sys.stdout.write('\nBatch Time: %4fs' % (1.0 * (time.time() - start_time) / config.summary_steps))

            sys.stdout.write('\nTrain acc: %.6f' % train_acc)
            sys.stdout.write('\tTrain Loss: %.6f' % train_loss)
            sys.stdout.write('\nValid acc: %.6f' % valid_acc)
            sys.stdout.write('\tValid Loss: %.6f' % valid_loss)
            sys.stdout.write('\nBest acc: %.6f' % best_acc)
            sys.stdout.write('\tBest Loss: %.6f' % best_loss)
            sys.stdout.write('\n\n')

            #print('\nBatches: %d' % steps, end='')
            #print('\nBatch Time: %4fs' % (1.0 * (time.time() - start_time) / config.summary_steps), end='')

            #print('\nTrain acc: %.6f' % train_acc, end='')
            #print('\tTrain Loss: %.6f' % train_loss, end='')
            #print('\nValid acc: %.6f' % valid_acc, end='')
            #print('\tValid Loss: %.6f' % valid_loss, end='')
            #print('\nBest acc: %.6f' % best_acc, end='')
            #print('\tBest Loss: %.6f' % best_loss, end='')
            #print('\n\n')
            start_time = time.time()

    print('\nModel saved at {}'.format(config.exp_dir))


if __name__ == '__main__':
    main()
