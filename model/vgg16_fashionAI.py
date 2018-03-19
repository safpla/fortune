import tensorflow as tf
import numpy as np
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from model.vgg16 import vgg_16
slim = tf.contrib.slim

class Model:
    def __init__(self, config, num_classes):
        self.config = config
        self.num_classes = num_classes

        self._create_placeholder()
        self._inference_graph()

    def _create_placeholder(self):
        self.input_plh = tf.placeholder(tf.float32,
                                        [None,
                                         self.config.image_height,
                                         self.config.image_width,
                                         self.config.image_channels],
                                        name='X_inputs')

        self.label_plh = tf.placeholder(tf.int32, [None, self.num_classes],
                                        name='y_inputs')

        self.is_training = tf.placeholder(dtype=tf.bool,
                                          shape=[],
                                          name='is_training')

    def _inference_graph(self):
        logits, _ = vgg_16(self.input_plh,
                           num_classes=self.num_classes,
                           is_training=self.is_training,
                           dropout_keep_prob=0.5,
                           scope='vgg_16')
        self.variables_to_restore = slim.get_variables_to_restore(
            exclude=['vgg_16/fc8'])
        self.logits = logits

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.cast(self.label_plh, tf.float32),
            logits=logits,
            name='softmax')
        loss = tf.reduce_mean(loss)
        self.loss = loss

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.lr_policy_params['base_lr'],
            beta1=self.config.lr_policy_params['beta1'],
            beta2=self.config.lr_policy_params['beta2'],
            epsilon=self.config.lr_policy_params['epsilon'])

        #self.optimizer = tf.train.AdamOptimizer(
        #    learning_rate=self.config.lr_policy_params['base_lr'],
        #    epsilon=self.config.lr_policy_params['epsilon']).minimize(loss)

        #self.train_step = self.optimizer

        gvs = self.optimizer.compute_gradients(loss)
        #capped_gvs = [((tf.clip_by_norm(grad, self.config.grad_lim)), var)
        #              for grad, var in gvs]
        #self.train_step = self.optimizer.apply_gradients(capped_gvs)
        self.train_step = self.optimizer.apply_gradients(gvs)

    def load_pretrain(self, sess):
        restorer = tf.train.Saver(self.variables_to_restore)
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, self.config.pretrain_model_file)

    def load_model(self, sess, checkpoint_dir, checkpoint=None):
        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        if not checkpoint:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                checkpoint = ckpt.model_checkpoint_path
        print(checkpoint)
        try:
            print('loading model')
            restorer.restore(sess, checkpoint)
        except:
            raise('failed to load the model')

    def train_one_step(self, sess, samples):
        img = np.asarray(samples['img'])
        label = np.asarray(samples['label'])
        feed_dict = {self.input_plh: img,
                     self.label_plh: label,
                     self.is_training: True}
        self.train_step.run(feed_dict)

    def test_by_batch(self, sess, samples):
        img = np.asarray(samples['img'])
        label = np.asarray(samples['label'])
        feed_dict = {self.input_plh: img,
                     self.label_plh: label,
                     self.is_training: False}
        checkout = [self.logits, self.loss, self.label_plh]
        r = sess.run(checkout, feed_dict=feed_dict)
        loss = r[1]
        results = {}
        results['logits'] = r[0]
        results['label'] = label
        return loss, results

    def test_by_dataset(self, sess, dataset):
        dataset.reset()
        loss = []
        results = {}
        results['logits'] = []
        results['label'] = []
        while dataset.epochs_completed == 0:
            samples = dataset.next_batch(self.config.batch_size)

