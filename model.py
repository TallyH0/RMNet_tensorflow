from rmnet import RMNet
from data_loader import Data_loader
import tensorflow as tf
import numpy as np
from loss import structure_loss
from ops import config_common
import os
from time import time

class RMNet_model():
  def __init__(self, cfg):
    self.cfg = cfg

  def build(self, train):
    if train:
      self.data_loader = Data_loader(self.cfg.txt_dataset, self.cfg.img_h, self.cfg.img_w, True)
      self.ds_size, num_class = self.data_loader.get_status()
      step_per_epoch = self.ds_size // self.cfg.batch_size
      
      self.placeholder_image = tf.placeholder(tf.float32, [self.cfg.batch_size, self.cfg.img_h, self.cfg.img_w, self.cfg.img_c], name='input')
      self.placeholder_label = tf.placeholder(tf.int32, [self.cfg.batch_size], name='label')
      placeholder_is_train = True
      
      pre_process = self.placeholder_image / 127.5 - 1
      pre_process = tf.image.random_flip_left_right(pre_process)
      self.global_step = tf.Variable(0, False, name='global_step')
      placeholder_dropout = tf.train.piecewise_constant(self.global_step, [100 * step_per_epoch, 200 * step_per_epoch, 300 * step_per_epoch, 400 * step_per_epoch],
                                      [0.9, 0.9, 0.9, 1., 1.])
      lr = tf.train.piecewise_constant(self.global_step, [100 * step_per_epoch, 200 * step_per_epoch, 300 * step_per_epoch, 400 * step_per_epoch],
                                      [1e-3, 1e-4, 5e-4, 1e-4, 5e-5])

      embedding_local, embedding_global, end_point = RMNet(pre_process, placeholder_is_train, placeholder_dropout, self.cfg.activation, num_class)
      logit_local = end_point['logit_local']
      logit_global = end_point['logit_global']
      
      with tf.name_scope('loss_local'):
          loss_local_center, loss_local_push, loss_local_gpush, center, local_margin = structure_loss(embedding_local, self.placeholder_label, self.cfg.alpha_center_loss, num_class, scope = 'structure_local')
          loss_local_softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_local, labels=self.placeholder_label))
          loss_local = loss_local_softmax + self.cfg.factor_center_loss * loss_local_center + self.cfg.factor_push_loss * loss_local_push + self.cfg.factor_gpush_loss * loss_local_gpush
          
      with tf.name_scope('loss_global'):
          loss_global_center, loss_global_push, loss_global_gpush, center, global_margin = structure_loss(embedding_global, self.placeholder_label, self.cfg.alpha_center_loss, num_class, scope = 'structure_global')
          loss_global_softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_global, labels=self.placeholder_label))
          loss_global = loss_global_softmax + self.cfg.factor_center_loss * loss_global_center + self.cfg.factor_push_loss * loss_global_push + self.cfg.factor_gpush_loss * loss_global_gpush
          
      loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])
      loss = loss_local + loss_global + self.cfg.weight_decay * loss_l2
      acc_global = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_global, axis=1), tf.cast(self.placeholder_label, tf.int64)), tf.float32))
      acc_local = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_local, axis=1), tf.cast(self.placeholder_label, tf.int64)), tf.float32))
          
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
          opt = tf.train.MomentumOptimizer(lr, 0.9)
          self.train_op = tf.group([update_ops, opt.minimize(loss, global_step = self.global_step)])
          
      self.saver = tf.train.Saver(max_to_keep=1)
      tf.summary.scalar('loss/loss', loss)
      tf.summary.scalar('loss/loss_softmax', loss_local_softmax + loss_global_softmax)
      tf.summary.scalar('loss/loss_center', self.cfg.factor_center_loss * (loss_local_center + loss_global_center))
      tf.summary.scalar('loss/loss_push', self.cfg.factor_push_loss * (loss_local_push + loss_global_push))
      tf.summary.scalar('loss/loss_gpush', self.cfg.factor_gpush_loss * (loss_local_gpush + loss_global_gpush))
      tf.summary.scalar('loss/loss_l2', loss_l2 * self.cfg.weight_decay)
      tf.summary.scalar('acc/acc_global', acc_global)
      tf.summary.scalar('acc/acc_local', acc_local)
      tf.summary.scalar('margin/local', local_margin)
      tf.summary.scalar('margin/global', global_margin)
      self.merged = tf.summary.merge_all()
      self.sess = tf.Session(config=config_common())

    else:
      self.placeholder_image = tf.placeholder(tf.float32, [1, None, None, self.cfg.img_c], name='input')
      pre_process = tf.image.resize_bilinear(self.placeholder_image, (self.cfg.img_h, self.cfg.img_w))
      pre_process = (pre_process / 127.5) - 1
      emb_local, emb_global, end_point = RMNet(pre_process, False, 1, self.cfg.activation)
      self.embedding = emb_global
      self.saver = tf.train.Saver()
      self.sess = tf.Session(config=config_common())


  def train(self):
    checkpoint = tf.train.latest_checkpoint(self.cfg.model_dir)
    summary_writer = tf.summary.FileWriter(self.cfg.model_dir)
    if checkpoint:
      self.saver.restore(self.sess, checkpoint)
    else:
      self.sess.run(tf.global_variables_initializer())

    train_progress = 0
    while train_progress < 100:
        try:
            begin = time()
            batch_image, batch_label = self.data_loader.batch(self.cfg.batch_size, cfg.param_erasing)
            feed_dict = {
                self.placeholder_image : batch_image,
                self.placeholder_label : batch_label
            }
            _, run_summary, run_step = self.sess.run([self.train_op, self.merged, self.global_step], feed_dict = feed_dict)
            print(time() - begin, 'second')
            train_progress = 100 * run_step * self.cfg.batch_size / (self.ds_size * self.cfg.max_epoch)
            print(train_progress, '%')
            if run_step % 100 == 99:
                summary_writer.add_summary(run_summary, run_step)
            
            if run_step % 10000 == 9999:
                self.saver.save(self.sess, os.path.join(self.cfg.model_dir, 'model'), global_step=run_step)
        except Exception as e:
            print(e)
            break
            
    self.saver.save(self.sess, os.path.join(self.cfg.model_dir, 'model'), global_step=run_step)

  def load(self):
    checkpoint = tf.train.latest_checkpoint(self.cfg.model_dir)
    if checkpoint:
      self.saver.restore(self.sess, checkpoint)
      print('Restore model from', checkpoint)
    else:
      print('Error : No checkpoint')
      exit(-1)

  def inference(self, img):
    feed_dict = {
      self.placeholder_image : [img]
    }
    result = self.sess.run(self.embedding, feed_dict = feed_dict)
    return np.squeeze(result)

    
if __name__ == '__main__':
  import importlib
  config = 'configuration'
  cfg = importlib.import_module(config)

  network = RMNet_model(cfg)
  network.build(train=True)