# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf
import glob

from datasets import dataset_utils
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 8, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 180, 'Eval image size')

tf.app.flags.DEFINE_integer(
    'num_classes', 5270, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
#  if not FLAGS.dataset_dir:
#    raise ValueError('You must supply the dataset directory with --dataset_dir')

    labels_to_names = None
    if dataset_utils.has_labels(FLAGS.dataset_dir):
        labels_to_names = dataset_utils.read_label_file(FLAGS.dataset_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        filenames = glob.glob(os.path.join(FLAGS.dataset_dir, 'cdiscount_images/*-0.jpg'))
        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)

        # image = tf.image.decode_jpeg(value)
        image = tf.image.decode_jpeg(value, channels=3)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)


        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
        
        images = tf.train.batch(
            [image],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, end_points = network_fn(images)

        if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
          variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
          variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
          variables_to_restore = slim.get_variables_to_restore()

        prediction = end_points['Predictions']


        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        #init_fn = slim.assign_from_checkpoint_fn(
        #'test_res/inception_resnet_v2_2016_08_30.ckpt', slim.get_model_variables('InceptionResnetV2'))
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables('InceptionResnetV2'))

        with tf.Session() as sess:
            init_fn(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            num_batches = math.ceil(len(filenames) / float(FLAGS.batch_size))
            print('_id,category_id')
            for i in range(num_batches):
                pred = sess.run([prediction])[0]
                
                inds = np.argmax(pred,axis=1)
                #if i == (num_batches-1):
                #    inds = inds[:(len(filenames)%FLAGS.batch_size)]
                for ii, ind in enumerate(inds):
                    filename_idx = ((i*FLAGS.batch_size)+ii) % len(filenames)
                    if ((i*FLAGS.batch_size)+ii) == len(filenames):
                        break
                    product_id = os.path.basename(filenames[filename_idx]).split('-')[0]
                    print(product_id, labels_to_names[ind], sep=',')
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
  tf.app.run()
