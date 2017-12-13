# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

if __name__ == "__main__":
  dataset_dir = "/home/jysong/Documents/data/cdiscount/cdiscount_test_images"
  ckpt_path = \
    "/home/jysong/Documents/log/cdiscount/train_e2e_lr0.01"
  input_height = 224
  input_width = 224

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_dir", help="image dataset directory to be processed")
  parser.add_argument("--ckpt", help="graph/model to be executed")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  args = parser.parse_args()

  if args.ckpt:
    ckpt_path = args.ckpt
  if args.dataset_dir:
    dataset_dir = args.image_dataset
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width

  file_names = glob.glob(os.path.join(dataset_dir, 'cdiscount_images/*-0.jpg'))
  #file_name = file_names[0]

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  graph = tf.Graph()
  with graph.as_default():
    network_fn = nets_factory.get_network_fn('nasnet_mobile', num_classes=5270, is_training=False)

    file_name = tf.placeholder(tf.string)
    file_reader = tf.read_file(file_name)
    image = tf.image.decode_jpeg(file_reader, channels=3)
    image_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile', is_training=False)
    processed_image = image_preprocessing_fn(image, input_height, input_width)
    images = tf.expand_dims(processed_image, 0)

    logits, end_points = network_fn(images)

    variables_to_restore = slim.get_variables_to_restore()

    prediction = end_points['Predictions']

    if tf.gfile.IsDirectory(ckpt_path):
      ckpt_path = tf.train.latest_checkpoint(ckpt_path)

    tf.logging.info('Evaluating %s' % ckpt_path)
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, variables_to_restore, ignore_missing_vars=True) # TODO::

    print("_id,category_id")
    with tf.Session() as sess:
      init_fn(sess)
      for f in file_names:
        pred = sess.run([prediction], feed_dict={file_name: f})
        pred = np.squeeze(pred)
        inds = np.argmax(pred)
        print(os.path.basename(f).split('-')[0], labels_to_names[inds], sep=',')
