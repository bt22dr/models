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
import collections

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

if __name__ == "__main__":
  dataset_dir = "/home/jysong/Documents/data/cdiscount/cdiscount_test_images"
  model_name = 'nasnet_mobile'
  ckpt_path = \
    "/home/jysong/Documents/log/cdiscount/train_e2e_lr0.01"
  batch_size = 128
  image_size = 224

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_dir", help="image dataset directory to be processed")
  parser.add_argument("--model_name", help="model name (ex. nasnet_mobile, inception_resnet_v2, ...")
  parser.add_argument("--ckpt_path", help="graph/model to be executed")
  parser.add_argument("--batch_size", type=int, help="batch size")
  parser.add_argument("--image_size", type=int, help="image width, height")
  args = parser.parse_args()

  if args.ckpt_path:
    ckpt_path = args.ckpt_path
  if args.model_name:
    model_name = args.model_name
  if args.dataset_dir:
    dataset_dir = args.dataset_dir
  if args.batch_size:
    batch_size = args.batch_size

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  file_names = glob.glob(os.path.join(dataset_dir, 'cdiscount_images/*.jpg'))
  #file_names = file_names[:10]
  num_total_images = len(file_names)

  network_fn = nets_factory.get_network_fn(model_name, num_classes=5270, is_training=False)

  file_names = tf.convert_to_tensor(file_names)
  file_name = tf.train.slice_input_producer([file_names], num_epochs=1, shuffle=False)

  file_reader = tf.read_file(file_name[0])
  image = tf.image.decode_jpeg(file_reader, channels=3)
  image_preprocessing_fn = preprocessing_factory.get_preprocessing('inception', is_training=False)

  image_size = args.image_size or network_fn.default_image_size
  processed_image = image_preprocessing_fn(image, image_size, image_size)

  images, image_names = tf.train.batch(
      [processed_image, file_name], 
      batch_size=batch_size, 
      allow_smaller_final_batch=True)

  logits, end_points = network_fn(images)

  variables_to_restore = slim.get_variables_to_restore()

  prediction = end_points['Predictions']

  if tf.gfile.IsDirectory(ckpt_path):
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)

  tf.logging.info('Evaluating %s' % ckpt_path)
  init_fn = slim.assign_from_checkpoint_fn(ckpt_path, variables_to_restore, ignore_missing_vars=True) # TODO::

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    #sess.run(tf.global_variables_initializer())
    init_fn(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    product_probs_dict = collections.defaultdict(lambda: 0)

    print("_id,category_id")
    while True:
      try:
        preds, names = sess.run([prediction, image_names])
        inds = np.argmax(preds, axis=1)

        for i, ind in enumerate(inds):
          product_id = os.path.basename(names[i][0]).decode().split('-')[0]
          #print(product_id, labels_to_names[ind], sep=',')
          product_probs_dict[product_id] += preds[i]
      except tf.errors.OutOfRangeError:
        break

    coord.request_stop()
    coord.join(threads)

  for k,v in product_probs_dict.items():
    print(k, labels_to_names[np.argmax(v)], sep=',')
