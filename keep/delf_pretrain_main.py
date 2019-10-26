# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np

from nsml import DATASET_PATH
import tensorflow as tf
from data_loader import train_data_loader, batch_iter, split_train_test

from delf import delf_v1
from model.cnn import *
from model.delf_model import *

import sys
import time

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform

_DELF_EXT = '.delf'
_STATUS_CHECK_ITERATIONS = 100
_DISTANCE_THRESHOLD = 0.8



def bind_model(sess):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(file_path)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(file_path, checkpoint))
        else:
            raise NotImplementedError('No checkpoint!')

        print('model loaded!')

    def infer(queries, references):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print("123123123")
        print(dir_path)
        files = os.listdir(os.curdir)
        print(files)
        delf_config = delf_config_pb2.DelfConfig()
        with tf.gfile.FastGFile('../delf_config_example.pbtxt', 'r') as f:
            text_format.Merge(f.read(), delf_config)
        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322
        num_que_images = len(queries)
        num_ref_images = len(references)
        tf.logging.info('done! Found %d images', num_que_images)

        # Tell TensorFlow that the model will be built into the default Graph.

        # Reading list of images.
        # image_path --> query, reference pair
        que_filename_queue = tf.train.string_input_producer(queries, shuffle=False)
        ref_filename_queue = tf.train.string_input_producer(references, shuffle=False)
        reader = tf.WholeFileReader()
        _, que_img = reader.read(que_filename_queue)
        _, ref_img = reader.read(ref_filename_queue)
        que_img_tf = tf.image.decode_jpeg(que_img, channels=3)
        ref_img_tf = tf.image.decode_jpeg(ref_img, channels=3)

        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('input_image:0')
        input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
        input_image_scales = graph.get_tensor_by_name('input_scales:0')
        input_max_feature_num = graph.get_tensor_by_name(
            'input_max_feature_num:0')
        boxes = graph.get_tensor_by_name('boxes:0')
        raw_descriptors = graph.get_tensor_by_name('features:0')
        feature_scales = graph.get_tensor_by_name('scales:0')
        attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
        attention = tf.reshape(attention_with_extra_dim,
                                [tf.shape(attention_with_extra_dim)[0]])
        locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
    boxes, raw_descriptors, delf_config)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start = time.clock()

        os.makedirs('data/features/que_features')
        os.makedirs('data/features/ref_features')
        def write_feature(idx, img_tf, num_images, output_dir, start ):
            if i == 0:
                tf.logging.info('Starting to extract DELF features from images...')
            elif i % _STATUS_CHECK_ITERATIONS == 0:
                elapsed = (time.clock() - start)
                tf.logging.info('Processing image %d out of %d, last %d '
                                'images took %f seconds', i, num_images,
                                _STATUS_CHECK_ITERATIONS, elapsed)
            start = time.clock()

            # # Get next que_image.
            im = sess.run(img_tf)

            # If descriptor already exists, skip its computation.
            out_desc_filename = os.path.splitext(os.path.basename(
                'image_{}'.format(idx)))[0] + _DELF_EXT
            out_desc_fullpath = os.path.join(output_dir, out_desc_filename)
            if tf.gfile.Exists(out_desc_fullpath):
                print("exist?")
                tf.logging.info('Skipping %s', 'image_{}'.format(idx))
                
            else:
            # Extract and save features.
                (locations_out, descriptors_out, feature_scales_out,
                attention_out) = sess.run(
                    [locations, descriptors, feature_scales, attention],
                    feed_dict={
                        input_image:
                            im,
                        input_score_threshold:
                            delf_config.delf_local_config.score_threshold,
                        input_image_scales:
                            list(delf_config.image_scales),
                        input_max_feature_num:
                            delf_config.delf_local_config.max_feature_num
                    })

                feature_io.WriteToFile(out_desc_fullpath, locations_out,
                                    feature_scales_out, descriptors_out,
                                    attention_out)

        print("write que feat")
        for i in range(num_que_images):
            # Write to log-info once in a while.
            write_feature(i, que_img_tf, num_que_images, 'data/features/que_features', start)

        print("write ref feat")
        for i in range(num_ref_images):
            # Write to log-info once in a while.
            write_feature(i, ref_img_tf, num_ref_images, 'data/features/ref_features', start)
            
        # Finalize enqueue threads.
        print("enqueue threads")
        coord.request_stop()
        coord.join(threads)

        def get_cnt_inliers(que_outputs, ref_outputs):

            locations_1, _, descriptors_1, _, _ = que_outputs
            locations_2, _, descriptors_2, _, _ = ref_outputs

            num_features_1 = locations_1.shape[0]
            num_features_2 = locations_2.shape[0]

            # Find nearest-neighbor matches using a KD tree.
            d1_tree = cKDTree(descriptors_1)
            _, indices = d1_tree.query(
                descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

            # Select feature locations for putative matches.
            locations_2_to_use = np.array([ 
                locations_2[i,] 
                for i in range(num_features_2) 
                if indices[i] != num_features_1
            ])
            locations_1_to_use = np.array([
                locations_1[indices[i],]
                for i in range(num_features_2)
                if indices[i] != num_features_1
            ])

            # Perform geometric verification using RANSAC.
            _, inliers = ransac(
                (locations_1_to_use, locations_2_to_use),
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=1000)
            return sum(inliers)

        # Read feature, get inliers 
        result = []
        print("read feat")
        for i in range(num_que_images):
            query = feature_io.ReadFromFile('data/features/que_features/image_{}.delf'.format(i))
            for j in range(num_ref_images):
                reference = feature_io.ReadFromFile('data/features/ref_features/image_{}.delf'.format(j))
                result.append(get_cnt_inliers(query, reference))

        result = np.array(result)
        result = result.reshape((num_que_images, num_ref_images))

        retrieval_results = {}

        print("query list")
        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, result[i].tolist())
            print(type(sim_list))
            print(sim_list)
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list
            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=1)
    args.add_argument('--batch_size', type=int, default=128)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--dev_querynum', type=int, default=200, help='dev split percentage')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    
    input_shape = (224, 224, 3)  # input image shape

    """ initialize """
    delf_config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile('../delf_config_example.pbtxt', 'r') as f:
        text_format.Merge(f.read(), delf_config)
    
    sess = tf.Session()

    # Loading model that will be used.
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                '../parameters/delf_v1_20171026/model')
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name('input_image:0')
    input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
    input_image_scales = graph.get_tensor_by_name('input_scales:0')
    input_max_feature_num = graph.get_tensor_by_name(
        'input_max_feature_num:0')
    boxes = graph.get_tensor_by_name('boxes:0')
    raw_descriptors = graph.get_tensor_by_name('features:0')
    feature_scales = graph.get_tensor_by_name('scales:0')
    attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
    attention = tf.reshape(attention_with_extra_dim,
                            [tf.shape(attention_with_extra_dim)[0]])
    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
                                boxes, raw_descriptors, delf_config)

    init = tf.global_variables_initializer()
    sess.run(init)
    bind_model(sess)
    print("i binded!!")
    if config.pause:
        nsml.paused(scope=locals())
    
    bTrainmode = True

    nsml.save(str(0))