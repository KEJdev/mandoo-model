from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import argparse
import time
import pickle
import sys
import math

import nsml
from nsml import DATASET_PATH
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import numpy as np
import tensorflow as tf
from image_processing import preprocess, get_aug_config
from data_loader import get_assignment_map_from_checkpoint,\
    get_balanced_dual_dataset, get_dual_dataset, image_load, train_data_loader, \
    convert_to_query_db_data, convert_to_query_db_data_fixed_window, \
    convert_to_query_db_data_for_generator
from measure import evaluate_mAP, evaluate_rank
from inference import get_feature
from train_utils import l2_normalize
from loss import batch_hard_triplet_loss
from model.delf_model import *
from imgaug import augmenters as iaa
import imgaug as ia

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from scipy.spatial import cKDTree

local_infer = None

def bind_model(sess):
    global local_infer

    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    def load(file_path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(file_path)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(file_path, checkpoint))
        else:
            raise NotImplementedError('No checkpoint!')
        print('model loaded :' + file_path)

    def infer(queries, references, _query_img=None, _reference_img=None, batch_size=128):

        # load, and process images
        if _query_img is None:
            # not debug
            test_path = DATASET_PATH + '/test/test_data'
            db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]
            queries = [v.split('/')[-1].split('.')[0] for v in queries]
            db = [v.split('/')[-1].split('.')[0] for v in db]
            queries.sort()
            db.sort()
            queries_full_paths = list(map(lambda x: '/data/ir_ph2/test/test_data/query/' + x + '.jpg', queries))
            db_full_path = list(map(lambda x: '/data/ir_ph2/test/test_data/reference/' + x + '.jpg', db))
            _, query_vecs, _, reference_vecs = get_feature(queries_full_paths, db_full_path, sess, batch_size)

        else:
            # debug
            _, query_vecs, _, reference_vecs = get_feature(_query_img, _reference_img, sess, batch_size)
            db = references

        kd_tree = cKDTree(query_vecs)
        _, indices = kd_tree.query(reference_vecs, distance_upper_bound = 0.8)

        retrieval_results = {}

        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)
        print("indices == {}".format(indices))

        def get_ranked_list(query, reference_vecs):
            sim_matrix = np.dot(query, reference_vecs.T)
            sub_indices = np.argsort(sim_matrix)
            sub_indices = np.flip(sub_indices)
            ranked_list = [db[k] for k in sub_indices]
            return ranked_list
        
        queries_name = {}
        for idx, query_name in enumerate(queries):
            queries_name[idx] = query_name

        for i, query in enumerate(query_vecs):
            ref_indices = np.argwhere(indices == i)
            if len(ref_indices) == 0:
                ranked_list = get_ranked_list(query, reference_vecs)
            else:
                clustered_ref = reference_vecs[ref_indices]
                left_over_indices = np.argwhere(indices != i)
                left_over_ref = reference_vecs[left_over_indices]
                clustered_ranked_list = get_ranked_list(query, clustered_ref)
                left_over_ranked_list = get_ranked_list(query, left_over_ref)
                ranked_list = clustered_ref + left_over_ref
            retrieval_results[queries_name[i]] = ranked_list
        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    
    if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--debug', action='store_true', help='debug mode')
    args.add_argument('--debug_data', type=str, default="./debug_data", help='debug_data')
    args.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    args.add_argument('--dev_querynum', type=int, default=300, help='dev split percentage')
    args.add_argument('--dev_referencenum', type=int, default=20, help='dev split percentage')

    # augmentation
    args.add_argument('--augmentation', action='store_true', help='apply random crop in processing')
    args.add_argument('--crop', action='store_true', help='set crop images')
    args.add_argument('--fliplr', action='store_true', help='set fliplr')
    args.add_argument('--flipud', action='store_true', help='set flipud')
    args.add_argument('--gausian', action='store_true', help='set gausian')
    args.add_argument('--dropout', action='store_true', help='set dropout')
    args.add_argument('--noise', action='store_true', help='set noise')
    args.add_argument('--rotate', action='store_true', help='rotate -45 degree to +45 degree')
    # loss calculation
    args.add_argument('--train_logits', action='store_true', help='train similarity and logit jointly')
    args.add_argument('--train_sim', action='store_true', help='train similarity and logit jointly')
    args.add_argument('--train_sim_dist', action='store_true', help='train similarity and logit jointly using squared loss')
    args.add_argument('--train_max_neg', action='store_true', help='train max negative loss')
    args.add_argument('--train_max_neg_topk', type=int, default=5, help='set top_k max negative')
    args.add_argument('--train_triplet', action="store_true", help="train triplet loss")
    # pre trained model
    args.add_argument('--pretrained_model', type=str, default=None, help='restore pretrained model')
    args.add_argument('--stop_gradient_sim', action='store_true', help='stop gradient similarity')
    args.add_argument('--skipcon_attn', action='store_true', help='skip connection attention')
    args.add_argument('--logit_concat_sim', action='store_true', help='skip connection attention')
    args.add_argument('--resnet_type', type=str, default='resnet_v1_50/block4', help='resnet version')

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)
    
    config = args.parse_args()
    print("Model configuration", config)

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
