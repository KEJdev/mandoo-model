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
    convert_to_query_db_data_for_generator, get_triplet_dataset
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

local_infer = None

# bind training model with nsml
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
            _, query_vecs, _, reference_vecs = get_feature(queries_full_paths, db_full_path, sess)

        else:
            # debug
            _, query_vecs, _, reference_vecs = get_feature(_query_img, _reference_img, sess, config.batch_size)
            db = references

        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        # query = 1, ref = 10
        # sim_matrix[0] = [0.1, 0.56, 0.2, 0.5, ....]
        # Sort cosine similarity values to rank it
        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [db[k] for k in indices[i]]
            ranked_list = ranked_list[:5000]
            retrieval_results[query] = ranked_list
        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)
    local_infer = infer

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
    args.add_argument('--margin', type=float, default=10.0, help='set triplet loss margin')

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)

    config = args.parse_args()
    print("Model configuration", config)

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size

    """-------- Model Part -------------------------------------------------"""
    num_classes = 1384
    input_shape = (224, 224, 3)  # input image shape

    # set input placeholders
    X1 = tf.placeholder(
        tf.float32,
        [None, input_shape[0], input_shape[1], 3],
        name="input_X1")
    Y1 = tf.placeholder(tf.float32, [None, num_classes], name="input_Y1")
    X2 = tf.placeholder(
        tf.float32,
        [None, input_shape[0], input_shape[1], 3],
        name="input_X2"
    )
    Y2 = tf.placeholder(tf.float32, [None, num_classes], name="input_Y2")
    X3 = tf.placeholder(
        tf.float32,
        [None, input_shape[0], input_shape[1], 3],
        name="input_X3"
    )
    Y3 = tf.placeholder(tf.float32, [None, num_classes], name="input_Y3")


    # init model
    global_step = tf.Variable(0, name="mandoo_global_step")

    model = triplet_model(X1, X2, X3, num_classes,
                                resnet_v=config.resnet_type,
                                skipcon_attn=config.skipcon_attn,
                                stop_gradient_sim=config.stop_gradient_sim,
                                logit_concat_sim=config.logit_concat_sim)

    # define loss function to optimize 
    acc_logit = tf.zeros([])
    loss_squared_logit = tf.zeros([])
    loss_crossent_logit = tf.zeros([])
    loss_sim = tf.zeros([])
    loss_sim_dist = tf.zeros([])
    loss_max_neg = tf.zeros([])
    acc_sim = tf.zeros([])
    loss_triplet = tf.zeros([])
    Y_sim = tf.expand_dims(tf.reduce_sum(Y1 * Y2, axis=1), axis=-1)

    if config.train_logits:
        loss_crossent_1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits_1, labels=Y1)
        loss_crossent_2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits_2, labels=Y2)
        loss_crossent_3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits_3,labels=Y3)
        loss_crossent_logit = tf.reduce_sum(
            loss_crossent_1 + loss_crossent_2 + loss_crossent_3)

        loss_squared_1 = tf.losses.mean_squared_error(labels=Y1, predictions=tf.nn.softmax(model.logits_1))
        loss_squared_2 = tf.losses.mean_squared_error(labels=Y2, predictions=tf.nn.softmax(model.logits_2))
        loss_squared_3 = tf.losses.mean_squared_error(labels=Y3, predictions=tf.nn.softmax(model.logits_3))
        loss_squared_logit = tf.reduce_sum(loss_squared_1 + loss_squared_2 + loss_squared_3)

        pred_1 = tf.argmax(model.logits_1, 1, name="pred_1")
        pred_2 = tf.argmax(model.logits_2, 1, name="pred_2")
        pred_3 = tf.argmax(model.logits_3, 1, name="pred_3")
        acc_1 = tf.reduce_mean(tf.cast(tf.equal(pred_1, tf.argmax(Y1, 1)), "float"))
        acc_2 = tf.reduce_mean(tf.cast(tf.equal(pred_2, tf.argmax(Y2, 1)), "float"))
        acc_3 = tf.reduce_mean(tf.cast(tf.equal(pred_3, tf.argmax(Y3, 1)), "float"))
        acc_logit = (acc_1 + acc_2 + acc_3) / 3.0

    if config.train_sim:
        Y_sim = tf.expand_dims(tf.reduce_sum(Y1 * Y2, axis=1), axis=-1)
        pred_sim = tf.cast(tf.greater(tf.nn.sigmoid(model.similarity), 0.5), tf.int64)
        acc_sim = tf.reduce_mean(tf.cast(tf.equal(pred_sim, tf.cast(Y_sim, tf.int64)), "float"))
        loss_sim = tf.nn.sigmoid_cross_entropy_with_logits(logits=model.similarity, labels=Y_sim)
        loss_sim = tf.reduce_sum(loss_sim)

    if config.train_sim_dist:
        sig_f1 = tf.nn.sigmoid(model.feat_attn_1)
        sig_f2 = tf.nn.sigmoid(model.feat_attn_2)
        Y_sim_dist = (tf.expand_dims(tf.reduce_sum(Y1 * Y2, axis=1), axis=-1) - 0.5) * 2
        loss_sim_dist = Y_sim_dist * tf.losses.mean_squared_error(labels=sig_f1, predictions=sig_f2)
        loss_sim_dist = tf.reduce_sum(loss_sim_dist)

    if config.train_max_neg:
        logit_sum_1, _ = tf.reduce_sum(model.logits_1, axis=1)
        pred_value_1, _ = tf.nn.top_k(model.logits_1, k=1)
        logit_sum_2, _ = tf.reduce_sum(model.logits_2, axis=1)
        pred_value_2, _ = tf.nn.top_k(model.logits_2, k=1)
        
        loss_max_neg_1 = tf.reduce_sum(tf.maximum(tf.subtract(logit_sum_1, pred_value_1), 0.0))
        loss_max_neg_2 = tf.reduce_sum(tf.maximum(tf.subtract(logit_sum_2, pred_value_2), 0.0))

        loss_max_neg = (loss_max_neg_1 + loss_max_neg_2) / 2.0

    if config.train_triplet:
        positive_dist = tf.sqrt(tf.reduce_sum(
            tf.pow((model.feat_attn_1 - model.feat_attn_2), 2)))
        negative_dist = tf.sqrt(tf.reduce_sum(
            tf.pow((model.feat_attn_1 - model.feat_attn_3), 2)))
        loss_triplet = tf.maximum(
            0.0, positive_dist - negative_dist + config.margin)

    loss = loss_sim + loss_squared_logit + loss_crossent_logit \
        + loss_sim_dist + loss_max_neg + loss_triplet

    # optimize the loss
    optimizer = tf.train.AdamOptimizer(config.lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    """------------------------------------------------------------------------------------"""
    # start session
    sess = tf.Session()

    # load model from scratch or pretrained_model(resnet_v1_50)
    if config.pretrained_model == None:
        print("Initialize model from the scratch")
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        print("Initialize model from pretrained_model")
        all_vars = tf.global_variables()
        # load pretrained_model checkpoint
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
             tvars=all_vars,
             init_checkpoint=config.pretrained_model
         )
        for var in initialized_variable_names:
            print(str(var) + " *INIT FROM CKPT* ")           
        print("Total {:g} variables are restored from ckpt : {}".format(
            len(initialized_variable_names), str(config.pretrained_model)))

        tf.train.init_from_checkpoint(
            config.pretrained_model, assignment_map)

        # find uninitialized variables and initialize it
        is_initialized = sess.run([tf.is_variable_initialized(var)
                                   for var in all_vars])
        not_initialized_vars = [var
                                for (var, f) in zip(all_vars, is_initialized)
                                if not f]
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    saver = tf.train.Saver()
    bind_model(sess)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True
        
        """ Load data """
        print(DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path,
                        output_path=output_path)
        else:
            train_dataset_path = config.debug_data
            train_data_loader(train_dataset_path, output_path=output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        queries, references, queries_img, reference_img \
            = convert_to_query_db_data_for_generator(img_list, label_list, input_shape, config.dev_querynum, config.dev_referencenum)
        print("mAP devset : query(%d), reference(%d) " % (len(queries), len(references)))

        dataset = get_triplet_dataset(
            train_dataset_path, batch_size, nb_epoch, num_classes=num_classes)

        size_of_epoch = len(label_list) * 2

        iterator = dataset.make_initializable_iterator()
        img_batch_1, img_batch_2, img_batch_3, label_batch_1, label_batch_2, label_batch_3 = iterator.get_next()

        sess.run([iterator.initializer])

        # set data augmentation
        seq = iaa.Sequential(get_aug_config(config))

        # train batches
        def train_step(img_batch_1, label_batch_1, img_batch_2, label_batch_2,
                       img_batch_3, label_batch_3, config):
            
            images_1, labels_1, images_2, labels_2, images_3, labels_3 = sess.run(
                [img_batch_1, label_batch_1, img_batch_2, label_batch_2,
                 img_batch_3, label_batch_3])

            if config.augmentation:
                images_1 = seq.augment_images(images_1)
                images_2 = seq.augment_images(images_2)
                images_3 = seq.augment_images(images_3)

            feed_dict = {
                X1: images_1, Y1: labels_1,
                X2: images_2, Y2: labels_2,
                X3: images_3, Y3: labels_3,
            }
            outputs = sess.run(
                    [train_op,
                     loss,
                     loss_crossent_logit,
                     loss_squared_logit,
                     loss_sim,
                     loss_sim_dist,
                     loss_max_neg,
                     global_step,
                     acc_logit,
                     acc_sim,
                     loss_triplet ], feed_dict)
            return outputs

        """ Training loop """
        print("Mandoo model train start!..")
        best_mAP = 0.
        mAP = 0
        best_mAP_step = 0
        best_loss = 99999.
        best_loss_step = 0
        best_min_first_K = 99999.
        best_min_first_K_step = 0.
        start_time = time.time()

        epoch = 0
        while True:
            try:
                _, train_loss, train_loss_logit_cross, train_loss_logit_sqrt, train_loss_sim, train_loss_sim_dist, \
                train_loss_max_neg, step, train_acc_logit, train_acc_sim, train_loss_triplet = train_step(
                    img_batch_1, label_batch_1, img_batch_2, label_batch_2,
                    img_batch_3, label_batch_3,config)
                prev_epoch = epoch
                epoch = math.floor(step * batch_size / size_of_epoch)
                # print process
                if step % 30 == 0 or config.debug:
                    print_second = int(time.time() - start_time)
                    start_time = time.time()
                    print("[{:g} sec] epoch {:g}, step {:g}, acc_logit {:g}, acc_sim {:g}".format(
                        print_second, epoch, step, train_acc_logit, train_acc_sim))
                    print("..... total_loss {:g}, loss_logit_cross {:g}, loss_logit_sqrt {:g}, loss_sim {:g}, "
                          "loss_sim_dist {:g}, loss_max_neg {:g}, loss_triplet {:g}".format(
                        train_loss, train_loss_logit_cross, train_loss_logit_sqrt, train_loss_sim,
                        train_loss_sim_dist, train_loss_max_neg, train_loss_triplet))

                do_save = False
                if step % 150 == 0 or (config.debug and step % 1 == 0):
                    infer_result = local_infer(queries, references, queries_img, reference_img, batch_size)
                    mAP, mean_recall_at_K, min_first_1_at_K = evaluate_rank(infer_result)

                    if best_min_first_K >= min_first_1_at_K:
                        best_min_first_K = min_first_1_at_K
                        best_min_first_K_step = step
                        print("----> First_K @ 1 recall : %d / %d" % (min_first_1_at_K, len(mean_recall_at_K)))
                        do_save = True

                    if best_mAP <= mAP :
                        best_mAP = mAP
                        print("----> Best mAP : best-mAP {:g}".format(best_mAP))
                        do_save = True

                if epoch - prev_epoch == 1:
                    print("----> Epoch changed saving")
                    do_save = True

                if do_save:
                    # save model
                    nsml.report(summary=True, epoch=str(step), epoch_total=nb_epoch)
                    nsml.save(step)
                    print("Model saved : %d step" % step)
                    print("=============================================================================================================")


            except tf.errors.OutOfRangeError:
                print("finish train!")
                break
