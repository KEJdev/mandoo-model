import numpy as np 
import tensorflow as tf

def evaluate_mAP(result):
    mAP = 0.
    for _, (query, ranklist) in result:
        query_class = query.split("@")[0]
        correct_count = 0.
        pk_sum = 0.
        for i, item in enumerate(ranklist):
            item_class= item.split("@")[0]
            if query_class == item_class:
                correct_count += 1.
                pk_sum += correct_count/(i+1.)
        if correct_count == 0:
            continue
        mAP += pk_sum / correct_count
    return mAP / len(result)

def evaluate_rank(result):
    mAP = 0.
    min_first_1_at_K = 0.
    mean_recall_at_K = [0 for i in range(len(result[0][1][1]))]
    for _, (query, ranklist) in result:
        query_class = query.split("@")[0]
        correct_count = 0
        pk_sum = 0.
        for i, item in enumerate(ranklist):
            item_class= item.split("@")[0]
            if query_class == item_class:
                correct_count += 1.
                pk_sum += correct_count/(i+1.)
        if not correct_count:
            correct_count = 1
        recall_count = 0.
        min_counted = False
        for k, item in enumerate(ranklist):
            item_class = item.split("@")[0]
            if query_class == item_class:
                recall_count += 1.
            mean_recall_at_K[k] += recall_count/(correct_count)
            if recall_count == correct_count and not min_counted:
                min_first_1_at_K += (k+1)
                min_counted = True
        mAP += pk_sum / correct_count
    min_first_1_at_K = min_first_1_at_K / len(result)
    mean_recall_at_K = [recall/len(result) for recall in mean_recall_at_K]
    return mAP / len(result), mean_recall_at_K, min_first_1_at_K
