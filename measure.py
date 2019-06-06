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


