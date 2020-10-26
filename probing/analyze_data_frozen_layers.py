import sys, os, re, json
import itertools
import collections
import matplotlib.pyplot as plt
from importlib import reload
import pandas as pd
import numpy as np
from sklearn import metrics

BERT_LAYERS=12

import analysis
reload(analysis)

# getting preds for all layers
preds_dict = dict()
score_dict = dict()
score_dict_long = dict()
for i in list(range(BERT_LAYERS + 1)):
    run_dir = "/cs/labs/oabend/lovodkin93/jiant_rep/jiant_outputs/bert-base-uncased-mix_" + str(i) + "-edges-coref-ontonotes/run"
    preds = analysis.Predictions.from_run(run_dir, 'edges-coref-ontonotes', 'test')
    preds_dict[f'layer_{i}'] = preds

# calculating wide F1 for all layers
# for i, layer in enumerate(preds_dict):
#     preds= preds_dict[layer]
#     wide_df = preds.target_df_wide
#     scores_by_label = {}
#     for label in preds.all_labels:
#         y_true = wide_df['label.true.' + label]
#         y_pred = wide_df['preds.proba.' + label] >= 0.5
#         score = metrics.f1_score(y_true=y_true, y_pred=y_pred)
#         scores_by_label[label] = score
#     scores = pd.Series(scores_by_label)
#     score_dict[layer] = scores

# calculating long F1 for all layers
for i, layer in enumerate(preds_dict):
    preds= preds_dict[layer]
    long_df = preds.target_df_long
    score_dict_long[layer] = metrics.f1_score(y_true=long_df['label.true'], y_pred=(long_df['preds.proba'] >= 0.5))
# #ploting the data
# x_axis = list(range(1, BERT_LAYERS + 1))
# # label 0 plotting
# plt.figure(1)
# y_axis_0 = [score_dict[f'layer_{i}'][0] for i in x_axis]
# plt.plot(x_axis, y_axis_0, '-ok')
# plt.xlabel('num of layers')
# plt.ylabel('F1 score')
# plt.title('F1 score for label 0 - no coreference ')
# plt.show()
#
# # label 1 plotting
# plt.figure(2)
# y_axis_1 = [score_dict[f'layer_{i}'][1] for i in x_axis]
# plt.plot(x_axis, y_axis_1, '-ok')
# plt.xlabel('num of layers')
# plt.ylabel('F1 score')
# plt.title('F1 score for label 1 - there is coreference ')
# plt.show(2)
#
# # total F1 plotting
# plt.figure(3)
# y_axis_total = [score_dict[f'layer_{i}'].mean() for i in x_axis]
# plt.plot(x_axis, y_axis_total, '-ok')
# plt.xlabel('num of layers')
# plt.ylabel('F1 score')
# plt.title('Total F1 score')
# plt.show(3)
#
# # plot all of them together
# plt.figure(4)
# plt.plot(x_axis, y_axis_0, '-ok', label='0 tag')
# plt.plot(x_axis, y_axis_1, '-or', label='1 tag')
# plt.plot(x_axis, y_axis_total, '-ob', label='total')
# plt.text(5.16343, 0.967555, '0 tag')
# plt.text(5.20419, 0.880246, '1 tag')
# plt.text(5.12266, 0.922031, 'total')
# plt.show(4)


# # printing F1 for all layers
# for i, layer in enumerate(score_dict, 1):
#     scores = score_dict[layer]
#     print("F1 of " + layer + " is:")
#     print(scores[0])
#     print("Macro average F1: %.04f" % scores.mean())

# calculating the expected layer
prev_score = score_dict_long["layer_0"]
numerator = 0
denominator = 0
for i, layer in enumerate(score_dict_long):
    if (i==0):
        continue
    curr_score = score_dict_long[layer]
    delta = curr_score - prev_score
    numerator = numerator + (i*delta)
    denominator = denominator + delta
    prev_score = curr_score
exp_layer = numerator/denominator
print("expected layer is : " + str(exp_layer))

