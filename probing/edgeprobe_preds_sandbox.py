import sys, os, re, json
import itertools
import collections
from importlib import reload
import pandas as pd
import numpy as np
from sklearn import metrics

import analysis
reload(analysis)

run_dir = "/cs/labs/oabend/lovodkin93/jiant_rep/jiant_outputs/bert-base-uncased-mix_12-edges-coref-ontonotes/run"
print("starting predictions extraction")
preds = analysis.Predictions.from_run(run_dir, 'edges-coref-ontonotes', 'test')
print("Number of examples: %d" % len(preds.example_df))
print("Number of total targets: %d" % len(preds.target_df))
print("Labels (%d total):" % len(preds.all_labels))
print(preds.all_labels)
print("---------------------EXAMPLE INFO----------------------------------")
preds.example_df.head()
print("---------------------TARGET INFO AND PREDICTIONS---------------------------------")
preds.target_df.head()
print("---------------------WIDE DATA---------------------------------")
preds.target_df_wide.head()
wide_df = preds.target_df_wide
scores_by_label = {}
for label in preds.all_labels:
    y_true = wide_df['label.true.' + label]
    y_pred = wide_df['preds.proba.' + label] >= 0.5
    score = metrics.f1_score(y_true=y_true, y_pred=y_pred)
    scores_by_label[label] = score
scores = pd.Series(scores_by_label)
print(scores)
print("Macro average F1: %.04f" % scores.mean())
print("---------------------LONG DATA---------------------------------")
preds.target_df_long.head()
preds.target_df_long.label.unique()
from sklearn import metrics
long_df = preds.target_df_long
print("F1 long data: %.04f" % metrics.f1_score(y_true=long_df['label.true'], y_pred=(long_df['preds.proba'] >= 0.5)))