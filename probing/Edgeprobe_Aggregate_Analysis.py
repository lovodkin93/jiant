import sys, os, re, json
from importlib import reload
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import analysis
reload(analysis)

tasks = analysis.TASKS
exp_types = analysis.EXP_TYPES
palette = analysis.EXP_PALETTE

task_sort_key = analysis.task_sort_key
exp_type_sort_key = analysis.exp_type_sort_key

MAX_THRESHOLD_DISTANCE = 120
BERT_LAYERS=12

from scipy.special import logsumexp
from scipy.stats import entropy

def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

import bokeh
import bokeh.plotting as bp
bp.output_notebook()

import datetime
import socket

ID_COLS = ['run', 'task', 'split']

def agg_stratifier_group(df, stratifier, key_predicate, group_name):
    agg_map = {k:"sum" for k in df.columns if k.endswith("_count")}
    # Use this for short-circuit evaluation, so we don't call key_predicate on invalid keys
    mask = [(s == stratifier and key_predicate(key))
            for s, key in zip(df['stratifier'], df['stratum_key'])]
    sdf = df[mask].groupby(by=ID_COLS).agg(agg_map).reset_index()
    sdf['label'] = group_name
    return sdf

def load_scores_file(filename, tag=None, seed=None):
    df = pd.read_csv(filename, sep="\t", header=0)
    df.drop(['Unnamed: 0'], axis='columns', inplace=True)
    # df['task_raw'] = df['task'].copy()
    df['task'] = df['task'].map(analysis.clean_task_name)
    if not "stratifier" in df.columns:
        df["stratifier"] = None
    if not "stratum_key" in df.columns:
        df["stratum_key"] = 0
    # Custom aggregations - Span distances - for every THRESHOLD_DISTANCE between 1 and MAX_THRESHOLD_DISTANCEsplit into bigger than THRESHOLD_DISTANCE and smaller than it
    for THRESHOLD_DISTANCE in range(1,MAX_THRESHOLD_DISTANCE):
        _eg = []
        _eg.append(agg_stratifier_group(df, 'span_distance', lambda x: int(x) <= THRESHOLD_DISTANCE, f'close_coref_{THRESHOLD_DISTANCE}_thr'))
        _eg.append(agg_stratifier_group(df, 'span_distance', lambda x: int(x) > THRESHOLD_DISTANCE, f'far_coref_{THRESHOLD_DISTANCE}_thr'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)

    df.insert(0, "exp_name", df['run'].map(lambda p: os.path.basename(os.path.dirname(p.strip("/")))))
    df.insert(1, "exp_type", df['exp_name'].map(analysis.get_exp_type))
    df.insert(1, "layer_num", df['exp_name'].map(analysis.get_layer_num))
    if tag is not None:
        df.insert(0, "tag", tag)
    df.insert(1, "seed", seed)
    return df

def _format_display_col(exp_type, layer_num, tag):
    ret = exp_type
    if layer_num:
        ret += f"-{layer_num}"
    if tag:
        ret += f" ({tag})"
    return ret

def get_data(args):
    score_files = [("mix", "/cs/labs/oabend/lovodkin93/jiant_rep/jiant-ep_frozen_20190723/probing/scores.tsv")]
    dfs = []
    for tag, score_file in score_files:
        df = load_scores_file(score_file, tag=tag)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)

    df['display_col'] = list(map(_format_display_col, df.exp_type, df.layer_num, df.tag))
    analysis.score_from_confusion_matrix(df)

    def _get_final_score(row):
        return row['f1_score'], row['f1_errn95']

    df['score'], df['score_errn95'] = zip(*(_get_final_score(row) for i, row in df.iterrows()))
    return df

def calc_expected_layer(df):
    f1_scores = df[['layer_num', 'f1_score']]
    numerator = 0
    denominator = 0
    for i in range(1, BERT_LAYERS + 1):
        prev_score = f1_scores.loc[f1_scores['layer_num'] == str(i - 1)]['f1_score'].values[0]
        curr_score = f1_scores.loc[f1_scores['layer_num'] == str(i)]['f1_score'].values[0]
        delta = curr_score - prev_score
        numerator = numerator + (i * delta)
        denominator = denominator + delta
    exp_layer = numerator / denominator
    return exp_layer

def calc_best_layer_num(df):
    f1_scores = df[['layer_num', 'f1_score']]
    best_num_layer = 0
    best_score = f1_scores.loc[f1_scores['layer_num'] == '0']['f1_score'].values[0]
    for i in range(1, BERT_LAYERS + 1):
        curr_score = f1_scores.loc[f1_scores['layer_num'] == str(i)]['f1_score'].values[0]
        if (curr_score > best_score):
            best_score = curr_score
            best_num_layer = i
    return best_num_layer

def get_exp_and_best_layer_dict(df):
    exp_layer_dict = dict()
    best_layer_dict = dict()
    for THRESHOLD_DISTANCE in range(1,MAX_THRESHOLD_DISTANCE):
        curr_df = df.loc[(df['label'] == f'far_coref_{THRESHOLD_DISTANCE}_thr') & (df['split'] == 'test')]
        exp_layer_dict[THRESHOLD_DISTANCE] = calc_expected_layer(curr_df)
        best_layer_dict[THRESHOLD_DISTANCE] = calc_best_layer_num(curr_df)
    return exp_layer_dict, best_layer_dict

def main(args):
    df = get_data(args)
    exp_layer_dict, best_layer_dict = get_exp_and_best_layer_dict(df)

    # getting the samw value for the expected layer as in the article (3.67)
    #tmp = df.loc[(df['label'] == '0') & (df['split'] == 'val')]
    #f1_article = calc_expected_layer(tmp)

    # ploting the data
    # expected layer
    x_axis = list(range(1, MAX_THRESHOLD_DISTANCE))
    plt.figure(1)
    y_axis_0 = [exp_layer_dict[i] for i in x_axis]
    plt.plot(x_axis, y_axis_0, '-ok')
    plt.xlabel('Threshold distance')
    plt.ylabel('Expected Layer')
    plt.title('Expected layer as a function of minimal distance ')
    plt.show()

    # best layer number
    x_axis = list(range(1, MAX_THRESHOLD_DISTANCE))
    plt.figure(1)
    y_axis_0 = [best_layer_dict[i] for i in x_axis]
    plt.plot(x_axis, y_axis_0, '-ok')
    plt.xlabel('Threshold distance')
    plt.ylabel('Best Number of Layers')
    plt.title('Best layer number as a function of minimal distance ')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)