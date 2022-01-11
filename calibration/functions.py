# Based on the work of ethen8181
# https://github.com/ethen8181/machine-learning
#
# Direct link to notebook for the calibration
# https://nbviewer.org/github/ethen8181/machine-learning/blob/master/recsys/calibration/calibrated_reco.ipynb

import numpy as np
import matplotlib.pyplot as plt


class Item:
    def __init__(self, _id, title, attr, score=None):
        self.id = _id
        self.score = score
        self.title = title
        self.attr = attr

    def __repr__(self):
        return self.title


def create_item_mapping(df_item, item_col, title_col, attr_col):
    '''
    Creates a dictionary with all the items in df_item with
    a distribution of the attributes for each one.
    '''
    item_mapping = {}

    for row in df_item.itertuples():
        item_id = getattr(row, item_col)
        item_title = getattr(row, title_col)
        item_attr = getattr(row, attr_col)

        splitted = item_attr.split('|')
        attr_ratio = 1. / len(splitted)
        item_attr = {attr: attr_ratio for attr in splitted}

        item = Item(item_id, item_title, item_attr)
        item_mapping[item_id] = item

    return item_mapping


def compute_attr_distr(items):
    '''
    Creates the attribute distribution for a list of items.
    It loops over the items, and for each attribute of the item
    aggregates the distribution into a dictionary by attribute.

    After obtaining the total distribution for the items, these
    distributions are normalized by dividing for the amount of
    items.
    '''
    distr = {}

    for item in items:
        for attr, score in item.attr.items():
            attr_score = distr.get(attr, 0.)
            distr[attr] = attr_score + score

    for item, attr_score in distr.items():
        normed_attr_score = round(attr_score / len(items), 10)
        distr[item] = normed_attr_score

    return distr


def generate_item_candidates(df_reco, user_item, user_id, user_col,
                             item_col, score_col, item_mapping,
                             filter_already_liked_items=True):
    '''
    Because the recommendation is done outside, this function
    goes through the items recommendation generated for the user,
    saves the corresponding score given and returns a list of the
    items with the scores for calibration.
    '''

    scores = df_reco[df_reco[user_col] == user_id][[item_col, score_col]]

    liked = set()
    if filter_already_liked_items:
        liked = set(user_item[user_item[user_col]
                    == user_id][item_col].to_list())

    item_ids = set(df_reco[df_reco[user_col] == user_id][item_col].to_list())
    item_ids -= liked

    items = []
    for item_id in item_ids:
        item = item_mapping[item_id]
        item.score = scores[scores[item_col] == item_id][score_col].iloc[0]
        items.append(item)

    return items


def compute_kl_divergence(interacted_distr, reco_distr, alpha=0.01):
    '''
    Obtains the KL Divergence of the interacted distribution versus
    the recommended distribution. This is the calibration metric.
    '''
    kl_div = 0.
    for attr, score in interacted_distr.items():
        reco_score = reco_distr.get(attr, 0.)
        reco_score = (1 - alpha) * reco_score + alpha * score
        kl_div += score * np.log2(score / reco_score)

    return kl_div


def compute_utility(reco_items, interacted_distr, lmbda=0.5):
    '''
    Our objective function for computing the utility score for
    the list of recommended items

    lmbda: float, 0.0 - 1.0, default 0.5
        Lambda term controls the score and calibration tradeoff,
        the higher the lambda the higher the resulting recommendation
        will be calibrated.
    '''
    reco_distr = compute_attr_distr(reco_items)
    kl_div = compute_kl_divergence(interacted_distr, reco_distr)

    total_score = 0.0
    for item in reco_items:
        total_score += item.score

    utility = (1 - lmbda) * total_score - lmbda * kl_div
    return utility


def calib_recommend(items, inter_distr, topn, lmbda=0.5):
    '''
    Starts with an empty recommendation list
    Loop over the topn cardinality, during each iteration
    update the list with the item that maximizes the utility function
    and aggregate the kl divergence

    Returns both the calibrated recommendations and the kl divergence
    '''
    calib_reco = []
    for _ in range(topn):
        max_utility = -np.inf
        best_item = ""
        for item in items:
            if item in calib_reco:
                continue

            utility = compute_utility(calib_reco + [item], inter_distr, lmbda)
            if utility > max_utility:
                max_utility = utility
                best_item = item

        if best_item == "":
            continue
        else:
            calib_reco.append(best_item)

    return calib_reco
