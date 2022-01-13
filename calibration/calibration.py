
import numpy as np
import pandas as pd

from tqdm import tqdm

from .functions import *

user_col = 'userId'
item_col = 'itemId'
value_col = 'rating'
time_col = 'timestamp'

title_col = 'title'
attr_col = 'genres'

score_col = 'score'

attr_names = [item_col, title_col, attr_col]
train_names = [user_col, item_col, value_col, time_col]
reco_names = [user_col, item_col, score_col]

pd_params = {
    'sep': {
        'csv': ',',
        'tsv': '\t'
    }
}


class Calibration:
    def __init__(self):
        self.model_name = None

        self.df_attr = None
        self.df_train = None
        self.df_reco = None

        self.item_mapping = None

        self.top_k = None
        self.lmbda = None

        self.inter_distr = None
        self.recom_distr = None
        self.calib_distr = None

        self.worst_case = None

        self.average_table = None
        self.single_table = None

    def set_config(self, model_name, attr_file, train_file, reco_file, top_k, lmbda):
        self.model_name = model_name

        self.df_attr = pd.read_csv(
            attr_file[1], header=None,
            sep=pd_params['sep'][attr_file[0]],
            names=attr_names
        )
        self.df_train = pd.read_csv(
            train_file[1], header=None,
            sep=pd_params['sep'][train_file[0]],
            names=train_names
        )
        self.df_reco = pd.read_csv(
            reco_file[1], header=None,
            sep=pd_params['sep'][reco_file[0]],
            names=reco_names
        )

        self.item_mapping = create_item_mapping(
            self.df_attr, item_col, title_col, attr_col)

        self.top_k = top_k
        self.lmbda = lmbda

        self.inter_distr = {}
        self.recom_distr = {}
        self.calib_distr = {}

        self.worst_case = []

        self.average_table = None
        self.single_table = None

    def prepare(self):
        top_k = self.top_k

        user_list = np.array(self.df_reco[user_col].unique())
        kl_div_list = np.array([])

        for user_id in tqdm(user_list, desc='Calculating distributions...'):
            inter_ids = self.df_train[self.df_train[user_col]
                                      == user_id][item_col].to_list()
            inter_items = [self.item_mapping[index] for index in inter_ids]
            inter_distr = compute_attr_distr(inter_items)

            recom_ids = self.df_reco[self.df_reco[user_col]
                                     == user_id][item_col][:top_k].to_list()
            recom_items = [self.item_mapping[index] for index in recom_ids]
            recom_distr = compute_attr_distr(recom_items)

            kl_div_list = np.append(kl_div_list, compute_kl_divergence(
                inter_distr, recom_distr))

            self.inter_distr[user_id] = inter_distr
            self.recom_distr[user_id] = recom_distr

        ten_percent = int(len(kl_div_list) / 10)
        sorted_list = np.argsort(-np.array(kl_div_list))[:ten_percent]
        self.worst_case = user_list[sorted_list]

    def calibrate(self):
        top_k = self.top_k
        lmbda = self.lmbda

        for user_id in tqdm(self.worst_case, desc='Calibrating...'):
            items = generate_item_candidates(self.df_reco, self.df_train, user_id,
                                             user_col, item_col, score_col, self.item_mapping)
            calib_items = calib_recommend(
                items, self.inter_distr[user_id], top_k, lmbda)
            calib_distr = compute_attr_distr(calib_items)

            self.calib_distr[user_id] = calib_distr

    def generate_tables(self):

        distr_dict = {
            'inter_distr': self.inter_distr,
            'recom_distr': self.recom_distr,
            'calib_distr': self.calib_distr
        }

        average_table = {
            'inter_distr': {},
            'recom_distr': {},
            'calib_distr': {},
            'neg_pre_delta': {},
            'pos_pre_delta': {},
            'neg_post_delta': {},
            'pos_post_delta': {},
        }

        for user_id in self.worst_case:
            user_vals = {}

            # Loops over the 3 distributions (interacted, recommended and calibrated)
            for kind in set(distr_dict):
                kind_distr = distr_dict[kind][user_id]

                user_vals[kind] = {}

                # Loops over the attributes of the user for this distribution
                for attr in set(kind_distr):
                    accu_val = average_table[kind].get(attr, 0)
                    average_table[kind][attr] = accu_val + kind_distr[attr]

                    user_vals[kind][attr] = kind_distr[attr]

            # Gets the delta BEFORE the calibration separated into positive and negative
            for attr in set(user_vals['inter_distr']) | set(user_vals['recom_distr']):
                delta = user_vals['recom_distr'].get(
                    attr, 0) - user_vals['inter_distr'].get(attr, 0)
                if delta < 0:
                    accu_delta = average_table['neg_pre_delta'].get(attr, 0)
                    average_table['neg_pre_delta'][attr] = accu_delta + delta
                else:
                    accu_delta = average_table['pos_pre_delta'].get(attr, 0)
                    average_table['pos_pre_delta'][attr] = accu_delta + delta

            # Gets the delta AFTER the calibration separated into positive and negative
            for attr in set(user_vals['inter_distr']) | set(user_vals['calib_distr']):
                delta = user_vals['calib_distr'].get(
                    attr, 0) - user_vals['inter_distr'].get(attr, 0)
                if delta < 0:
                    accu_delta = average_table['neg_post_delta'].get(attr, 0)
                    average_table['neg_post_delta'][attr] = accu_delta + delta
                else:
                    accu_delta = average_table['pos_post_delta'].get(attr, 0)
                    average_table['pos_post_delta'][attr] = accu_delta + delta

        self.average_table = average_table / len(self.worst_case)

        user_id = np.random.choice(self.worst_case)

        user_inter_distr = distr_dict['inter_distr'][user_id]
        user_inter_distr = {x: y for x,
                            y in user_inter_distr.items() if y != 0}

        user_recom_distr = distr_dict['recom_distr'][user_id]
        user_recom_distr = {x: y for x,
                            y in user_recom_distr.items() if y != 0}

        user_calib_distr = distr_dict['calib_distr'][user_id]
        user_calib_distr = {x: y for x,
                            y in user_calib_distr.items() if y != 0}

        single_table = {
            'inter_distr': user_inter_distr,
            'recom_distr': user_recom_distr,
            'calib_distr': user_calib_distr,
            'pre_delta': {key: user_recom_distr.get(key, 0) - user_inter_distr.get(key, 0) for key in user_inter_distr.keys() | user_recom_distr.keys()},
            'post_delta': {key: user_calib_distr.get(key, 0) - user_inter_distr.get(key, 0) for key in user_inter_distr.keys() | user_calib_distr.keys()},
            'recom_delta': {key: user_calib_distr.get(key, 0) - user_recom_distr.get(key, 0) for key in user_calib_distr.keys() | user_recom_distr.keys()}
        }

        self.single_table = single_table

    def save_to_csv(self, save_path):
        pd.DataFrame(self.inter_distr).to_csv(save_path + 'inter_distr.csv')
        pd.DataFrame(self.recom_distr).to_csv(
            save_path + 'recom_distr.csv')
        pd.DataFrame(self.calib_distr).to_csv(
            save_path + 'calib_distr.csv')
        pd.DataFrame(self.single_table).to_csv(
            save_path + 'single_table.csv')
        pd.DataFrame(self.average_table).to_csv(
            save_path + 'average_table.csv')
