
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
