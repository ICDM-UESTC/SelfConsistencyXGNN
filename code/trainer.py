import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_sparse import transpose
from torch_geometric.utils import is_undirected, sort_edge_index
from collections import defaultdict
import numpy as np
from datetime import datetime
import random
from itertools import combinations
import copy

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score


class BaseTrainer(object):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        self.method_name = None
        self.checkpoints_path = None

        self.model = model
        self.explainer = explainer
        self.dataloader = dataloader
        self.cfg = cfg

        self.device = device

        self.best_valid_score = 0.0
        self.lowest_valid_loss = float('inf')
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(os.path.join(save_dir, 'checkpoints'))

    def set_method_name(self, method_name):
        self.method_name = method_name
        self.checkpoints_path = os.path.join(self.save_dir, 'checkpoints',
                                             f'{self.method_name}_{self.cfg.dataset_name}.pth')

    def _train_batch(self, data):
        raise NotImplementedError

    def _valid_batch(self, data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def valid(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    @staticmethod
    def process_data(data, use_edge_attr):
        if not use_edge_attr:
            data.edge_attr = None
        if data.get('edge_label', None) is None:
            data.edge_label = torch.zeros(data.edge_index.shape[1])
        return data

    @torch.inference_mode()
    def calculate_shd_auc_fid_acc(self, method_name, ensemble_numbers=[0]):
        assert self.cfg.multi_label is False
        assert len(ensemble_numbers) % 2 == 0

        ori_data = []
        for data in self.dataloader['test_by_sample']:
            ori_data.append(copy.deepcopy(data))

        for model_index in ensemble_numbers:
            new_checkpoints_path = f'{self.checkpoints_path[:-4]}_{model_index}.pth'
            self.load_model(new_checkpoints_path)
            self.model.eval()
            self.explainer.eval()

            for data_index, data in enumerate(self.dataloader['test_by_sample']):
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)

                emb = self.model.get_emb(x=data.x, edge_index=data.edge_index,
                                         edge_attr=data.edge_attr, batch=data.batch)
                att = self.concrete_sample(self.explainer(emb, data.edge_index, data.batch), training=False)
                edge_att = self.process_att_to_edge_att(data, att)  # Gs
                ori_data[data_index][f'edge_att_{model_index}'] = edge_att.squeeze()

                '''for fid-'''
                minus_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index,
                                               edge_attr=data.edge_attr, batch=data.batch, edge_atten=edge_att)
                minus_att = self.concrete_sample(self.explainer(minus_emb, data.edge_index, data.batch), training=False)
                minus_edge_att_final = self.process_att_to_edge_att(data, minus_att)

                '''for fid+'''
                edge_att_c = 1 - edge_att  # Gc\s
                plus_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index,
                                              edge_attr=data.edge_attr, batch=data.batch, edge_atten=edge_att_c)
                plus_att = self.concrete_sample(self.explainer(plus_emb, data.edge_index, data.batch), training=False)
                plus_edge_att_final = self.process_att_to_edge_att(data, plus_att)

                if 'cal' in method_name:
                    s_edge_att = 1 - edge_att
                    c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch,
                                               edge_atten=edge_att)
                    s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch,
                                               edge_atten=s_edge_att)
                    csi_emb = c_emb + s_emb
                    logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)  # [1, 1]

                    '''for fid-'''
                    # minus_s_edge_att_final = 1 - minus_edge_att_final
                    minus_s_edge_att_final = (edge_att - minus_edge_att_final).clamp(min=0)
                    minus_c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                     batch=data.batch,
                                                     edge_atten=minus_edge_att_final)
                    minus_s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                     batch=data.batch,
                                                     edge_atten=minus_s_edge_att_final)
                    minus_csi_emb = minus_s_emb + minus_c_emb
                    logits_minus = self.model.get_pred_from_csi_emb(emb=minus_csi_emb, batch=data.batch)

                    '''for fid+'''
                    # plus_s_edge_att_final = 1 - plus_edge_att_final
                    plus_s_edge_att_final = (edge_att_c - plus_edge_att_final).clamp(min=0)
                    plus_c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                    batch=data.batch,
                                                    edge_atten=plus_edge_att_final)
                    plus_s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                    batch=data.batch,
                                                    edge_atten=plus_s_edge_att_final)
                    plus_csi_emb = plus_s_emb + plus_c_emb
                    logits_plus = self.model.get_pred_from_csi_emb(emb=plus_csi_emb, batch=data.batch)
                else:
                    logits = self.model(x=data.x, edge_index=data.edge_index,
                                        edge_attr=data.edge_attr, batch=data.batch, edge_atten=edge_att)
                    '''for fid-'''
                    logits_minus = self.model(x=data.x, edge_index=data.edge_index,
                                              edge_attr=data.edge_attr, batch=data.batch,
                                              edge_atten=minus_edge_att_final)
                    '''for fid+'''
                    logits_plus = self.model(x=data.x, edge_index=data.edge_index,
                                             edge_attr=data.edge_attr, batch=data.batch, edge_atten=plus_edge_att_final)

                ori_data[data_index][f'y_hat_{model_index}'] = logits.sigmoid()  # binary classification
                ori_data[data_index][f'y_hat_minus_{model_index}'] = logits_minus.sigmoid()  # binary classification
                ori_data[data_index][f'y_hat_plus_{model_index}'] = logits_plus.sigmoid()  # binary classification

        # ====== SPA ======
        n = len(ensemble_numbers)
        all_model_spa_list = []
        for model_index in range(n):
            model_spa_list = []
            for data in ori_data:
                model_spa_list.append(torch.mean(data[f'edge_att_{model_index}']).cpu().data)
            all_model_spa_list.append(np.mean(model_spa_list))
        spa_mean = np.mean(all_model_spa_list)
        spa_std = np.std(all_model_spa_list)

        '''calculate fid-'''
        fid_scores_minus = []
        for model_index in range(len(ensemble_numbers)):
            fid_model_scores_minus = []
            for data in ori_data:
                logits = data[f'y_hat_{model_index}'].squeeze()
                pred_ori = (logits > 0.5).int()
                logits_minus = data[f'y_hat_minus_{model_index}'].squeeze()
                pred_minus = (logits_minus > 0.5).int()
                score = torch.abs(pred_ori - pred_minus).item()
                fid_model_scores_minus.append(score)
            fid_scores_minus.append(np.mean(fid_model_scores_minus))
        fid_minus_mean, fid_minus_std = np.mean(fid_scores_minus), np.std(fid_scores_minus)

        '''calculate fid+'''
        fid_scores_plus = []
        for model_index in range(len(ensemble_numbers)):
            fid_model_scores_plus = []
            for data in ori_data:
                logits = data[f'y_hat_{model_index}'].squeeze()
                pred_ori = (logits > 0.5).int()
                logits_plus = data[f'y_hat_plus_{model_index}'].squeeze()
                pred_plus = (logits_plus > 0.5).int()
                score = torch.abs(pred_ori - pred_plus).item()
                fid_model_scores_plus.append(score)
            fid_scores_plus.append(np.mean(fid_model_scores_plus))
        fid_plus_mean, fid_plus_std = np.mean(fid_scores_plus), np.std(fid_scores_plus)

        '''calculate fid'''
        arr_plus = np.asarray(fid_scores_plus, dtype=float)
        arr_minus = np.asarray(fid_scores_minus, dtype=float)

        fid_per_model = (arr_plus + (1.0 - arr_minus)) / 2

        fid_mean = float(fid_per_model.mean())
        fid_std = float(fid_per_model.std())
        # ------------------------------------------------

        n = len(ensemble_numbers)

        all_scores_dict = defaultdict(list)
        for data in ori_data:
            edge_atts = [data[f'edge_att_{i}'] for i in range(n)]
            combinations_list = list(combinations(range(n), 1))
            for pair in combinations(combinations_list, 2):
                if set(pair[0]).isdisjoint(pair[1]):
                    edge_atts_first = torch.stack([edge_atts[i] for i in pair[0]]).mean(dim=0)
                    edge_atts_second = torch.stack([edge_atts[i] for i in pair[1]]).mean(dim=0)
                    mae_distance = torch.abs(edge_atts_first - edge_atts_second).mean().item()
                    all_scores_dict[f'{pair[0]}_{pair[1]}'].append(mae_distance)
        shd_ori_scores = [np.mean(v) for v in all_scores_dict.values()]
        shd_ori_mean, shd_ori_std = np.mean(shd_ori_scores), np.std(shd_ori_scores)

        # after EE：
        all_scores_dict = defaultdict(list)
        for data in ori_data:
            edge_atts = [data[f'edge_att_{i}'] for i in range(n)]
            combinations_list = list(combinations(range(n), int(n / 2)))
            for pair in combinations(combinations_list, 2):
                if set(pair[0]).isdisjoint(pair[1]):
                    edge_atts_first = torch.stack([edge_atts[i] for i in pair[0]]).mean(dim=0)
                    edge_atts_second = torch.stack([edge_atts[i] for i in pair[1]]).mean(dim=0)
                    mae_distance = torch.abs(edge_atts_first - edge_atts_second).mean().item()
                    all_scores_dict[f'{pair[0]}_{pair[1]}'].append(mae_distance)
        shd_ee_scores = [np.mean(v) for v in all_scores_dict.values()]
        shd_ee_mean, shd_ee_std = np.mean(shd_ee_scores), np.std(shd_ee_scores)

        # ====== AUC ======
        # before EE：
        model_auc_list = []
        for model_index in range(n):
            edge_att_list, exp_label_list = [], []
            for data in ori_data:
                edge_att_list.append(data[f'edge_att_{model_index}'])
                exp_label_list.append(data.edge_label.data)
            model_auc = roc_auc_score(torch.cat(exp_label_list).cpu(), torch.cat(edge_att_list).cpu())
            model_auc_list.append(model_auc)
        auc_ori_mean, auc_ori_std = np.mean(model_auc_list), np.std(model_auc_list)

        # after EE：
        model_auc_list = []
        combinations_list = list(combinations(range(n), int(n / 2)))
        for pair in combinations_list:
            edge_att_list, exp_label_list = [], []
            for data in ori_data:
                edge_atts = [data[f'edge_att_{i}'] for i in range(n)]
                edge_att = torch.stack([edge_atts[i] for i in pair]).mean(dim=0)
                edge_att_list.append(edge_att)
                exp_label_list.append(data.edge_label.data)
            model_auc = roc_auc_score(torch.cat(exp_label_list).cpu(), torch.cat(edge_att_list).cpu())
            model_auc_list.append(model_auc)
        auc_ee_mean, auc_ee_std = np.mean(model_auc_list), np.std(model_auc_list)
        # auc_ori_mean, auc_ori_std, auc_ee_std, auc_ee_mean = 0.0, 0.0, 0.0, 0.0  # for no-gt-dataset (e.g., aids)

        # ====== ACC ======
        # before EE：
        model_acc_list = []
        for model_index in range(n):
            y_hat_list, y_list = [], []
            for data in ori_data:
                y_hat = data[f'y_hat_{model_index}'].squeeze().item()
                y = data.y.squeeze().item()
                y_hat_list.append(1 if y_hat > 0.5 else 0)
                y_list.append(1 if y > 0.5 else 0)
            acc = accuracy_score(y_list, y_hat_list)
            model_acc_list.append(acc)
        acc_ori_mean, acc_ori_std = np.mean(model_acc_list), np.std(model_acc_list)

        # after EE：
        model_acc_list = []
        combinations_list = list(combinations(range(n), int(n / 2)))
        for pair in combinations_list:
            y_hat_list, y_list = [], []
            for data in ori_data:
                y_hat = torch.stack([data[f'y_hat_{i}'] for i in pair]).mean(dim=0).squeeze().item()
                y = data.y.squeeze().item()
                y_hat_list.append(1 if y_hat > 0.5 else 0)
                y_list.append(1 if y > 0.5 else 0)
            acc = accuracy_score(y_list, y_hat_list)
            model_acc_list.append(acc)
        acc_ee_mean, acc_ee_std = np.mean(model_acc_list), np.std(model_acc_list)

        print("====================================================================================")
        print("before ensemble:")
        print(f"shd: {shd_ori_mean:.4f}±{shd_ori_std:.4f}\n"
              f"auc: {auc_ori_mean:.4f}±{auc_ori_std:.4f}\n"
              f"acc: {acc_ori_mean:.4f}±{acc_ori_std:.4f}\n"
              f"fid-: {fid_minus_mean:.4f}±{fid_minus_std:.4f}\n"
              f"fid+: {fid_plus_mean:.4f}±{fid_plus_std:.4f}\n"
              f"fid: {fid_mean:.4f}±{fid_std:.4f}\n"
              f"spa:{spa_mean:.4f}±{spa_std:.4f}")
        print("------------------------------------------------------------------------------------")
        print("after ensemble (5 models):")
        print(f"shd: {shd_ee_mean:.4f}±{shd_ee_std:.4f}\n"
              f"auc: {auc_ee_mean:.4f}±{auc_ee_std:.4f}\n"
              f"acc: {acc_ee_mean:.4f}±{acc_ee_std:.4f}")
        print("====================================================================================")


class ATTTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, logits, labels):
        ce_loss = self.criterion(logits, labels)
        loss = ce_loss * self.ce_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class ATTSCTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef
        self.sc_loss_coef = cfg.sc_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, logits, labels, att_old, att_new):
        ce_loss = self.criterion(logits, labels)
        distill_loss = torch.mean(torch.abs(att_old - att_new))
        loss = ce_loss * self.ce_loss_coef + distill_loss * self.sc_loss_coef
        return loss, distill_loss * self.sc_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(logits=clf_logits, labels=data.y, att_old=edge_att_old,
                                           att_new=edge_att_new)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(logits=clf_logits, labels=data.y, att_old=edge_att_old,
                                           att_new=edge_att_new)

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)

        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()

        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class SIZETrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.sparsity_mask_coef = cfg.sparsity_mask_coef
        self.sparsity_ent_coef = cfg.sparsity_ent_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    def sparsity(self, edge_mask, eps=1e-6):
        sparsity = 0.
        # sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.c * edge_mask.mean()
        # ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        # sparsity += self.sparsity_ent_coef * ent.mean()
        return sparsity

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_sparsity_c(self, current_epoch):
        c = self.sparsity_mask_coef * (current_epoch + 1) / 10
        if c > self.sparsity_mask_coef:
            c = self.sparsity_mask_coef
        return c

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):
        ce_loss = self.criterion(logits, labels)
        reg_loss = self.sparsity(att)
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.c = self.get_sparsity_c(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (self.c == self.sparsity_mask_coef) and (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class SIZESCTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.sparsity_mask_coef = cfg.sparsity_mask_coef
        self.sparsity_ent_coef = cfg.sparsity_ent_coef
        self.sc_loss_coef = cfg.sc_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    def sparsity(self, edge_mask, eps=1e-6):
        sparsity = 0.
        # sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.c * edge_mask.mean()
        # ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        # sparsity += self.sparsity_ent_coef * ent.mean()
        return sparsity

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_sparsity_c(self, current_epoch):
        c = self.sparsity_mask_coef
        return c

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels, att_old, att_new):
        ce_loss = self.criterion(logits, labels)
        reg_loss = self.sparsity(att)
        distill_loss = torch.mean(torch.abs(att_old - att_new))
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef + distill_loss * self.sc_loss_coef
        return loss, distill_loss * self.sc_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_old=edge_att_old,
                                           att_new=edge_att_new)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_old=edge_att_old,
                                           att_new=edge_att_new)

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            self.c = self.get_sparsity_c(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class GSATTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):
        ce_loss = self.criterion(logits, labels)
        reg_loss = (att * torch.log(att / self.r + 1e-6)
                    + (1 - att) * torch.log((1 - att) / (1 - self.r + 1e-6) + 1e-6)).mean()
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (self.r == self.final_r) and (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class GSATSCTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.sc_loss_coef = cfg.sc_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels, att_old, att_new):
        ce_loss = self.criterion(logits, labels)
        reg_loss = (att * torch.log(att / self.r + 1e-6)
                    + (1 - att) * torch.log((1 - att) / (1 - self.r + 1e-6) + 1e-6)).mean()
        distill_loss = torch.mean(torch.abs(att_old - att_new))
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef + distill_loss * self.sc_loss_coef
        return loss, distill_loss * self.sc_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_old=edge_att_old,
                                           att_new=edge_att_new)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_old=edge_att_old,
                                           att_new=edge_att_new)

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")

            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, c_logits, s_logits, csi_logits, labels):
        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / 0.5 + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - 0.5 + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)
        csi_loss = self.criterion(csi_logits, labels)
        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef

        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALCRTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef
        self.cr_loss_coef = cfg.cr_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    def get_sparsity_c(self, current_epoch):
        c = self.cr_loss_coef * (current_epoch + 1) / 10
        if c > self.cr_loss_coef:
            c = self.cr_loss_coef
        return c

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, c_logits, s_logits, csi_logits, labels):
        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / 0.5 + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - 0.5 + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)
        csi_loss = self.criterion(csi_logits, labels)
        reg_loss = att.mean()
        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef + reg_loss * self.c

        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(att, c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                             labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(att, c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                             labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            all_att, all_exp = [], []

            self.c = self.get_sparsity_c(e)

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALSCTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef
        self.sc_loss_coef = cfg.sc_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, c_logits, s_logits, csi_logits, labels, att_old, att_new):
        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / 0.5 + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - 0.5 + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)
        csi_loss = self.criterion(csi_logits, labels)
        distill_loss = torch.mean(torch.abs(att_old - att_new))
        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef + distill_loss * self.sc_loss_coef

        return loss, distill_loss * self.sc_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=c_edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                                           labels=data.y, att_old=edge_att_old, att_new=edge_att_new)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=c_edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                                           labels=data.y, att_old=edge_att_old, att_new=edge_att_new)

        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []
        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALCRSCTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef
        self.cr_loss_coef = cfg.cr_loss_coef
        self.sc_loss_coef = cfg.sc_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, c_logits, s_logits, csi_logits, labels, att_old, att_new):
        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / 0.5 + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - 0.5 + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)
        csi_loss = self.criterion(csi_logits, labels)
        distill_loss = torch.mean(torch.abs(att_old - att_new))
        reg_loss = att.mean()

        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef + reg_loss * self.cr_loss_coef + distill_loss * self.sc_loss_coef

        return loss, distill_loss * self.sc_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=c_edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(att, c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                                           labels=data.y, att_old=edge_att_old, att_new=edge_att_new)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        'self consistency: old & new'
        emb_no_grad = emb.detach()
        att_log_logit_old = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_old = self.concrete_sample(att_log_logit_old, training=False)
        edge_att_old = self.process_att_to_edge_att(data, att_old)

        emb_new = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     edge_atten=c_edge_att)
        emb_new_no_grad = emb_new.detach()
        att_new_log_logit = self.explainer(emb_new_no_grad, data.edge_index, data.batch)
        att_new = self.concrete_sample(att_new_log_logit, training=False)
        edge_att_new = self.process_att_to_edge_att(data, att_new)

        loss, distill_loss = self.__loss__(att, c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                                           labels=data.y, att_old=edge_att_old, att_new=edge_att_new)

        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []
        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


def get_trainer(method_name, model, explainer, dataloader, cfg, device, save_dir):
    trainer = None
    if method_name == 'att':
        trainer = ATTTrainer(model=model,
                             explainer=explainer,
                             dataloader=dataloader,
                             cfg=cfg,
                             device=device,
                             save_dir=save_dir)
    elif method_name == 'att_sc':
        trainer = ATTSCTrainer(model=model,
                               explainer=explainer,
                               dataloader=dataloader,
                               cfg=cfg,
                               device=device,
                               save_dir=save_dir)
    elif method_name == 'size':
        trainer = SIZETrainer(model=model,
                              explainer=explainer,
                              dataloader=dataloader,
                              cfg=cfg,
                              device=device,
                              save_dir=save_dir)
    elif method_name == 'size_sc':
        trainer = SIZESCTrainer(model=model,
                                explainer=explainer,
                                dataloader=dataloader,
                                cfg=cfg,
                                device=device,
                                save_dir=save_dir)
    elif method_name == 'gsat':
        trainer = GSATTrainer(model=model,
                              explainer=explainer,
                              dataloader=dataloader,
                              cfg=cfg,
                              device=device,
                              save_dir=save_dir)
    elif method_name == 'gsat_sc':
        trainer = GSATSCTrainer(model=model,
                                explainer=explainer,
                                dataloader=dataloader,
                                cfg=cfg,
                                device=device,
                                save_dir=save_dir)
    elif method_name == 'cal':
        trainer = CALTrainer(model=model,
                             explainer=explainer,
                             dataloader=dataloader,
                             cfg=cfg,
                             device=device,
                             save_dir=save_dir)
    elif method_name == 'cal_cr':
        trainer = CALCRTrainer(model=model,
                               explainer=explainer,
                               dataloader=dataloader,
                               cfg=cfg,
                               device=device,
                               save_dir=save_dir)
    elif method_name == 'cal_sc':
        trainer = CALSCTrainer(model=model,
                               explainer=explainer,
                               dataloader=dataloader,
                               cfg=cfg,
                               device=device,
                               save_dir=save_dir)
    elif method_name == 'cal_cr_sc':
        trainer = CALCRSCTrainer(model=model,
                                 explainer=explainer,
                                 dataloader=dataloader,
                                 cfg=cfg,
                                 device=device,
                                 save_dir=save_dir)
    trainer.set_method_name(method_name)
    return trainer
