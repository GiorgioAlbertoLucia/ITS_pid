'''
    Class to train and test a BDT model
'''

import os
import logging
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna
import shap
import matplotlib.pyplot as plt
import pickle
import yaml
from typing import Tuple

import sys
sys.path.append('..')
from core.dataset import DataHandler
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.src.graph_handler import GraphHandler
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.timeit import timeit
from framework.utils.root_setter import obj_setter
from framework.utils.matplotlib_to_root import save_mpl_to_root

from ROOT import TDirectory, TMultiGraph
from ROOT import kOrange, kCyan

shap.initjs()

class BDTTrainer:
    def __init__(self, cfg_bdt_file: str, output_file: TDirectory):
        self.cfg_bdt_file = cfg_bdt_file
        self.cfg_bdt = self.load_config(self.cfg_bdt_file)
        self.model = None
        self.output_file = output_file
        self.hyperparameters = None

        self.train_dataset = None
        self.train_dataset_plots = None
        self.validation_dataset = None
        self.validation_dataset_plots = None
        self.test_dataset = None
        self.test_dataset_plots = None
        self.train_id_part_map = None
        self.test_id_part_map = None
        self.feature_columns = None
        self.target_columns = None


    @staticmethod
    def load_config(file_path: str):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self, train_handler: DataHandler, validation_handler: DataHandler, test_handler: DataHandler, normalized: bool = False):
        if normalized:
            self.train_dataset = train_handler.normalized_dataset
            self.validation_dataset = validation_handler.normalized_dataset
            self.test_dataset = test_handler.normalized_dataset
        else:
            self.train_dataset = train_handler.dataset
            self.validation_dataset = validation_handler.dataset
            self.test_dataset = test_handler.dataset
        self.train_dataset_plots = train_handler.dataset
        self.validation_dataset_plots = validation_handler.dataset
        self.test_dataset_plots = test_handler.dataset
        print('BDT Trainer: Data loaded')
        self.train_id_part_map = train_handler.id_part_map ### here
        self.test_id_part_map = test_handler.id_part_map ### here
        self.feature_columns = self.cfg_bdt['features']
        self.target_columns = self.cfg_bdt['target']
    
    def save_model(self):
        pickle.dump(self.model, open(self.cfg_bdt['model_path'], 'wb'))

    def load_model(self):
        self.model = pickle.load(open(self.cfg_bdt['model_path'], 'rb'))
    
    def save_output(self):
        self.train_dataset_plots.write_parquet(self.cfg_bdt['output_file_train'])
        self.test_dataset_plots.write_parquet(self.cfg_bdt['output_file_test'])

    def evaluate(self):
        raise NotImplementedError("This method should be implemented in the subclass")

    def prepare_for_plots(self):
        self.train_hist_handlers = {'all': HistHandler.createInstance(self.train_dataset_plots)}
        self.validation_hist_handlers = {'all': HistHandler.createInstance(self.validation_dataset_plots)}
        self.test_hist_handlers = {'all': HistHandler.createInstance(self.test_dataset_plots)}
        for ipart in self.train_id_part_map.keys():
            self.train_hist_handlers[part] = HistHandler.createInstance(self.train_dataset_plots.filter(pl.col('fPartID') == ipart))
            self.validation_hist_handlers[part] = HistHandler.createInstance(self.validation_dataset_plots.filter(pl.col('fPartID') == ipart))
            self.test_hist_handlers[part] = HistHandler.createInstance(self.test_dataset_plots.filter(pl.col('fPartID') == ipart))

    def draw_beta_distribution(self):
        cfg_plot_ml = self.cfg_bdt['betaml_plot']
        axisSpecX_ml = AxisSpec.from_dict(cfg_plot_ml['axisSpecX'])
        axisSpecY_ml = AxisSpec.from_dict(cfg_plot_ml['axisSpecY'])
        cfg_plot = self.cfg_bdt['beta_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        for part, train_handler, validation_handler, test_handler in zip(self.train_hist_handlers.keys(), self.train_hist_handlers.values(), self.validation_hist_handlers.values(), self.test_hist_handlers.values()):
            axisSpecX_ml.name = cfg_plot_ml['axisSpecX']['name']+f'_train_{part}'
            hist_train_ml = train_handler.buildTH2('fP', 'fBetaML', axisSpecX_ml, axisSpecY_ml)
            axisSpecX_ml.name = cfg_plot_ml['axisSpecX']['name']+f'_validation_{part}'
            hist_validation_ml = validation_handler.buildTH2('fP', 'fBetaML', axisSpecX_ml, axisSpecY_ml)
            axisSpecX_ml.name = cfg_plot_ml['axisSpecX']['name']+f'_test_{part}'
            hist_test_ml = test_handler.buildTH2('fP', 'fBetaML', axisSpecX_ml, axisSpecY_ml)
            axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_train_{part}'
            hist_train = train_handler.buildTH2('fP', 'fBetaAbs', axisSpecX, axisSpecY)
            axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_validation_{part}'
            hist_validation = validation_handler.buildTH2('fP', 'fBetaAbs', axisSpecX, axisSpecY)
            axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_test_{part}'
            hist_test = test_handler.buildTH2('fP', 'fBetaAbs', axisSpecX, axisSpecY)

            self.output_file.cd()
            hist_train_ml.Write()
            hist_validation_ml.Write()
            hist_test_ml.Write()
            hist_train.Write()
            hist_validation.Write()
            hist_test.Write()

    def draw_part_id_distribution(self):
        cfg_plot = self.cfg_bdt['part_id_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        axisSpecX.name = cfg_plot['axisSpecX']['name']+'_train'
        hist_train = self.train_hist_handlers['all'].buildTH2('fP', 'fPartID', axisSpecX, axisSpecY)
        axisSpecX.name = cfg_plot['axisSpecX']['name']+'_validation'
        hist_validation = self.validation_hist_handlers['all'].buildTH2('fP', 'fPartID', axisSpecX, axisSpecY)
        axisSpecX.name = cfg_plot['axisSpecX']['name']+'_test'
        hist_test = self.test_hist_handlers['all'].buildTH2('fP', 'fPartID', axisSpecX, axisSpecY)

        self.output_file.cd()
        hist_train.Write()
        hist_validation.Write()
        hist_test.Write()
        
class BDTRegressorTrainer(BDTTrainer):
    @staticmethod
    def delta(pred: np.ndarray, train: xgb.DMatrix) -> Tuple[str, float]:   
        labels = train.get_label()
        pred[pred < -1] = -1 + 1e-6
        elements = np.abs((pred - labels)/labels)
        return 'delta', float(np.mean(elements))

    @staticmethod
    def delta_exp(pred: np.ndarray, train: xgb.DMatrix) -> Tuple[str, float]:   
        labels = train.get_label()
        pred[pred < -1] = -1 + 1e-6
        elements = np.abs((pred - labels)/labels)
        resolution = 0.02
        weight = 20.0
        elements = np.where(np.abs(pred - labels) > resolution, elements*np.exp(weight*elements), elements)
        return 'delta_exp', float(np.mean(elements))

    @staticmethod
    def weighted_regression_loss(pred: np.ndarray, train: xgb.DMatrix) -> Tuple[str, float]:
        labels = train.get_label()
        margin = 1.0  # Hyperparameter to tune
        
        errors = np.abs(pred - labels)
        weights = np.exp(-errors / margin)
        weighted_mae = np.mean(weights * errors)

        return 'weighted_mae', weighted_mae
    
    @timeit
    def train(self):
        print(tc.BOLD+tc.GREEN+'Training the BDT Regressor'+tc.RESET)
        print
        eval_set = [(self.train_dataset[self.feature_columns], self.train_dataset[self.target_columns]),
                    (self.validation_dataset[self.feature_columns], self.validation_dataset[self.target_columns])]

        if self.hyperparameters is None:
            self.hyperparameters = self.cfg_bdt['hyperparameters']
        if self.cfg_bdt['load_from_optuna']:
            with open(self.cfg_bdt['optuna_hyperparameters_path'], 'rb') as file:
                self.hyperparameters = pickle.load(file)
        self.model = xgb.XGBRegressor(**self.hyperparameters,
                                      #custom_metric=self.delta,
                                      custom_metric=self.delta_exp,
                                      #custom_metric=self.weighted_regression_loss,
                                      disable_default_eval_metric=1,
                                      device='cuda')
        self.model.set_params(n_jobs=20)
        self.model.fit(self.train_dataset[self.feature_columns], 
                       self.train_dataset[self.target_columns],
                       eval_set=eval_set,
                       #eval_metric=self.delta,
                       eval_metric=self.delta_exp,
                       #eval_metric=self.weighted_regression_loss,
                       **self.cfg_bdt['kwargs_fit'])
        
        self.save_model()
        return self.model

    def _optuna_objective(self, trial):
        params = {
            'verbosity': 0,
            #'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',
            'n_jobs': 20,
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),           # L2 regularization term on weights
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),             # L1 regularization term on weights
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),                # Subsample ratio of the training instances
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),  # Subsample ratio of columns when constructing each tree    
            'max_depth': trial.suggest_int('max_depth', 2, 6),                      # Maximum depth of a tree
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),           # Number of boosting rounds
            'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),                 # Boosting learning rate
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),             # Minimum loss reduction required to make a further partition on a leaf node of the tree
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)       # Minimum sum of instance weight (hessian) needed in a child
        }

        model = xgb.XGBRegressor(**params)
        model.fit(self.train_dataset[self.feature_columns], self.train_dataset[self.target_columns], eval_metric=self.delta)
        predictions = model.predict(self.validation_dataset[self.feature_columns])
        dtrain = xgb.DMatrix(self.validation_dataset[self.feature_columns], label=self.validation_dataset[self.target_columns])
        score = self.delta(predictions, dtrain)[1]
        return score

    def hyperparameter_optimization(self):

        study = optuna.create_study(direction='minimize')
        study.optimize(self._optuna_objective, n_trials=100)
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        with open(self.cfg_bdt['optuna_hyperparameters_path'], 'wb') as file:
            pickle.dump(study.best_params, file)
        
        self.hyperparameters = study.best_params

    @timeit
    def evaluate(self):
        print(tc.BOLD+tc.GREEN+'Evaluating the BDT Regressor'+tc.RESET)
        self.model.set_params(n_jobs=20)

        train_predictions = self.model.predict(self.train_dataset[self.feature_columns])
        self.train_dataset = self.train_dataset.with_columns(fBetaML=train_predictions)
        self.train_dataset = self.train_dataset.with_columns(fDeltaBeta=((pl.col('fBetaML') - pl.col('fBetaAbs')) / pl.col('fBetaAbs')))
        self.train_dataset_plots = self.train_dataset_plots.with_columns(fBetaML=train_predictions)
        self.train_dataset_plots = self.train_dataset_plots.with_columns(fDeltaBeta=((pl.col('fBetaML') - pl.col('fBetaAbs')) / pl.col('fBetaAbs')))

        validation_predictions = self.model.predict(self.validation_dataset[self.feature_columns])
        self.validation_dataset = self.validation_dataset.with_columns(fBetaML=validation_predictions)
        self.validation_dataset = self.validation_dataset.with_columns(fDeltaBeta=((pl.col('fBetaML') - pl.col('fBetaAbs')) / pl.col('fBetaAbs')))
        self.validation_dataset_plots = self.validation_dataset_plots.with_columns(fBetaML=validation_predictions)
        self.validation_dataset_plots = self.validation_dataset_plots.with_columns(fDeltaBeta=((pl.col('fBetaML') - pl.col('fBetaAbs')) / pl.col('fBetaAbs')))

        test_predictions = self.model.predict(self.test_dataset[self.feature_columns])
        self.test_dataset = self.test_dataset.with_columns(fBetaML=test_predictions)
        self.test_dataset = self.test_dataset.with_columns(fDeltaBeta=((pl.col('fBetaML') - pl.col('fBetaAbs')) / pl.col('fBetaAbs')))
        self.test_dataset_plots = self.test_dataset_plots.with_columns(fBetaML=test_predictions)
        self.test_dataset_plots = self.test_dataset_plots.with_columns(fDeltaBeta=((pl.col('fBetaML') - pl.col('fBetaAbs')) / pl.col('fBetaAbs')))

    def draw_delta_beta_distribution(self):
        cfg_plot = self.cfg_bdt['delta_beta_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        for part, train_handler, validation_handler, test_handler in zip(self.train_hist_handlers.keys(), self.train_hist_handlers.values(), self.validation_hist_handlers.values(), self.test_hist_handlers.values()):
            axisSpecX.name == cfg_plot['axisSpecX']['name']+f'_train_{part}'
            hist_train = train_handler.buildTH2('fP', 'fDeltaBeta', axisSpecX, axisSpecY)
            axisSpecX.name == cfg_plot['axisSpecX']['name']+f'_validation_{part}'
            hist_validation = validation_handler.buildTH2('fP', 'fDeltaBeta', axisSpecX, axisSpecY)
            axisSpecX.name == cfg_plot['axisSpecX']['name']+f'_test_{part}'
            hist_test = test_handler.buildTH2('fP', 'fDeltaBeta', axisSpecX, axisSpecY)

            self.output_file.cd()
            hist_train.Write(cfg_plot['axisSpecX']['name']+f'_train_{part}')
            hist_validation.Write(cfg_plot['axisSpecX']['name']+f'_validation_{part}')
            hist_test.Write(cfg_plot['axisSpecX']['name']+f'_test_{part}')

class BDTClassifierTrainer(BDTTrainer):
    
    @timeit
    def train(self):
        print(tc.BOLD+tc.GREEN+'Training the BDT Classifier'+tc.RESET)
        eval_set = [(self.train_dataset[self.feature_columns], self.train_dataset[self.target_columns]),
                    (self.validation_dataset[self.feature_columns], self.validation_dataset[self.target_columns])]

        self.model = xgb.XGBClassifier(**self.cfg_bdt['hyperparameters'])
        self.model.fit(self.train_dataset[self.feature_columns], 
                       self.train_dataset[self.target_columns],
                       eval_set=eval_set,
                       **self.cfg_bdt['kwargs_fit'])
        
        self.save_model()
        return self.model

    @timeit
    def evaluate(self, predict_classes: bool = False):
        print(tc.BOLD+tc.GREEN+'Evaluating the BDT Classifier'+tc.RESET)
        if predict_classes:
            train_predictions = self.model.predict(self.train_dataset[self.feature_columns])
            self.train_dataset = self.train_dataset.with_columns(fPartIDML=train_predictions)
            self.train_dataset_plots = self.train_dataset_plots.with_columns(fPartIDML=train_predictions)
            validation_predictions = self.model.predict(self.validation_dataset[self.feature_columns])
            self.validation_dataset = self.validation_dataset.with_columns(fPartIDML=validation_predictions)
            self.validation_dataset_plots = self.validation_dataset_plots.with_columns(fPartIDML=validation_predictions)
            test_predictions = self.model.predict(self.test_dataset[self.feature_columns])
            self.test_dataset = self.test_dataset.with_columns(fPartIDML=test_predictions)
            self.test_dataset_plots = self.test_dataset_plots.with_columns(fPartIDML=test_predictions)
        else:
            train_predictions = self.model.predict_proba(self.train_dataset[self.feature_columns])
            for ipart, part in self.train_id_part_map.items():
                self.train_dataset = self.train_dataset.with_columns(pl.Series(values=train_predictions[:, ipart], name=f'fProbML{part}'))
                self.train_dataset_plots = self.train_dataset_plots.with_columns(pl.Series(values=train_predictions[:, ipart], name=f'fProbML{part}'))
            validation_predictions = self.model.predict_proba(self.validation_dataset[self.feature_columns])
            for ipart, part in self.train_id_part_map.items():
                self.validation_dataset = self.validation_dataset.with_columns(pl.Series(values=validation_predictions[:, ipart], name=f'fProbML{part}'))
                self.validation_dataset_plots = self.validation_dataset_plots.with_columns(pl.Series(values=validation_predictions[:, ipart], name=f'fProbML{ipart}'))
            test_predictions = self.model.predict_proba(self.test_dataset[self.feature_columns])
            for ipart, part in self.test_id_part_map.items():
                self.test_dataset = self.test_dataset.with_columns(pl.Series(values=test_predictions[:, ipart], name=f'fProbML{part}'))
                self.test_dataset_plots = self.test_dataset_plots.with_columns(pl.Series(values=test_predictions[:, ipart], name=f'fProbML{part}'))

    def draw_class_scores(self):
        cfg_plot = self.cfg_bdt['class_scores_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        for predicted_class in self.train_id_part_map.values():
            for part, train_handler, validation_handler, test_handler in zip(self.train_hist_handlers.keys(), self.train_hist_handlers.values(), self.validation_hist_handlers.values(), self.test_hist_handlers.values()):
                
                axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_train_{part}_predas_{predicted_class}'
                hist_train = train_handler.buildTH2(f'fP', f'fProbML{predicted_class}', axisSpecX, axisSpecY)
                axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_validation_{part}_predas_{predicted_class}'
                hist_validation = validation_handler.buildTH2(f'fP', f'fProbML{predicted_class}', axisSpecX, axisSpecY)
                axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_test_{part}_predas_{predicted_class}'
                hist_test = test_handler.buildTH2(f'fP', f'fProbML{predicted_class}', axisSpecX, axisSpecY)

                self.output_file.cd()
                hist_train.Write()
                hist_validation.Write()
                hist_test.Write()
        
    def draw_confusion_matrix(self):
        '''
            WIP
        '''
        cfg_plot = self.cfg_bdt['confusion_matrix_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        for train_handler, validation_handler, test_handler in zip(self.train_hist_handlers.values(), self.validation_hist_handlers.values(), self.test_hist_handlers.values()):
            hist_train = train_handler.buildTH2('fPartID', 'fPartIDML ', axisSpecX, axisSpecY)
            hist_validation = validation_handler.buildTH2('fPartID', 'fPartIDML ', axisSpecX, axisSpecY)
            hist_test = test_handler.buildTH2('fPartID', 'fPartIDML ', axisSpecX, axisSpecY)

            output_file.cd()
            hist_train.Write()
            hist_validation.Write()
            hist_test.Write()   

    def draw_efficiency_purity(self, momentum_bins: np.array):
        '''
            Calculate efficiency and purity per momentum bin

            Parameters
            ----------
            input_file (str): input file
            output_file (str): output file

        '''

        roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                                 'momentum': pl.Series(values=[], dtype=pl.Float32),
                                 'threshold': pl.Series(values=[], dtype=pl.Float32),
                                 'efficiency': pl.Series(values=[], dtype=pl.Float32),
                                 'purity': pl.Series(values=[], dtype=pl.Float32)})

        test_part_id_map = {part: ipart for ipart, part in self.test_id_part_map.items()}

        for momentum_bin in range(len(momentum_bins)):

            data_bin = self.test_dataset_plots.filter((pl.col('fPAbs').is_between(momentum_bins[momentum_bin], momentum_bins[momentum_bin+1])))

            for particle in test_part_id_map.keys():

                other_particles = [part for part in test_part_id_map.keys() if part != particle]
                for th in np.linspace(0.0, 1.0, 30):

                    tp = data_bin.filter((pl.col(f'fProbML{particle}') > th) & (pl.col('fPartID') == test_part_id_map[particle])).shape[0]
                    fps = [data_bin.filter((pl.col(f'fProbML{particle}') > th) & (pl.col('fPartID') == test_part_id_map[other_particle])).shape[0] for other_particle in other_particles]
                    fn = data_bin.filter((pl.col(f'fProbML{particle}') < th) & (pl.col('fPartID') == test_part_id_map[particle])).shape[0]
                    tns = [data_bin.filter((pl.col(f'fProbML{particle}') < th) & (pl.col('fPartID') == test_part_id_map[other_particle])).shape[0] for other_particle in other_particles]


                    eff = tp / (tp + fn) if (tp + fn) > 0 else -1.
                    if (tp + fn) < 0:
                        pur = -1.
                    else:
                        den = tp / (tp + fn)
                        for fp, tn in zip(fps, tns):
                            if fp + tn > 0:
                                den = den + fp / (fp + tn) 
                        if den > 0:
                            pur = tp / (tp + fn) / den
                        else:
                            pur = -1.

                    roc_data_tmp = pl.DataFrame({'particle': pl.Series([particle], dtype=str),
                                                 'momentum': pl.Series([momentum_bins[momentum_bin]], dtype=pl.Float32),
                                                 'threshold': pl.Series([th], dtype=pl.Float32),
                                                 'efficiency': pl.Series([eff], dtype=pl.Float32),
                                                 'purity': pl.Series([pur], dtype=pl.Float32)})

                    if len(roc_data) == 0:
                        roc_data = roc_data_tmp
                    else:
                        roc_data = pl.concat([roc_data, roc_data_tmp])

                roc_data = roc_data.filter((pl.col('purity') > -0.5) & (pl.col('efficiency') > -0.5))
                graph_handler = GraphHandler(roc_data)
                roc = graph_handler.createTGraph('purity', 'efficiency')
                if roc is not None:
                    obj_setter(roc, name=f'roc_{momentum_bins[momentum_bin]}_{particle}', title=f'ROC curve for {particle} at {momentum_bins[momentum_bin]} GeV/#it{{c}}; Purity; Efficiency', marker_color=4, marker_style=20)
                    self.output_file.cd()
                    roc.Write()

                # reset the dataframe
                roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                                 'momentum': pl.Series(values=[], dtype=pl.Float32),
                                 'threshold': pl.Series(values=[], dtype=pl.Float32),
                                 'efficiency': pl.Series(values=[], dtype=pl.Float32),
                                 'purity': pl.Series(values=[], dtype=pl.Float32)})

class BDTClassifierEnsembleTrainer(BDTTrainer):
    
    def __init__(self, cfg_bdt_file: str, output_file: TDirectory, momentum_bins: np.array):
        super().__init__(cfg_bdt_file, output_file)
        self.models = []

        self.momentum_bins = momentum_bins
        self.train_datasets = []
        self.validation_datasets = []
        self.test_datasets = []
        self.train_datasets_plots = []
        self.validation_datasets_plots = []
        self.test_datasets_plots = []

        self.outdirs = []
        self.hyperparameters_list = []

    def slice_datasets(self):
        
        for ibin, _ in enumerate(self.momentum_bins):
            if ibin == len(self.momentum_bins) - 1:
                continue
            train_dataset_ibin = self.train_dataset.filter(pl.col('fPAbs').is_between(self.momentum_bins[ibin], self.momentum_bins[ibin+1]))
            validation_dataset_ibin = self.validation_dataset.filter(pl.col('fPAbs').is_between(self.momentum_bins[ibin], self.momentum_bins[ibin+1]))
            test_dataset_ibin = self.test_dataset.filter(pl.col('fPAbs').is_between(self.momentum_bins[ibin], self.momentum_bins[ibin+1]))
            train_dataset_plots_ibin = self.train_dataset_plots.filter(pl.col('fPAbs').is_between(self.momentum_bins[ibin], self.momentum_bins[ibin+1]))
            validation_dataset_plots_ibin = self.validation_dataset_plots.filter(pl.col('fPAbs').is_between(self.momentum_bins[ibin], self.momentum_bins[ibin+1]))
            test_dataset_plots_ibin = self.test_dataset_plots.filter(pl.col('fPAbs').is_between(self.momentum_bins[ibin], self.momentum_bins[ibin+1]))
            self.train_datasets.append(train_dataset_ibin)
            self.validation_datasets.append(validation_dataset_ibin)
            self.test_datasets.append(test_dataset_ibin)
            self.train_datasets_plots.append(train_dataset_ibin)
            self.validation_datasets_plots.append(validation_dataset_ibin)
            self.test_datasets_plots.append(test_dataset_ibin)

    def load_data(self, train_handler: DataHandler, validation_handler: DataHandler, test_handler: DataHandler, normalized: bool = False):
        
        super().load_data(train_handler, validation_handler, test_handler, normalized)
        self.slice_datasets()

    @staticmethod
    def focal_loss_objective(labels: np.ndarray, preds: np.ndarray):
        grad = np.zeros((len(y_true), self.num_class), dtype=float)
        hess = np.zeros((len(y_true), self.num_class), dtype=float)

        target = np.eye(self.num_class)[y_true.astype('int')]
        pred = np.reshape(y_pred, (len(y_true), self.num_class), order='F')
        # """get softmax probability"""
        softmax_p = np.exp(pred)
        softmax_p = np.multiply(softmax_p, 1/np.sum(softmax_p, axis=1)[:, np.newaxis])
        for c in range(pred.shape[1]):
            pc = softmax_p[:,c]
            pt = softmax_p[:][target == 1]
            grad[:,c][target[:,c] == 1] = (self.gamma * np.power(1-pt[target[:,c] == 1],self.gamma-1) * pt[target[:,c] == 1] * np.log(pt[target[:,c] == 1]) - np.power(1-pt[target[:,c] == 1],self.gamma) ) * (1 - pc[target[:,c] == 1])
            grad[:,c][target[:,c] == 0] = (self.gamma * np.power(1-pt[target[:,c] == 0],self.gamma-1) * pt[target[:,c] == 0] * np.log(pt[target[:,c] == 0]) - np.power(1-pt[target[:,c] == 0],self.gamma) ) * (0 - pc[target[:,c] == 0])
            hess[:,c][target[:,c] == 1] = (-4*(1-pt[target[:,c] == 1])*pt[target[:,c] == 1]*np.log(pt[target[:,c] == 1])+np.power(1-pt[target[:,c] == 1],2)*(2*np.log(pt[target[:,c] == 1])+5))*pt[target[:,c] == 1]*(1-pt[target[:,c] == 1])
            hess[:,c][target[:,c] == 0] = pt[target[:,c] == 0]*np.power(pc[target[:,c] == 0],2)*(-2*pt[target[:,c] == 0]*np.log(pt[target[:,c] == 0])+2*(1-pt[target[:,c] == 0])*np.log(pt[target[:,c] == 0]) + 4*(1-pt[target[:,c] == 0])) - pc[target[:,c] == 0]*(1-pc[target[:,c] == 0])*(1-pt[target[:,c] == 0])*(2*pt[target[:,c] == 0]*np.log(pt[target[:,c] == 0]) - (1-pt[target[:,c] == 0]))

        return grad.flatten('F'), hess.flatten('F')

    @staticmethod
    def focal_loss_metric(labels: np.ndarray, preds: np.ndarray):
        gamma = 1.5
        alpha = 0.5
        preds = preds.reshape(-1, len(np.unique(labels)))

        # Convert preds to probabilities using softmax
        preds = np.exp(preds - np.max(preds, axis=1).reshape(-1, 1))
        preds /= np.sum(preds, axis=1).reshape(-1, 1)

        # One-hot encoding for labels
        labels_one_hot = np.eye(preds.shape[1])[labels.astype(int)]

        # Compute the focal loss
        pt = preds * labels_one_hot
        pt = np.sum(pt, axis=1)
        loss = -alpha * (1 - pt) ** gamma * np.log(pt)

        # Return the mean focal loss and the name of the metric
        return np.mean(loss)

    @timeit
    def train(self):
        print(tc.BOLD+tc.GREEN+'Training the BDT Classifier'+tc.RESET)

        self.models = [None for _ in range(len(self.train_datasets))]
        for imodel, (train_dataset, validation_dataset) in enumerate(zip(self.train_datasets, self.validation_datasets)):
            print(tc.GREEN+'[INFO]: '+tc.RESET+f'Training model {imodel}')
            #if imodel != 6:
            #    continue
            eval_set = [(self.train_dataset[self.feature_columns], self.train_dataset[self.target_columns]),
                        (self.validation_dataset[self.feature_columns], self.validation_dataset[self.target_columns])]

            if len(self.hyperparameters_list) == 0:
                self.hyperparameters_list = [None for _ in range(len(self.train_datasets))]
            if self.cfg_bdt['load_from_optuna']:
                ipath = self.cfg_bdt['optuna_hyperparameters_path'].split('.')
                ipath = ipath[0] + f'_{imodel}' + '.' + ipath[1]
                self.hyperparameters_list[imodel] = pickle.load(open(ipath, 'rb'))
            if self.hyperparameters_list[imodel] is None:
                self.hyperparameters_list[imodel] = self.cfg_bdt['hyperparameters']

            model = xgb.XGBClassifier(**self.hyperparameters_list[imodel],
                                      #objective=self.focal_loss_objective,
                                      #eval_metric=self.focal_loss_metric,
                                      )
            model.set_params(n_jobs=20)
            model.fit(self.train_dataset[self.feature_columns], 
                           self.train_dataset[self.target_columns],
                           eval_set=eval_set,
                           **self.cfg_bdt['kwargs_fit'])

            self.models[imodel] = model
            self.save_model(imodel)

        return self.models

    def _optuna_objective_imodel(self, imodel:int):
        def _optuna_objective(trial):
            params = {
                'verbosity': 0,
                'objective': 'multi:softprob',
                'num_class': len(self.train_id_part_map),
                'tree_method': 'gpu_hist',
                #'eval_metric': 'auc',
                'n_jobs': 20,
                #'early_stopping_rounds': 10,
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),               # L2 regularization term on weights
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),                 # L1 regularization term on weights
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),                    # Subsample ratio of the training instances
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),      # Subsample ratio of columns when constructing each tree    
                'max_depth': trial.suggest_int('max_depth', 2, 6),                          # Maximum depth of a tree
                'n_estimators': trial.suggest_int('n_estimators', 600, 1500),               # Number of boosting rounds
                'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),                     # Boosting learning rate
                'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True), # Boosting learning rate
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),                 # Minimum loss reduction required to make a further partition on a leaf node of the tree
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)           # Minimum sum of instance weight (hessian) needed in a child
            }

            dtrain = xgb.DMatrix(self.train_datasets[imodel][self.feature_columns], label=self.train_datasets[imodel][self.target_columns])
            dval = xgb.DMatrix(self.validation_datasets[imodel][self.feature_columns], label=self.validation_datasets[imodel][self.target_columns])
            model = xgb.XGBClassifier(**params)
            model.fit(self.train_datasets[imodel][self.feature_columns], self.train_datasets[imodel][self.target_columns], 
                        eval_set=[(self.train_datasets[imodel][self.feature_columns], self.train_datasets[imodel][self.target_columns]),
                        (self.validation_datasets[imodel][self.feature_columns], self.validation_datasets[imodel][self.target_columns])],
                        verbose=False)
            preds = model.predict(self.validation_datasets[imodel][self.feature_columns])
            accuracy = accuracy_score(self.validation_datasets[imodel][self.target_columns], preds)
            return accuracy

        return _optuna_objective

    def hyperparameter_optimization(self):
        
        self.hyperparameters_list = [None for _ in range(len(self.train_datasets))]
        for imodel, _ in enumerate(self.train_datasets):
            study = optuna.create_study(direction='maximize')
            optuna_objective = self._optuna_objective_imodel(imodel)
            study.optimize(optuna_objective, n_trials=100)
            print('Number of finished trials: ', len(study.trials))
            print('Best trial:')
            trial = study.best_trial
            print('  Value: ', trial.value)
            print('  Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))

            ipath = self.cfg_bdt['optuna_hyperparameters_path'].split('.')
            ipath = ipath[0] + f'_{imodel}' + '.' + ipath[1]
            pickle.dump(study.best_params, open(ipath, 'wb'))

            self.hyperparameters_list[imodel] = study.best_params

    def save_model(self, imodel:int):
        ipath = self.cfg_bdt['model_path'].split('.')
        ipath = ipath[0] + f'_{imodel}' + '.' + ipath[1]
        pickle.dump(self.models[imodel], open(ipath, 'wb'))

    def save_models(self):
        for imodel, model in enumerate(self.models):
            ipath = self.cfg_bdt['model_path'].split('.')
            ipath = ipath[0] + f'_{imodel}' + '.' + ipath[1]
            pickle.dump(model, open(ipath, 'wb'))

    def load_model(self):
        self.models = []
        for imodel, _ in enumerate(self.train_datasets):
            ipath = self.cfg_bdt['model_path'].split('.')
            ipath = ipath[0] + f'_{imodel}' + '.' + ipath[1]
            self.models.append(pickle.load(open(ipath, 'rb')))

    def save_output(self):
        
        test_df = None
        for test_dataset in self.test_datasets:
            if test_df is None:
                test_df = test_dataset
            else:
                test_df = pl.concat([test_df, test_dataset])
        
        test_df.write_parquet(self.cfg_bdt['output_file_test'])

    @timeit
    def evaluate(self, predict_classes: bool = False):
        print(tc.BOLD+tc.GREEN+'Evaluating the BDT Classifier'+tc.RESET)
        for model in self.models:
            model.set_params(n_jobs=20)
        if predict_classes:
            self.predict_classes()
        else:
            self.predict_scores()

    def predict_classes(self):
        for model, train_dataset, validation_dataset, test_dataset in zip(self.models, self.train_datasets, self.validation_datasets, self.test_datasets):
            train_predictions = model.predict(train_dataset[self.feature_columns])
            train_dataset = train_dataset.with_columns(fPartIDML=train_predictions)
            validation_predictions = model.predict(validation_dataset[self.feature_columns])
            validation_dataset = validation_dataset.with_columns(fPartIDML=validation_predictions)
            test_predictions = model.predict(test_dataset[self.feature_columns])
            test_dataset = test_dataset.with_columns(fPartIDML=test_predictions)
    
    def predict_scores(self):
        for imomentum, model in enumerate(self.models):
            train_predictions = model.predict_proba(self.train_datasets[imomentum][self.feature_columns])
            for ipart, part in self.train_id_part_map.items():
                self.train_datasets[imomentum] = self.train_datasets[imomentum].with_columns(pl.Series(values=train_predictions[:, ipart], name=f'fProbML{part}'))
                self.train_datasets_plots[imomentum] = self.train_datasets_plots[imomentum].with_columns(pl.Series(values=train_predictions[:, ipart], name=f'fProbML{part}'))
            validation_predictions = model.predict_proba(self.validation_datasets[imomentum][self.feature_columns])
            for ipart, part in self.train_id_part_map.items():
                self.validation_datasets[imomentum] = self.validation_datasets[imomentum].with_columns(pl.Series(values=validation_predictions[:, ipart], name=f'fProbML{part}'))
                self.validation_datasets_plots[imomentum] = self.validation_datasets_plots[imomentum].with_columns(pl.Series(values=validation_predictions[:, ipart], name=f'fProbML{part}'))
            test_predictions = model.predict_proba(self.test_datasets[imomentum][self.feature_columns])
            for ipart, part in self.test_id_part_map.items():
                self.test_datasets[imomentum] = self.test_datasets[imomentum].with_columns(pl.Series(values=test_predictions[:, ipart], name=f'fProbML{part}'))
                self.test_datasets_plots[imomentum] = self.test_datasets_plots[imomentum].with_columns(pl.Series(values=test_predictions[:, ipart], name=f'fProbML{part}'))

    def prepare_for_plots(self):
        self.train_hist_handlers = [{'all': HistHandler.createInstance(train_dataset_plots)} for train_dataset_plots in self.train_datasets_plots]
        self.validation_hist_handlers = [{'all': HistHandler.createInstance(validation_dataset_plots)} for validation_dataset_plots in self.validation_datasets_plots]
        self.test_hist_handlers = [{'all': HistHandler.createInstance(test_dataset_plots)} for test_dataset_plots in self.test_datasets_plots]
        for imomentum, _ in enumerate(self.train_hist_handlers):
            self.outdirs.append(self.output_file.mkdir(f'bin_{imomentum}'))
            for ipart, part in self.train_id_part_map.items():
                self.train_hist_handlers[imomentum][part] = HistHandler.createInstance(self.train_datasets_plots[imomentum].filter(pl.col('fPartID') == ipart))
                self.validation_hist_handlers[imomentum][part] = HistHandler.createInstance(self.validation_datasets_plots[imomentum].filter(pl.col('fPartID') == ipart))
                self.test_hist_handlers[imomentum][part] = HistHandler.createInstance(self.test_datasets_plots[imomentum].filter(pl.col('fPartID') == ipart))   
                
    def draw_class_scores(self):
        cfg_plot = self.cfg_bdt['class_scores_plot_th1']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        
        for imomentum, _ in enumerate(self.train_hist_handlers):
            for predicted_class in self.train_id_part_map.values():
                for part in self.train_hist_handlers[imomentum].keys():

                    axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_train_{part}_predas_{predicted_class}'
                    hist_train = self.train_hist_handlers[imomentum][part].buildTH1(f'fProbML{predicted_class}', axisSpecX)
                    axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_validation_{part}_predas_{predicted_class}'
                    hist_validation = self.validation_hist_handlers[imomentum][part].buildTH1(f'fProbML{predicted_class}', axisSpecX)
                    axisSpecX.name = cfg_plot['axisSpecX']['name']+f'_test_{part}_predas_{predicted_class}'
                    hist_test = self.test_hist_handlers[imomentum][part].buildTH1(f'fProbML{predicted_class}', axisSpecX)

                    self.outdirs[imomentum].cd()
                    hist_train.Write()
                    hist_validation.Write()
                    hist_test.Write()

    def draw_feature_importance(self):

        for imomentum, model in enumerate(self.models):
            explainer = shap.TreeExplainer(model)
            x_sampled = self.train_datasets[imomentum][self.feature_columns].sample(n=10000)
            shap_values = explainer.shap_values(x_sampled)

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, x_sampled, plot_type='bar', show=False)
            print('------------------------\n\n\t\tSHAP values\n\n------------------------\n\n')

            for ipart, part_shap_values in enumerate(shap_values):
                print(part_shap_values.shape)
                print(type(part_shap_values))

                vals = np.mean(np.abs(part_shap_values), axis=0)
                feature_importance = {'column_name': self.feature_columns, 'feature_importance_vals': vals}
                feature_importance = pl.DataFrame(feature_importance)
                feature_importance.sort(by=['feature_importance_vals'], descending=True)

                ax.set_yticklabels(feature_importance['column_name'])
                ax.set_xlabel('|SHAP value|')
                save_mpl_to_root(fig, self.outdirs[imomentum], f'SHAP_variable_importance_class{ipart}')

            plt.close('all')

    def draw_part_id_distribution(self):
        cfg_plot = self.cfg_bdt['part_id_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        for imomentum, _ in enumerate(self.train_hist_handlers):
            axisSpecX.name = cfg_plot['axisSpecX']['name']+'_train'
            hist_train = self.train_hist_handlers[imomentum]['all'].buildTH2('fPAbs', 'fPartID', axisSpecX, axisSpecY)
            axisSpecX.name = cfg_plot['axisSpecX']['name']+'_validation'
            hist_validation = self.validation_hist_handlers[imomentum]['all'].buildTH2('fPAbs', 'fPartID', axisSpecX, axisSpecY)
            axisSpecX.name = cfg_plot['axisSpecX']['name']+'_test'
            hist_test = self.test_hist_handlers[imomentum]['all'].buildTH2('fPAbs', 'fPartID', axisSpecX, axisSpecY)

            self.output_file.cd()
            hist_train.Write()
            hist_validation.Write()
            hist_test.Write()
        
    def draw_confusion_matrix(self):
        '''
            WIP
        '''
        cfg_plot = self.cfg_bdt['confusion_matrix_plot']
        axisSpecX = AxisSpec.from_dict(cfg_plot['axisSpecX'])
        axisSpecY = AxisSpec.from_dict(cfg_plot['axisSpecY'])

        for train_handler, validation_handler, test_handler in zip(self.train_hist_handlers.values(), self.validation_hist_handlers.values(), self.test_hist_handlers.values()):
            hist_train = train_handler.buildTH2('fPartID', 'fPartIDML ', axisSpecX, axisSpecY)
            hist_validation = validation_handler.buildTH2('fPartID', 'fPartIDML ', axisSpecX, axisSpecY)
            hist_test = test_handler.buildTH2('fPartID', 'fPartIDML ', axisSpecX, axisSpecY)

            self.output_file.cd()
            hist_train.Write()
            hist_validation.Write()
            hist_test.Write()   

    def draw_efficiency_purity(self):
        '''
            Calculate efficiency and purity per momentum bin

            Parameters
            ----------
            input_file (str): input file
            output_file (str): output file
        '''

        roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                                 'momentum': pl.Series(values=[], dtype=pl.Float32),
                                 'threshold': pl.Series(values=[], dtype=pl.Float32),
                                 'efficiency': pl.Series(values=[], dtype=pl.Float32),
                                 'purity': pl.Series(values=[], dtype=pl.Float32)})

        test_part_id_map = {part: ipart for ipart, part in self.test_id_part_map.items()}

        out_dir = self.output_file.mkdir('roc_curves')

        for imomentum, test_dataset in enumerate(self.test_datasets):

            for particle in test_part_id_map.keys():

                other_particles = [part for part in test_part_id_map.keys() if part != particle]
                for th in np.linspace(0.0, 1.0, 30):

                    tp = test_dataset.filter((pl.col(f'fProbML{particle}') > th) & (pl.col('fPartID') == test_part_id_map[particle])).shape[0]
                    fps = [test_dataset.filter((pl.col(f'fProbML{particle}') > th) & (pl.col('fPartID') == test_part_id_map[other_particle])).shape[0] for other_particle in other_particles]
                    fn = test_dataset.filter((pl.col(f'fProbML{particle}') < th) & (pl.col('fPartID') == test_part_id_map[particle])).shape[0]
                    tns = [test_dataset.filter((pl.col(f'fProbML{particle}') < th) & (pl.col('fPartID') == test_part_id_map[other_particle])).shape[0] for other_particle in other_particles]


                    eff = tp / (tp + fn) if (tp + fn) > 0 else -1.
                    if (tp + fn) < 0:
                        pur = -1.
                    else:
                        den = tp / (tp + fn)
                        for fp, tn in zip(fps, tns):
                            if fp + tn > 0:
                                den = den + fp / (fp + tn) 
                        if den > 0:
                            pur = tp / (tp + fn) / den
                        else:
                            pur = -1.

                    roc_data_tmp = pl.DataFrame({'particle': pl.Series([particle], dtype=str),
                                                 'momentum': pl.Series([self.momentum_bins[imomentum]], dtype=pl.Float32),
                                                 'threshold': pl.Series([th], dtype=pl.Float32),
                                                 'efficiency': pl.Series([eff], dtype=pl.Float32),
                                                 'purity': pl.Series([pur], dtype=pl.Float32)})

                    if len(roc_data) == 0:
                        roc_data = roc_data_tmp
                    else:
                        roc_data = pl.concat([roc_data, roc_data_tmp])

                roc_data = roc_data.filter((pl.col('purity') > -0.5) & (pl.col('efficiency') > -0.5))
                graph_handler = GraphHandler(roc_data)
                roc = graph_handler.createTGraph('purity', 'efficiency')
                if roc is not None:
                    obj_setter(roc, name=f'roc_{self.momentum_bins[imomentum]}_{particle}', title=f'ROC curve for {particle} at {self.momentum_bins[imomentum]} GeV/#it{{c}}; Purity; Efficiency', marker_color=4, marker_style=20)
                    out_dir.cd()
                    roc.Write()

                # reset the dataframe
                roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                                 'momentum': pl.Series(values=[], dtype=pl.Float32),
                                 'threshold': pl.Series(values=[], dtype=pl.Float32),
                                 'efficiency': pl.Series(values=[], dtype=pl.Float32),
                                 'purity': pl.Series(values=[], dtype=pl.Float32)})
