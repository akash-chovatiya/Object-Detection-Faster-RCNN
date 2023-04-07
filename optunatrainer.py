# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:15:30 2023

@author: Chovatiya
"""

import os
import sys
import inspect
import yaml

config_file = open('config.yaml', 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

maindir = os.path.dirname(currentdir)
sys.path.insert(0, maindir)

subdir = os.path.dirname(maindir)
sys.path.insert(0, subdir)

subsubdir = os.path.dirname(subdir)
sys.path.insert(0, subsubdir)

#%%
from lib.utilities import Averager, \
                            SaveBestModel, show_tranformed_image, \
                            save_model, save_loss_plot, save_data, \
                                show_tranformed_image_v2
from lib.timers import tic, toc
from lib.All_Dataset_Model import get_model
from lib.metrics import meanaverageprecision
from lib.trainer import trainer
from lib.validator import validater
import os
import torch, time
import optuna
from datetime import date
import joblib
from optuna.trial import TrialState

#%%
batch_size = config['dl']['batch_size']
shuffle = config['dl']['shuffle']
num_workers = config['dl']['num_workers']
VISUALIZE_TRANSFORMED_IMAGES = config['dl']['visualize_img']
MAP_PER_CLASS = config['dl']['MAP_PER_CLASS']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = config['h-params']['NUM_EPOCHS']
time_stamp = date.today().strftime('%d-%m-20%y')

class optuna_optimizer():
    def __init__(self, OUT_DIR, all_classes, training_dataloader,
                 testing_dataloader, validation_dataloader, train_data,
                 test_data, val_data, n_trials, save_study: str = None):
        self.training_dataloader = training_dataloader
        self.OUT_DIR = OUT_DIR
        self.all_classes = all_classes
        self.testing_dataloader = testing_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.save_study = save_study
        self.n_trials = n_trials
    
    def train_model(self, trial, params_opt):
        trial_num = trial.number
        model = get_model(
                              num_classes = len(self.all_classes), 
                              pretrained_model = config['h-params']['pretrained'], 
                              model_name = config['h-params']['model']  
                          )
        
        # initialize the model and move to the computation device
        model = model.to(DEVICE)
        
        # get the model parameters
        params = [p for p in model.parameters() if p.requires_grad]
        
        # define the optimizer
        # optimizer = torch.optim.SGD(params, lr=params_opt['lr'], 
        #                             momentum=params_opt['momentum'], 
        #                             weight_decay=params_opt['weight_decay'])
        
        optimizer = torch.optim.Adagrad(params, lr=params_opt['lr'], 
                                    lr_decay=params_opt['lr_decay'], 
                                    weight_decay=params_opt['weight_decay'])
        
        # define the learning rate scheduler
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
        # initialize the Averager class
        train_loss_hist, val_loss_hist, map_hist = Averager(), Averager(), Averager()
        
        # initialize the starting iteration count
        train_itr, val_itr, map_itr = 1, 1, 1
        
        # initialize the storage variables
        train_loss_list, val_loss_list, map_list = [], [], []
        
        # whether to show transformed images from data loader or not
        if VISUALIZE_TRANSFORMED_IMAGES is True:        
            show_tranformed_image_v2(self.training_dataloader, self.all_classes)
        
        # initialize SaveBestModel class
        save_best_model = SaveBestModel()
        
        #epoch_train_loss, epoch_val_loss, epoch_map = [], [], []
        
        # start the training epochs
        for epoch in range(NUM_EPOCHS):
            print(f'\nEPOCH {epoch+1} of {NUM_EPOCHS}')
    
            # reset the training and validation loss histories for the current epoch
            train_loss_hist.reset()
            val_loss_hist.reset()
            
            # start timer and carry out training and validation
            tic("Epoch {}".format(epoch))
            
            train_itr, train_loss_list, train_loss_hist, optimizer = trainer(
                train_itr, train_loss_list, train_loss_hist, optimizer, self.training_dataloader, model, DEVICE
                )
            
            val_itr, val_loss_list, val_loss_hist = validater(
                val_itr, val_loss_list, val_loss_hist, self.validation_dataloader, model, DEVICE
                )       
            
            map_itr, map_list, map_hist, map_results, map_global = meanaverageprecision(
                map_itr, map_list, map_hist, self.validation_dataloader, model, MAP_PER_CLASS, DEVICE
                )
            
            trial.report(map_global, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # map_results = metric.compute()
            # map_global = map_results["map"].cpu().detach().numpy()              
            if MAP_PER_CLASS:
                map_per_class = map_results["map_per_class"].numpy().tolist()
                if not map_per_class == -1:
                    print ( "Epoch #{} map per class: {}".format( epoch+1, list( round( i,3 ) for i in map_per_class ) ) )
            
            print ( "Epoch #{} train loss: {}".format( epoch+1, round( train_loss_hist.value, 3 ) ) )
            print ( "Epoch #{} validation loss: {}".format( epoch+1, round( val_loss_hist.value, 3 ) ) )
            print ( "Epoch #{} map global: {}".format( epoch+1, round( map_hist.value, 3 ) ) )
            
            # epoch_train_loss.append(train_loss_hist.value)
            # epoch_train_loss.append(train_loss_hist.value)
            # epoch_train_loss.append(train_loss_hist.value)
            
            # print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
            # print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")  
            # print(f"Epoch #{epoch+1} map global: {map_hist.value:.3f}") 
            # print(f"Epoch #{epoch+1} map per class: {map_per_class:.3f}")  
            toc("Epoch {}".format(epoch))
            
            # save the best model till now if we have the least loss in the...
            # ... current epoch
            save_model_filename = "{}_{}".format(config['h-params']['model'], config['h-params']['optimizer'])
            save_best_model(
                                OUT_DIR = os.path.join(self.OUT_DIR, f"trial_{trial_num}"), 
                                current_valid_loss = val_loss_hist.value, 
                                epoch = epoch, 
                                model = model, 
                                optimizer = optimizer, 
                                DATA_NAME =  "{}_trial_{}".format(save_model_filename,trial.number)
                            )
            # save the current epoch model
            save_model(
                            OUT_DIR = os.path.join(self.OUT_DIR, f"trial_{trial_num}"), 
                            epoch = epoch, 
                            model = model, 
                            optimizer = optimizer, 
                            DATA_NAME = "{}_trial_{}".format(save_model_filename,trial.number)
                        )
            
            # save loss plot
            save_loss_plot(
                                OUT_DIR = os.path.join(self.OUT_DIR, f"trial_{trial_num}"),
                                train_loss = train_loss_list, 
                                val_loss = val_loss_list, 
                                map_value = map_list,
                                DATA_NAME = "{}_epoch_{}_trial_{}".format(save_model_filename,epoch,trial.number)
                            )
            
            # save_loss_plot(
            #                     OUT_DIR = os.path.join(self.OUT_DIR, f"trial_{trial_num}"),
            #                     train_loss = train_loss_hist, 
            #                     val_loss = val_loss_hist, 
            #                     map_value = map_hist, 
            #                     DATA_NAME = "{}_epoch_{}_trial_{}_individual".format(save_model_filename,epoch,trial.number)
            #                 )
            
            save_data(OUT_DIR = os.path.join(self.OUT_DIR, f"trial_{trial_num}"), 
                      train_loss = train_loss_list,
                      val_loss = val_loss_list,
                      map_value = map_list,
                      DATA_NAME = "{}_trial_{}".format(save_model_filename,trial.number))
            
            # save_data(OUT_DIR = os.path.join(self.OUT_DIR, f"trial_{trial_num}"), 
            #           train_loss = train_loss_hist,
            #           val_loss = val_loss_hist,
            #           map_value = map_hist,
            #           DATA_NAME = "{}_epoch_{}_trial_{}".format(save_model_filename, epoch, trial.number))
            
            # sleep for 5 seconds after each epoch
            time.sleep(5)
        
        #scheduler.step()
        #print(scheduler.get_last_lr())
        
        return val_loss_hist.value
        
    def objective(self, trial):
        # Hyperparameters we want to optimize
        # if trial.number >= self.n_trials-1:
        #     self.study.stop()
        # lr_min = config['h-params']['learning_rate'][0]
        # lr_max = config['h-params']['learning_rate'][1]
        # momentum_min = config['h-params']['momentum'][0]
        # momentum_max = config['h-params']['momentum'][1]
        # weight_decay_min = config['h-params']['weight_decay'][0]
        # weight_decay_max = config['h-params']['weight_decay'][1]
        params_opt = {
            "lr":trial.suggest_loguniform('lr', 1e-4, 1e-2),
            # "momentum":trial.suggest_float('momentum', 0.8, 0.95),
            "lr_decay":trial.suggest_float('lr_decay', 0, 0),
            "weight_decay":trial.suggest_loguniform('weight_decay', 1e-4, 1e-3)
            }
        val_loss = self.train_model(trial, params_opt)
        if trial.number > 0:
            self.create_summary()
        return val_loss
    
    def optuna_study(self):
        study_name = 'RCNN_study'
        if self.save_study is not None:
            self.study = joblib.load(self.save_study)
        else:
            self.study = optuna.create_study(study_name=study_name,
                                             direction='minimize')
        self.study.optimize(self.objective, self.n_trials)
        
    def create_summary(self):
        joblib.dump(self.study, self.OUT_DIR + "/optuna_study.pkl")
        df = self.study.trials_dataframe(attrs=('number','value','params','state'))
        df.to_excel(self.OUT_DIR+'/optuna_study.xlsx', sheet_name = time_stamp)
        print('Best_trial: ')
        print(self.study.best_params)
        print(self.study.best_value)
        # print(self.study.best_trial)
        # study.trials

        # trial_ = study.best_trials
        # print(trial_.value)
        # print(trial_.params)

        # trial_best_map = max(self.study.best_trials, key=lambda t: t.values[0])
        # print("Trial with best map: ")
        # print(f"\tnumber: {trial_best_map.number}")
        # print(f"\tparams: {trial_best_map.params}")
        # print(f"\tvalues: {trial_best_map.values}")

        # trial_best_val_loss = min(self.study.best_trials, key=lambda t: t.values[1])
        # print("Trial with best val loss: ")
        # print(f"\tnumber: {trial_best_val_loss.number}")
        # print(f"\tparams: {trial_best_val_loss.params}")
        # print(f"\tvalues: {trial_best_val_loss.values}")

        # trial_best_train_loss = min(self.study.best_trials, key=lambda t: t.values[2])
        # print("Trial with best train loss: ")
        # print(f"\tnumber: {trial_best_train_loss.number}")
        # print(f"\tparams: {trial_best_train_loss.params}")
        # print(f"\tvalues: {trial_best_train_loss.values}")

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))