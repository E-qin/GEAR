# QWC edit 2023/07/02 GMT+8 22:00
from collections import defaultdict
import json
import os
import pdb
import sys
import time
import numpy as np
import psutil
import torch
import tqdm
from base import BaseTrainer
from utils import inf_loop, OutputMetricTracker, OutputMetricTracker_LeCaRDv2

from accelerate import Accelerator

# YWJ for fp16
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


def get_time(start_time):
    now = time.time()
    s_time = now - start_time
    if s_time < 60:
        return "{:.2f} s".format(s_time)
    m_time = s_time / 60
    if m_time < 60:
        return "{:.2f} min".format(m_time)
    h_time = m_time / 60
    return "{:.2f} h".format(h_time)
    


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, test_data_loader,extra_train_loader=None, lr_scheduler=None, len_epoch=None, 
                 ex_id='',data_provider=None):
        
        if config["accelerate"] == True:
            gradient_accumulation_steps = config["accumulate_steps"]
            self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
            model, optimizer, train_data_loader,test_data_loader, lr_scheduler = self.accelerator.prepare(
                model, optimizer, train_data_loader,test_data_loader, lr_scheduler
            )
        
            if extra_train_loader!=None:
                extra_train_loader = self.accelerator.prepare(extra_train_loader)
        
            
        
            
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.ex_id = ex_id
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.data_provider = data_provider
        self.dataset_name = data_provider.dataset_name
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.extra_train_loader = extra_train_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = 20 if "log_step" not in config else config["log_step"]

        # assert atomicId_labels_dict is not None and atomicId_docid_dict is not None
        self.train_metrics = OutputMetricTracker(self.config["ELAM_output"], metric_ftns, 
                                                 data_provider, 
                                                 writer=self.writer,  mode='train', ex_id=self.ex_id)
        self.test_metrics =  OutputMetricTracker(self.config["ELAM_output"], metric_ftns, 
                                                 data_provider, 
                                                 writer=self.writer, mode='test', ex_id=self.ex_id,)
        
        self.best_key_metrics = {k:0.0 for k in self.config["key_metrics"]}
        
        self.start_time = time.time()


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        if "eval_time" in self.config and self.config["eval_time"]:
            self.eval_time_test_epoch(epoch)
            return {}
        
        
        
        
        self.model.train()
        loss_per_epoch = 0.
        batch_num = 0
        self.train_metrics.start_a_epoch(epoch)
        print("\nTRAIN EPOCH {}".format(epoch))

        # Limit the number of times each q trains ranking loss in each epoch
        # Reset at the beginning of each epoch
        # Dict usually. None if it dont need to be constrained(when seperate_train) 
        # limit_times = self.data_provider.rlttpefeq() #{atomicId_str: int}
        
        if "+qg" in self.config[self.dataset_name + "_data_prepare"] and self.config[self.dataset_name + "_data_prepare"]["+qg"]:
            print("extra_train_loader:")
            for batch_idx, (data_batch_dict, target_1_batch, atomic_id_batch) in enumerate(self.extra_train_loader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    bath_pred,loss = self.model(data_batch_dict, extra=True) # need to check
                    if self.config["model_dbg"]:
                        print("data_batch_dict:",data_batch_dict)
                        print("target_1_batch:",target_1_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)

                    
                    if batch_idx % self.log_step == 0:
                        self.logger.debug("{}/{} Loss: {:.6f} | TIME: {}".format(batch_idx,len(self.extra_train_loader), loss.item(),
                                                            get_time(self.start_time)))
                    loss_per_epoch += loss.item()
                    self.accelerator.backward(loss)
                    # pdb.set_trace()
                    self.optimizer.step()
        
        
        print("train:")
        for batch_idx, (data_batch_dict, target_1_batch, target_list_batch, atomic_id_batch) in enumerate(self.train_data_loader):
            # print( atomic_id_batch)
            if self.config["accelerate"] and self.config["accelerator.accumulate"]:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    bath_pred, loss = self.model(data_batch_dict,extra=self.config["LM_loss_only"])
                    if self.config["model_dbg"]:
                        print("data_batch_dict:",data_batch_dict)
                        print("target_1_batch:",target_1_batch)
                        print("target_list_batch:",target_list_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)
        
                    batch_num += 1
                    # loss = loss.mean()  
                    loss_per_epoch += loss.item()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    # if batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(self.train_data_loader):
                    #     self.optimizer.step()
                    #     self.optimizer.zero_grad()
                    # pdb.set_trace()
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.train_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
        
                    if batch_idx % self.log_step == 0:
                        self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss.item(),
                            get_time(self.start_time)))
                        # print("MEM: {}%".format(psutil.virtual_memory().percent)) 
        
                    if batch_idx == self.len_epoch:
                        break
            elif self.config["accelerate"] and not self.config["accelerator.accumulate"]:
                # data_batch_dict = self._dict_to_device(data_batch_dict, self.device)
                self.optimizer.zero_grad()
                bath_pred, loss = self.model(data_batch_dict,extra=self.config["LM_loss_only"])
                if self.config["model_dbg"]:
                        print("target_1_batch:",target_1_batch)
                        print("target_list_batch:",target_list_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)
        
                batch_num += 1
                # loss = loss.mean()  # ？
                loss_per_epoch += loss.item()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss.item(),
                            get_time(self.start_time)))
                    # print("MEM: {}%".format(psutil.virtual_memory().percent)) 
        
                if batch_idx == self.len_epoch:
                    break
            else: # not self.config["accelerate"] and not self.config["accelerator.accumulate"]:
                ############ YWfp16
                data_batch_dict = self._dict_to_device(data_batch_dict, self.device)
                self.optimizer.zero_grad()
                accumulation_steps = 5
                
                bath_pred, loss = self.model(data_batch_dict,extra=self.config["LM_loss_only"])
                if self.config["model_dbg"]:
                        print("target_1_batch:",target_1_batch)
                        print("target_list_batch:",target_list_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)
                batch_num += 1
                loss = loss.mean()  #
                
                loss_per_epoch += loss.item()
                loss.backward()
                
                if (batch_idx+1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # ######## YWJ fp16
                # scaler.scale(loss).backward()
                # clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # scaler.step(self.optimizer) 
                # scaler.update() 

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
        
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss.item(),
                            get_time(self.start_time)))
                    # print("MEM: {}%".format(psutil.virtual_memory().percent)) 
        
                if batch_idx == self.len_epoch:
                    break
        metrics_result = self.train_metrics.end_a_epoch(epoch)
        print('|||train result:{}'.format(metrics_result))
        print('|||train epoch {} TIME: {}'.format( epoch, get_time(self.start_time)))
        print("|||MEM: {}%".format(psutil.virtual_memory().percent)) 

        save_or_not_dict = self._test_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        for sub_name,save_or_not in save_or_not_dict.items():
            if save_or_not:
                self._save_best_checkpoint(epoch, sub_name)
            
        # return metrics_result
        return {}
    
    def eval_time_test_epoch(self, epoch):

        self.model.eval()
        print("\nTEST EPOCH", epoch)
        with torch.no_grad():
            for batch_idx, (data_batch, target_1_batch, target_list_batch, atomic_id_batch) in enumerate(self.test_data_loader):
                _,_ = self.model(data_batch,extra=self.config["LM_loss_only"],eval_time=True)

        print('|||test epoch {} TIME: {}'.format( epoch, get_time(self.start_time)))
        return 
    
    
    def _test_epoch(self, epoch):

        self.model.eval()
        print("\nTEST EPOCH", epoch)
        self.test_metrics.start_a_epoch(epoch)
        with torch.no_grad():
            for batch_idx, (data_batch, target_1_batch, target_list_batch, atomic_id_batch) in enumerate(self.test_data_loader):
                # data_batch = self._dict_to_device(data_batch, self.device)
                # print(data_batch)
                self.optimizer.zero_grad()

                bath_pred, loss = self.model(data_batch,extra=self.config["LM_loss_only"])

                if self.config["model_dbg"]:
                    print("target_1_batch:",target_1_batch)
                    print("target_list_batch:",target_list_batch)
                    print("batch_pred:",bath_pred) # list batch_size * beam_size
                    print("atomic_id_batch:",atomic_id_batch)
                    
                self.test_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
                
        
        metrics_result = self.test_metrics.end_a_epoch(epoch)
        
        print('|||test result:{}'.format(metrics_result))
        print('|||test epoch {} TIME: {}'.format( epoch, get_time(self.start_time)))
        print("|||MEM: {}%".format(psutil.virtual_memory().percent)) 
        save_dict={}
        for metric in self.best_key_metrics:
            if metric not in metrics_result:
                continue
            else:
                if metrics_result[metric] > self.best_key_metrics[metric]:
                    save_dict[metric] = True
                    self.best_key_metrics[metric] = metrics_result[metric]
                else:
                    save_dict[metric] = False
        
        return save_dict


    def _save_best_checkpoint(self, epoch, sub_name=''):
        """
        Saving bes checkpoint

        Parameters:
            epoch (int): current epoch
            sub_name (str): sub name of the checkpoint file, metric name
        
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if os.path.exists('./best_models/'+self.dataset_name+'/'+self.ex_id) == False:
            os.makedirs('./best_models/'+self.dataset_name+'/'+self.ex_id)
        
        filename =  'best_models/{}/{}/{}_best_epoch.pth'.format(self.dataset_name, self.ex_id, sub_name)

        torch.save(state, filename)
        self.logger.info("Saving best checkpoint: {} ...".format(filename))

        
        

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _dict_to_device(self, input_dict, device):
        for key, value in input_dict.items():
            input_dict[key] = torch.tensor(input_dict[key]).to(device)
        # print('input_dict:{}'.format(input_dict))
        return input_dict
    
    
    ###########################################################################################################
    
  
  
  
  
  
  

class MultiCrimeTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, test_data_loader,extra_train_loader, lr_scheduler=None, len_epoch=None, 
                 ex_id='',data_provider=None,join_train_loader=None,join=False,join_epoch=-1):
        
        self.already_joined = False
        self.join = join
        if join:
            assert join_train_loader is not None
            assert type(join_epoch) == int
            if config["accelerate"] == True:
                gradient_accumulation_steps = config["accumulate_steps"]
                self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
                model, optimizer, train_data_loader, test_data_loader, extra_train_loader, join_train_loader, lr_scheduler = self.accelerator.prepare(
                    model, optimizer, train_data_loader, test_data_loader, extra_train_loader, join_train_loader, lr_scheduler
                )
            self.join_train_loader = join_train_loader 
            self.join_epoch = join_epoch
            self.join = join
            print("len(join_train_loader)",len(join_train_loader))
            print("len(train_data_loader)",len(train_data_loader))
        
        if config["accelerate"] == True:
            gradient_accumulation_steps = config["accumulate_steps"]
            self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
            model, optimizer, train_data_loader,test_data_loader,extra_train_loader, lr_scheduler = self.accelerator.prepare(
                model, optimizer, train_data_loader,test_data_loader,extra_train_loader, lr_scheduler
            )
            
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.ex_id = ex_id
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.data_provider = data_provider
        self.dataset_name = data_provider.dataset_name
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.extra_train_loader = extra_train_loader
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.log_step = 100 if "log_step" not in config else config["log_step"]

        # assert atomicId_labels_dict is not None and atomicId_docid_dict is not None
        self.train_metrics = OutputMetricTracker_LeCaRDv2(self.config[self.dataset_name+"_output"], metric_ftns, 
                                                 data_provider, 
                                                 writer=self.writer,  mode='train', ex_id=self.ex_id)
        self.test_metrics =  OutputMetricTracker_LeCaRDv2(self.config[self.dataset_name+"_output"], metric_ftns, 
                                                 data_provider, 
                                                 writer=self.writer, mode='test', ex_id=self.ex_id,)

        self.best_key_metrics = {k:0.0 for k in self.config["key_metrics"]}
        
        self.start_time = time.time()


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if "eval_time" in self.config and self.config["eval_time"]:
            self.eval_time_test_epoch(epoch)
            return {}
        
        
        if self.join:
            if self.already_joined == False and epoch >= self.join_epoch:
                self.already_joined = True
                self.train_data_loader = self.join_train_loader
                self.len_epoch = len(self.train_data_loader)
                self._save_best_checkpoint(epoch, sub_name='join_time') 
                print("-----------join_train_loader is used-------------")
                
        
        
        self.model.train()
        loss_per_epoch = 0.
        batch_num = 0
        self.train_metrics.start_a_epoch(epoch)
        print("\nTRAIN EPOCH {}".format(epoch))
        print("For index:")

        if "indexing" in self.config and self.config["indexing"]:
            indexing_times = 1 if "indexing_times" not in self.config else self.config["indexing_times"]
            for i in range(indexing_times):
                for batch_idx, (data_batch_dict, target_1_batch, atomic_id_batch) in enumerate(self.extra_train_loader):
                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        bath_pred,loss = self.model(data_batch_dict, extra=True) # need to check
                        # loss = self.model.extra_forward(data_batch_dict) # need to check

                        if self.config["model_dbg"]:
                            print("data_batch_dict:",data_batch_dict)
                            print("target_1_batch:",target_1_batch)
                            print("batch_pred:",bath_pred)
                            print("atomic_id_batch:",atomic_id_batch)

                        
                        if batch_idx % self.log_step == 0:
                            self.logger.debug("{}/{} Loss: {:.6f} | TIME: {}".format(batch_idx,len(self.extra_train_loader), loss.item(),
                                                                get_time(self.start_time)))
                            # self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            #     epoch,
                            #     self._progress(batch_idx),
                            #     loss.item(),
                            #     get_time(self.start_time)))
                        # loss = loss.mean()  
                        loss_per_epoch += loss.item()
                        self.accelerator.backward(loss)
                        # pdb.set_trace()
                        self.optimizer.step()
        
        
        print("For retrieval:")
        for batch_idx, (data_batch_dict, target_1_batch, target_list_batch, atomic_id_batch) in enumerate(self.train_data_loader):
            # print( atomic_id_batch)
            if self.config["accelerate"] and self.config["accelerator.accumulate"]:
                with self.accelerator.accumulate(self.model):
                    # data_batch = self._dict_to_device(data_batch, self.device) 
                    self.optimizer.zero_grad()
                    bath_pred, loss = self.model(data_batch_dict,extra=self.config["LM_loss_only"] and not self.already_joined)
                    if self.config["model_dbg"]:
                        print("data_batch_dict:",data_batch_dict)
                        print("target_1_batch:",target_1_batch)
                        print("target_list_batch:",target_list_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)
        
                    batch_num += 1
                    # loss = loss.mean()  
                    loss_per_epoch += loss.item()
                    self.accelerator.backward(loss)
                    # pdb.set_trace()
                    self.optimizer.step()
                    # pdb.set_trace()
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.train_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
        
                    if batch_idx % self.log_step == 0:
                        self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss.item(),
                            get_time(self.start_time)))
                        # print("MEM: {}%".format(psutil.virtual_memory().percent)) 
        
                    if batch_idx == self.len_epoch:
                        break
                
                # if self.device != torch.device('cpu'):
                #     with torch.cuda.device(self.device):
                #         torch.cuda.empty_cache()
            
            elif self.config["accelerate"] and not self.config["accelerator.accumulate"]:
                # data_batch_dict = self._dict_to_device(data_batch_dict, self.device)
                self.optimizer.zero_grad()
                bath_pred, loss = self.model(data_batch_dict,extra=self.config["LM_loss_only"] and not self.already_joined)
                if self.config["model_dbg"]:
                        print("target_1_batch:",target_1_batch)
                        print("target_list_batch:",target_list_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)
        
                batch_num += 1
                # loss = loss.mean() 
                loss_per_epoch += loss.item()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss.item(),
                            get_time(self.start_time)))
                    # print("MEM: {}%".format(psutil.virtual_memory().percent)) 
        
                if batch_idx == self.len_epoch:
                    break
            else: # not self.config["accelerate"] and not self.config["accelerator.accumulate"]:
                data_batch_dict = self._dict_to_device(data_batch_dict, self.device)
                self.optimizer.zero_grad()
                bath_pred, loss = self.model(data_batch_dict,extra=self.config["LM_loss_only"] and not self.already_joined)
                if self.config["model_dbg"]:
                        print("target_1_batch:",target_1_batch)
                        print("target_list_batch:",target_list_batch)
                        print("batch_pred:",bath_pred)
                        print("atomic_id_batch:",atomic_id_batch)
        
                batch_num += 1
                loss = loss.mean()  
                loss_per_epoch += loss.item()
                loss.backward()
                self.optimizer.step()
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
        
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} | TIME: {}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss.item(),
                            get_time(self.start_time)))
                    # print("MEM: {}%".format(psutil.virtual_memory().percent)) 
        
                if batch_idx == self.len_epoch:
                    break
        metrics_result = self.train_metrics.end_a_epoch(epoch)
        print('|||train result:{}'.format(metrics_result))
        print('|||train epoch {} TIME {}'.format( epoch, get_time(self.start_time)))
        print("|||MEM: {}%".format(psutil.virtual_memory().percent)) 


        save_or_not_dict = self._test_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        for sub_name,save_or_not in save_or_not_dict.items():
            if save_or_not:
                self._save_best_checkpoint(epoch, sub_name+ "_be" if self.already_joined else sub_name+"_af" )
            
        # return metrics_result
        return {}
    
    
    
    def eval_time_test_epoch(self, epoch):

        self.model.eval()
        print("\nTEST EPOCH", epoch)
        with torch.no_grad():
            for batch_idx, (data_batch, target_1_batch, target_list_batch, atomic_id_batch) in enumerate(self.test_data_loader):
                _,_ = self.model(data_batch,extra=self.config["LM_loss_only"],eval_time=True)

        print('|||test epoch {} TIME: {}'.format( epoch, get_time(self.start_time)))
        return 
    
    
    
    def _test_epoch(self, epoch):

        self.model.eval()
        print("\nTEST EPOCH", epoch)
        self.test_metrics.start_a_epoch(epoch)
        with torch.no_grad():
            for batch_idx, (data_batch, target_1_batch, target_list_batch, atomic_id_batch) in enumerate(self.test_data_loader):
                # data_batch = self._dict_to_device(data_batch, self.device)
                # print(data_batch)
                self.optimizer.zero_grad()

                bath_pred, loss = self.model(data_batch,extra=self.config["LM_loss_only"])

                if self.config["model_dbg"]:
                    print("target_1_batch:",target_1_batch)
                    print("target_list_batch:",target_list_batch)
                    print("batch_pred:",bath_pred) 
                    print("atomic_id_batch:",atomic_id_batch)
                    
                self.test_metrics.collect_a_batch(epoch,
                                                  target_batch=target_list_batch,
                                                  pred_batch=bath_pred,
                                                  atomic_id_batch=atomic_id_batch,
                                                  loss_batch=loss.item())
                
        
        metrics_result = self.test_metrics.end_a_epoch(epoch)
        
        print('|||test result:{}'.format(metrics_result))
        print('|||test epoch {} TIME: {}'.format( epoch, get_time(self.start_time)))
        print("|||MEM: {}%".format(psutil.virtual_memory().percent)) 
        save_dict={}
        for metric in self.best_key_metrics:
            if metric not in metrics_result:
                continue
            else:
                if metrics_result[metric] > self.best_key_metrics[metric]:
                    save_dict[metric] = True
                    self.best_key_metrics[metric] = metrics_result[metric]
                else:
                    save_dict[metric] = False
        
        return save_dict


    def _save_best_checkpoint(self, epoch, sub_name=''):
        """
        Saving bes checkpoint

        Parameters:
            epoch (int): current epoch
            sub_name (str): sub name of the checkpoint file, metric name
        
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if os.path.exists('./best_models/'+self.dataset_name+'/'+self.ex_id) == False:
            os.makedirs('./best_models/'+self.dataset_name+'/'+self.ex_id)
        
        filename =  'best_models/{}/{}/{}_best_epoch.pth'.format(self.dataset_name, self.ex_id, sub_name)

        torch.save(state, filename)
        self.logger.info("Saving best checkpoint: {} ...".format(filename))

        

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _dict_to_device(self, input_dict, device):
        for key, value in input_dict.items():
            input_dict[key] = torch.tensor(input_dict[key]).to(device)
        # print('input_dict:{}'.format(input_dict))
        return input_dict
    
    
    ###########################################################################################################
    
  