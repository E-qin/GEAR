from abc import ABC, abstractmethod
import json
import os
import pickle
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, defaultdict
import model.metric as module_metric
 

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    if fname.exists():
        with fname.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)
    else:
        return OrderedDict()

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """

    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
class OutputMetricTracker:
    '''QWC 2023-07-22 GMT+8 17:00
    OutputMetricTracker 
       Collect the results for each batch.
        Store the results for each epoch and save them to a file.
        Calculate metrics for each epoch.
    '''
    def __init__(self, output_config, metric_ftns, data_provider, writer=None, mode='train', ex_id=''):
        
        assert mode in ['train', 'test']
        assert ex_id != ''
        
        self.output_config = output_config
        self.atomicId_labels_dict_metrics = data_provider.atomicId_labels_dict_metrics
        self.atomicId_docid_dict = data_provider.atomicId_docid_dict
        self.mode = mode
        self.writer = writer
        self.ex_id = ex_id
        
        self.metric_ftns = metric_ftns
        self.columns = ['epoch','loss']+[met.__name__ for met in self.metric_ftns]
        self.metrics_per_epoch = pd.DataFrame(columns=self.columns) 
        
        self.loss_per_epoch = defaultdict(float) 
        self.batch_num_per_epoch = 0 
        self.output_per_epoch = {} 
        # {epoch: {'target': [t_list1, t_list2, ...],
        #           'pred':[p_list1,p_list1]
        #           'atomic_id:[id1,id2]},...}
        
        
        
        
    def start_a_epoch(self,epoch):
        self.output_per_epoch[epoch] = defaultdict(list)
        self.batch_num_per_epoch = 0
    
    
    def collect_a_batch(self, epoch, target_batch, pred_batch ,atomic_id_batch, loss_batch):
        '''
        collect a batch of results
            before a epoch, call start_a_epoch() to initialize the dict
            during a epoch, call collect_a_batch() to collect the results
            after a epoch, call end_a_epoch() to save the results
        '''
        self.loss_per_epoch[epoch] += loss_batch
        self.batch_num_per_epoch += 1
        batch_size = len(target_batch)
        pred_len = len(pred_batch)
        assert pred_len % batch_size == 0  
        assert len(atomic_id_batch) == batch_size
        beam_num = pred_len // batch_size
        pred_batch = [pred_batch[i:i+beam_num] for i in range(0, pred_len, beam_num)]
        
        for target, pred_list, atomic_id in zip(target_batch, pred_batch,atomic_id_batch):
            if type(target) != list:
                if type(target) != str:
                    raise TypeError('ERROR: target must be list or str, but got {}: ({})'.format(type(target), target))
                target = [target]
            self.output_per_epoch[epoch]['target'].append(target)
            self.output_per_epoch[epoch]['pred'].append(pred_list)
            self.output_per_epoch[epoch]['atomic_id'].append(atomic_id)
            
            
    def end_a_epoch(self, epoch):

        atomic_ids = self.output_per_epoch[epoch]['atomic_id'] 
        pred_lists = self.output_per_epoch[epoch]['pred']
        target_lists = self.output_per_epoch[epoch]['target']
        
        
        if self.mode == 'train':
            real_target_lists = [ [atomic_id] + self.atomicId_labels_dict_metrics[atomic_id] for atomic_id in atomic_ids ]
        elif self.mode == 'test':
            real_target_lists = [ self.atomicId_labels_dict_metrics[atomic_id] for atomic_id in atomic_ids ]
        for ith,rtl in enumerate(real_target_lists):
            real_target_lists[ith] = [self.atomicId_docid_dict[atomic_id] for atomic_id in rtl]

        self.loss_per_epoch[epoch] /= self.batch_num_per_epoch
        metrics_this_epoch={}
        for met_f in self.metric_ftns: 
            metrics_this_epoch[met_f.__name__] = met_f(real_target_lists, pred_lists)

        metrics_this_epoch['loss'] = self.loss_per_epoch[epoch]
        metrics_this_epoch['epoch'] = epoch   
        
        new_row = pd.DataFrame(metrics_this_epoch, index=[epoch])
        self.metrics_per_epoch = pd.concat([self.metrics_per_epoch, new_row])

        results_dir = self.output_config['results_dir']
        metrics_dir = self.output_config['metrics_dir']
        with open(os.path.join(results_dir, '{}_{}_results.json'.format(self.ex_id, self.mode)), 'w') as f:
            json.dump(self.output_per_epoch, f, indent=4, ensure_ascii=False)
        # with open(os.path.join(metrics_dir, '{}_{}_metrics.txt'.format(self.ex_id, self.mode)), 'a') as f:
        #     f.write('-'*50+'\nEpoch {} {}:\n'.format(epoch, self.mode))
        #     for key, value in metrics_this_epoch.items():
        #         f.write('    {:15s}: {}\n'.format(str(key), value))
        #     f.write('-'*50+'\n')
        self.metrics_per_epoch.to_excel(
            os.path.join(metrics_dir, '{}_{}_metrics.xlsx'.format(self.ex_id, self.mode)),
            sheet_name = "ELAM",index = False,na_rep = 0,inf_rep = 0
        )
        ret = {k:v for k,v in metrics_this_epoch.items() if k != 'epoch' and k != 'loss'}
        return ret
    
 
    
class OutputMetricTracker_LeCaRDv2:

    def __init__(self, output_config, metric_ftns, data_provider, writer=None, mode='train', ex_id=''):
        
        assert mode in ['train', 'test']
        assert ex_id != ''
        
        self.output_config = output_config
        self.semanticId_pid_dict = data_provider.semanticId_pid_dict
        self.qid_labels_dict_metrics = data_provider.qid_labels_dict_metrics
        # self.atomicId_labels_dict_metrics = data_provider.atomicId_labels_dict_metrics
        # self.atomicId_docid_dict = data_provider.atomicId_docid_dict
        self.mode = mode
        self.writer = writer
        self.ex_id = ex_id
        
        self.metric_ftns = metric_ftns
        self.columns = ['epoch','loss']+[met.__name__ for met in self.metric_ftns]
        self.metrics_per_epoch = pd.DataFrame(columns=self.columns)
        
        self.loss_per_epoch = defaultdict(float) 
        self.batch_num_per_epoch = 0  
        
        self.output_per_epoch = {} 
        # {epoch: {'target': [t_list1, t_list2, ...],
        #           'pred':[p_list1,p_list1]
        #           'atomic_id:[id1,id2]},...}
        
        
        
        
    def start_a_epoch(self,epoch):
        self.output_per_epoch[epoch] = defaultdict(list)
        self.batch_num_per_epoch = 0
    
    
    def collect_a_batch(self, epoch, target_batch, pred_batch ,atomic_id_batch, loss_batch):
        '''
        collect a batch of results
            before a epoch, call start_a_epoch() to initialize the dict
            during a epoch, call collect_a_batch() to collect the results
            after a epoch, call end_a_epoch() to save the results
        '''
        self.loss_per_epoch[epoch] += loss_batch
        self.batch_num_per_epoch += 1
        batch_size = len(target_batch)
        pred_len = len(pred_batch)
        assert pred_len % batch_size == 0  
        assert len(atomic_id_batch) == batch_size
        beam_num = pred_len // batch_size
        pred_batch = [pred_batch[i:i+beam_num] for i in range(0, pred_len, beam_num)]
        
        for target, pred_list, atomic_id in zip(target_batch, pred_batch, atomic_id_batch):
            if type(target) != list:
                if type(target) != str:
                    raise TypeError('ERROR: target must be list or str, but got {}: ({})'.format(type(target), target))
                target = [target]
            self.output_per_epoch[epoch]['target'].append(target)
            self.output_per_epoch[epoch]['pred'].append(pred_list)
            self.output_per_epoch[epoch]['atomic_id'].append(atomic_id)
            
            
    def end_a_epoch(self, epoch):
        qids = self.output_per_epoch[epoch]['atomic_id'] 
        pred_lists = self.output_per_epoch[epoch]['pred']
        # target_lists = self.output_per_epoch[epoch]['target']
        
        pid_pred_lists = []
        for ith, new_id_list in enumerate(pred_lists):
            original_id_list = []
            for new_id in new_id_list:
                if new_id not in self.semanticId_pid_dict:
                    continue
                old_pid = self.semanticId_pid_dict[new_id]
                if old_pid not in original_id_list:
                    original_id_list.append(old_pid) 
            pid_pred_lists.append(original_id_list)
        
        real_target_lists = [ self.qid_labels_dict_metrics[qid] for qid in qids ] # [[['1889986', 3], ....], ...]
        if type(real_target_lists[0][0]) != str:
            real_target_lists = [ [pid for pid, _ in rtl] for rtl in real_target_lists ] # [[1889986, ...], ...]
            
        self.loss_per_epoch[epoch] /= self.batch_num_per_epoch
        metrics_this_epoch={}
        for met_f in self.metric_ftns: 
            metrics_this_epoch[met_f.__name__] = met_f(real_target_lists, pid_pred_lists)

        metrics_this_epoch['loss'] = self.loss_per_epoch[epoch]
        metrics_this_epoch['epoch'] = epoch   
        
        new_row = pd.DataFrame(metrics_this_epoch, index=[epoch])
        self.metrics_per_epoch = pd.concat([self.metrics_per_epoch, new_row])
        results_dir = self.output_config['results_dir']
        metrics_dir = self.output_config['metrics_dir']
        with open(os.path.join(results_dir, '{}_{}_results.json'.format(self.ex_id, self.mode)), 'w') as f:
            json.dump(self.output_per_epoch, f, indent=4, ensure_ascii=False)
        self.metrics_per_epoch.to_excel(
            os.path.join(metrics_dir, '{}_{}_metrics.xlsx'.format(self.ex_id, self.mode)),
            sheet_name = "LeCaRDv2",index = False,na_rep = 0,inf_rep = 0
        )
        ret = {k:v for k,v in metrics_this_epoch.items() if k != 'epoch' and k != 'loss'}
        return ret
    
    
