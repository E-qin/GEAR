# QWC edit 2023/07/02 GMT+8 22:00
# from torchvision import datasets, transforms
import os
import pdb
from typing import List
from base import BaseDataLoader
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import pickle


def topCrimes(crimeSorted:list, topk=3)->str:
    r''' QQQWWWCCC
        topCrimes:
            Select the top k crimes from the list of crimes sorted by the number of occurrences.

        Parameters:
            crimeSorted (list) - List of crimes sorted by frequency.
            topk (int) - The maximum number of crimes to select.

        Returns: str
            top3Crimes (str) - Concatenated string of selected crimes, separated by spaces.
    '''
    crimeSorted_topk = crimeSorted[:topk]
    if len(crimeSorted_topk) != 0 :
        crimeSorted_topk_names, _ = zip(*crimeSorted_topk)
        return " ".join(crimeSorted_topk_names)
    else:
        return ""


def _big_batch_split(big_batch, effective_lens):
    '''
    split a big batch into list
    
    effective_len - list of the size
    big_batch - torch.tensor shape: (sizes' SUM, seq_len)
    '''
    assert big_batch.size(0) == sum(effective_lens)
    lens = effective_lens.copy()
    
    for i in range(len(lens)): # prefix sum
        if i-1>=0:
            lens[i]+=lens[i-1]
    lens = [0]+lens

    return [ big_batch[ lens[i-1]:lens[i] , :]  for i in range(1,len(lens)) ], lens


def _pad_tensor(split_batch:list, pad_token_id:int, force_size:int):
    '''
    Pad each tensor(?,seq_len) in the input list (containing multiple tensors) 
    with pad_token_id in their first dimension to the specified force_size
    '''
    assert type(pad_token_id) == int
    for i,_tensor in enumerate(split_batch):
        missing_size = force_size - _tensor.size(0)
        if missing_size == 0:
            continue
        pad_tensor = torch.full((missing_size, _tensor.size(-1)), pad_token_id, 
                                dtype=_tensor.dtype, device=_tensor.device)
        split_batch[i] = torch.cat((_tensor, pad_tensor),dim=0)




class GR_Extra_Train_Dataset(Dataset):
    def __init__(self, extra_train_data_src, config, ex_id ,data_provider):
        dataset_name = data_provider.dataset_name
        self.data_list = [] # to prepare [[pid, text, docid],..]
        for pid, pair_list in extra_train_data_src.items():
            for ith, pair in enumerate(pair_list):
                sent_list,docid = pair
                self.data_list.append([
                    pid,                                                                      # 0
                    ''.join(sent_list)[ : config[dataset_name + "_data_loaders"]["max_input_len"]],            # 1
                    docid                                                                          # 2
                ]) 
                

        if config['data_dbg']: 
            self.data_list = self.data_list[:21]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return  self.data_list[index]


class GR_Dataset(Dataset):
    def __init__(self, data_src, config, ex_id ,data_provider, mode='train'):
        dataset_name = data_provider.dataset_name
        if mode not in ['train','test']:
            raise ValueError("ERROR: Dataset mode must be 'train' or 'test', but got {}".format(mode))

        if dataset_name == 'ELAM':
            if mode == 'train':
                self.data_list = [] # to prepare [[atomic_id, text, docid, docid_list, flag],..]
                for atomic_id, pair_list in data_src.items():
                    for ith,pair in enumerate(pair_list):
                        sent_list,docid = pair
                        self.data_list.append([
                            atomic_id,                                                                      # 0
                            ''.join(sent_list)[ : config["ELAM_data_loaders"]["max_input_len"]],            # 1
                            docid,                                                                          # 2
                            data_provider.get_relevant_new_id_list(atomic_id),                              # 3
                            1 if ith==0 else 0                                                              # 4
                        ]) 
                        
            elif mode == 'test':
                self.data_list = []  # to prepare [[atomic_id, text, docid_list],...]
                for atomic_id, pair in data_src.items():
                    sent_list, label_list = pair
                    self.data_list.append([
                        atomic_id,                                                                          # 0
                        ''.join(sent_list)[ : config["ELAM_data_loaders"]["max_input_len"]],                # 1
                        label_list,                                                                         # 2
                        1                                                                                   # 3
                    ]) 
        elif dataset_name.startswith('LeCaRD'):
            if mode == 'train':
                self.data_list = [] # to prepare [[qid, text, docid, docid_list, flag],..]
                for atomic_id, pair_list in data_src.items():
                    for ith,pair in enumerate(pair_list):
                        sent_list,docid = pair
                        self.data_list.append([
                            atomic_id,                                                                      # 0
                            ''.join(sent_list)[ : config[dataset_name + "_data_loaders"]["max_input_len"]],            # 1
                            docid,                                                                          # 2
                            data_provider.qid_relevantNewId_dict[atomic_id] ,                             # 3
                            1 if ith==0 else 0                                                              # 4
                        ]) 
                        
            elif mode == 'test':
                self.data_list = []  # to prepare [[qid, text, docid_list],...]
                for atomic_id, pair in data_src.items():
                    sent_list, label_list = pair
                    self.data_list.append([
                        atomic_id,                                                                          # 0
                        ''.join(sent_list)[ : config[dataset_name + "_data_loaders"]["max_input_len"]],                # 1
                        label_list,                                                                         # 2
                        1                                                                                   # 3
                    ]) 
        else:
            raise ValueError("Unknown dataset_name: {}".format(dataset_name))            
        

            
        if config['data_dbg']: 
            self.data_list = self.data_list[:21]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return  self.data_list[index]
    
    
            
class GR_Extra_Train_Collate():
    def __init__(self, tokenizer, config, data_provider):         
        dataset_name = data_provider.dataset_name
        self.tokenizer = tokenizer
        self.max_input_len = config[dataset_name+"_data_loaders"]["max_input_len"]
        self.max_output_len = config[dataset_name+"_data_loaders"]["max_output_len"]
        self.data_provider = data_provider
        self.docid_ty = data_provider.docid_ty     
        
    def _encode_labels_batch(self, target_list_batch : List[List[str]]):
        '''
        Encode the label list in the batch and fill it to the same size with pad token id
        return:
            tensor (labels_size's SUM in 1 batch, seq_len)
            tensor (batch_size,)

        '''
        effective_labels_lens = [len(target_list) for target_list in target_list_batch]
        effective_labels_lens_tensor = torch.tensor(effective_labels_lens, dtype=torch.int64) # shape: (batch_size,)
        
        target_list_big_batch = [ target_str  for target_list in target_list_batch for target_str in target_list] # len: SUM of labels_size of the batch
        
        if self.docid_ty.startswith('decode_tree'): 
            encoded_big_batch = self.data_provider.decode_tree_encode_batch(target_list_big_batch,padding_value=0)
        else:  
            target_list_encoding = self.tokenizer(target_list_big_batch, padding="longest",
                                    max_length=self.max_output_len,
                                    truncation=True, return_tensors="pt",
                                    )
            encoded_big_batch = target_list_encoding.input_ids # shape: (labels_sizes' SUM, seq_len)
        return  encoded_big_batch,       effective_labels_lens_tensor
    
    def __call__(self, batch):

        atomic_id_batch, input_batch, target_1_batch = zip(*batch)
        atomic_id_batch, input_batch, target_1_batch = \
            list(atomic_id_batch), list(input_batch), list(target_1_batch)
       
        
        
        encoding = self.tokenizer(input_batch, padding="longest",
                                max_length=self.max_input_len,
                                truncation=True, return_tensors="pt",
                                )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        
        if self.docid_ty.startswith('decode_tree'): 
            label_1_batch = self.data_provider.decode_tree_encode_batch(target_1_batch, padding_value=0)[:, 1:]
        else:              
            target_1_encoding = self.tokenizer(target_1_batch, padding="longest",
                                    max_length=self.max_output_len,
                                    truncation=True, return_tensors="pt",
                                    )
            label_1_batch = target_1_encoding.input_ids 
            label_1_batch[label_1_batch == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100 so it's ignored by the loss
            
   
        
        return (
            {
                "input_ids":input_ids, 
                "attention_mask":attention_mask, 
                "label_1_batch":label_1_batch,   # pt (batch_size , seq_len)                      
            } , 
            target_1_batch , # list : [docid1, docid2, ...] 
            atomic_id_batch # query atomic_id batch, len : batch_size * 1
        )
        
            
class GR_Collate():
 
    def __init__(self, tokenizer, config, data_provider, mode='train'):
        if mode not in ['train','test']:
            raise ValueError("ERROR: GR_Collate mode must be 'train' or 'test', but got {}".format(mode))
        dataset_name = data_provider.dataset_name
        self.tokenizer = tokenizer
        self.max_input_len = config[dataset_name+"_data_loaders"]["max_input_len"]
        self.mode = mode
        self.max_output_len = config[dataset_name+"_data_loaders"]["max_output_len"]
        self.data_provider = data_provider
        # self.labels_size = data_provider.labels_size
        # self.do_pad_labels_lists = config[dataset_name+"_data_loaders"]["do_pad_labels_lists"] # 0 or 1
        self.docid_ty = data_provider.docid_ty


    
    def _encode_labels_batch(self, target_list_batch : List[List[str]]):
        '''
        Encode the label list in the batch and fill it to the same size with pad token id
        return:
            tensor (labels_size's SUM in 1 batch, seq_len)
            tensor (batch_size,)

        '''
        effective_labels_lens = [len(target_list) for target_list in target_list_batch]
        effective_labels_lens_tensor = torch.tensor(effective_labels_lens, dtype=torch.int64) # shape: (batch_size,)
        
        target_list_big_batch = [ target_str  for target_list in target_list_batch for target_str in target_list] # len: SUM of labels_size of the batch
        
        if self.docid_ty.startswith('decode_tree'): 
            encoded_big_batch = self.data_provider.decode_tree_encode_batch(target_list_big_batch,padding_value=0)
        else:  
            target_list_encoding = self.tokenizer(target_list_big_batch, padding="longest",
                                    max_length=self.max_output_len,
                                    truncation=True, return_tensors="pt",
                                    )
            encoded_big_batch = target_list_encoding.input_ids # shape: (labels_sizes' SUM, seq_len)

        return  encoded_big_batch,       effective_labels_lens_tensor
    
    def __call__(self, batch):
        if self.mode == 'train':
            atomic_id_batch, input_batch, target_1_batch, target_list_batch, mask_batch = zip(*batch)
            atomic_id_batch, input_batch, target_1_batch, target_list_batch, mask_batch = \
                list(atomic_id_batch), list(input_batch), list(target_1_batch), list(target_list_batch), list(mask_batch)
        elif self.mode == 'test':
            atomic_id_batch, input_batch,  target_list_batch, mask_batch = zip(*batch)
            atomic_id_batch, input_batch,  target_list_batch, mask_batch = \
                list(atomic_id_batch), list(input_batch),  list(target_list_batch), list(mask_batch)
            target_1_batch = [target_list[0] for target_list in target_list_batch]  
        
        
        encoding = self.tokenizer(input_batch, padding="longest",
                                max_length=self.max_input_len,
                                truncation=True, return_tensors="pt",
                                )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        
        if self.docid_ty.startswith('decode_tree'): 
            # target_1_batch_encoding_1 = [self.data_provider.decode_tree_encode(target_str,return_ty='pt') 
            #                              for target_str in target_1_batch]
            # label_1_batch = pad_sequence(target_1_batch_encoding_1, 
            #                            batch_first=True, padding_value=-100)[:, 1:] # remove the first pad token(root 0) of the label, which is not needed by LM original loss
            label_1_batch = self.data_provider.decode_tree_encode_batch(target_1_batch, padding_value=0)[:, 1:]
        else:              
            target_1_encoding = self.tokenizer(target_1_batch, padding="longest",
                                    max_length=self.max_output_len,
                                    truncation=True, return_tensors="pt",
                                    )
            label_1_batch = target_1_encoding.input_ids 
            label_1_batch[label_1_batch == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100 so it's ignored by the loss
            
        label_list_batch, effective_labels_lens_tensor =\
            self._encode_labels_batch(target_list_batch) # label_list_batch shape: (labels_sizes' SUM , seq_len) 
        
        return (
            {
                "input_ids":input_ids, 
                "attention_mask":attention_mask, 
                "label_1_batch":label_1_batch,   # pt (batch_size , seq_len)                      #  1 encoded docid batch
                "label_list_batch":label_list_batch[:, 1:],  # pt (labels_sizes' SUM or batch_size * labels_size , seq_len)   #  labels_size encoded docids batch
                "effective_labels_lens":effective_labels_lens_tensor, # pt (batch_size,)
                "rl_mask_batch": torch.tensor(mask_batch, dtype=torch.int64) # pt (batch_size,)
            } , 
            target_1_batch , # list : [docid1, docid2, ...] 
            target_list_batch, # list : [label_list1, label_list2, ...] 
            atomic_id_batch # query atomic_id batch, len : batch_size * 1
        )



def build_loaders(config, ex_id, tokenizer,
                    data_provider,
                    final_model_input=None,
                    extra_input=None,
                    join=False
                    ):
    
    '''
    build_loaders:

        final_model_input:
        {  
            "train":
                {
                    atomic_id: [[text, own_docid1],[text, relevant_docid1],[text, relevant_docid2],...]
                    ...
                }
            "test":
                {
                    atomic_id: [text, [relevant_docid1, relevant_docid2,...]]
                    ...
                }
        }
    
    '''

    if final_model_input==None:
        model_input_path = \
        os.path.join(
            config[data_provider.dataset_name+"_data_prepare"]["model_input_dir"], 
            data_provider.dataset_name,
            "{}_data.json".format(ex_id)
        )
        with open(model_input_path,'r') as f:
            data = json.load(f)
    else:
        data = final_model_input

    assert extra_input!=None 

    # train_data_src = {
    #     "train_pair":data["train_pair"],
    #     "train_list":data["train_list"]
    # }
    train_data_src = data["train"]
    test_data_src = data["test"]
    train_data_generator = GR_Dataset(train_data_src, config, ex_id,
                                            data_provider,mode='train')
    test_data_genarator = GR_Dataset(test_data_src, config, ex_id, 
                                            data_provider,mode='test')
    train_collate = GR_Collate(tokenizer, config, data_provider,mode='train')
    test_collate = GR_Collate(tokenizer, config, data_provider,mode='test')
    if data_provider.dataset_name=='ELAM':
        if extra_input!=None and len(extra_input)!=0:
            extra_train_data_generator = GR_Extra_Train_Dataset(extra_input, config, ex_id, data_provider)
            extra_train_collate = GR_Extra_Train_Collate(tokenizer, config, data_provider)
        
            extra_train_loader = DataLoader(
                extra_train_data_generator,
                batch_size=config[data_provider.dataset_name + "_data_loaders"]['batch_size'],
                shuffle=config[data_provider.dataset_name + "_data_loaders"]['shuffle'],
                num_workers=config[data_provider.dataset_name + "_data_loaders"]['num_workers'],
                collate_fn=extra_train_collate
            )
        
        train_data_loader =  DataLoader(
            train_data_generator,
            batch_size=config["ELAM_data_loaders"]['batch_size'],
            shuffle=config["ELAM_data_loaders"]['shuffle'],
            num_workers=config["ELAM_data_loaders"]['num_workers'],
            collate_fn=train_collate
        )
        
        test_data_loader =  DataLoader(
            test_data_genarator,
            batch_size=config["ELAM_data_loaders"]['batch_size'],
            shuffle=config["ELAM_data_loaders"]['shuffle'],
            num_workers=config["ELAM_data_loaders"]['num_workers'],
            collate_fn=test_collate
        )
        
        if not join:
            if extra_input!=None and len(extra_input)!=0:
                return train_data_loader, test_data_loader, extra_train_loader, None
            return train_data_loader, test_data_loader, None, None
        else:
            join_train_data_loader =  DataLoader(
                train_data_generator,
                batch_size=config["ELAM_data_loaders"]['join_batch_size'],
                shuffle=config["ELAM_data_loaders"]['shuffle'],
                num_workers=config["ELAM_data_loaders"]['num_workers'],
                collate_fn=train_collate
            )
            if extra_input!=None and len(extra_input)!=0:
                return train_data_loader, test_data_loader, extra_train_loader, join_train_data_loader
            return train_data_loader, test_data_loader, None, join_train_data_loader
    
    elif data_provider.dataset_name.startswith("LeCaRD"):
        if extra_input!=None and len(extra_input)!=0:
            extra_train_data_generator = GR_Extra_Train_Dataset(extra_input, config, ex_id, data_provider)
            extra_train_collate = GR_Extra_Train_Collate(tokenizer, config, data_provider)
            
            extra_train_loader = DataLoader(
                extra_train_data_generator,
                batch_size=config[data_provider.dataset_name + "_data_loaders"]['batch_size'],
                shuffle=config[data_provider.dataset_name + "_data_loaders"]['shuffle'],
                num_workers=config[data_provider.dataset_name + "_data_loaders"]['num_workers'],
                collate_fn=extra_train_collate
            )
        
        train_data_loader =  DataLoader(
            train_data_generator,
            batch_size=config[data_provider.dataset_name + "_data_loaders"]['batch_size'],
            shuffle=config[data_provider.dataset_name + "_data_loaders"]['shuffle'],
            num_workers=config[data_provider.dataset_name + "_data_loaders"]['num_workers'],
            collate_fn=train_collate
        )
        
        test_data_loader =  DataLoader(
            test_data_genarator,
            batch_size=config[data_provider.dataset_name + "_data_loaders"]['batch_size'],
            shuffle=config[data_provider.dataset_name + "_data_loaders"]['shuffle'],
            num_workers=config[data_provider.dataset_name + "_data_loaders"]['num_workers'],
            collate_fn=test_collate
        )
        
        if not join:
            return train_data_loader, test_data_loader, extra_train_loader, None  # must be extra
        else:
            join_train_data_loader =  DataLoader(
                train_data_generator,
                batch_size=config[data_provider.dataset_name + "_data_loaders"]['join_batch_size'],
                shuffle=config[data_provider.dataset_name + "_data_loaders"]['shuffle'],
                num_workers=config[data_provider.dataset_name + "_data_loaders"]['num_workers'],
                collate_fn=train_collate
            )
            return train_data_loader, test_data_loader, extra_train_loader, join_train_data_loader
    
    else:
        raise ValueError("Unknown dataset_name: {}".format(data_provider.dataset_name))
    

