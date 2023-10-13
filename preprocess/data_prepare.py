# QWC edit 2023/07/02 GMT+8 22:00
# from data_prepare import *

from collections import defaultdict
import copy
import json
import os
import pdb
import pickle
import random
import time
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from typing import Union, List
from Index_tree.tree_structure import Node
from collections import Counter






def remove_duplicates(lst):
    # Remove duplicates from the list and maintain the original order
    return list(dict.fromkeys(lst))



def change_docid(origin_docid,docid_ty,seperator="",pad_0=True,step_length:dict=None):
    '''
    change_docid:tree_atomic, tree_kmeans, random_seq
        
    step_length = {
        "tree_atomic":"11212",
        "tree_kmeans":"112121",
        "random_seq":"4",
        ...
    }
    '''
    assert step_length is not None
    if docid_ty not in step_length:
        raise ValueError("ERROR: docid_ty must be one of {}".format(list(step_length.keys())))
    
    def split_pad_id(origin_id, step_length_dict_str):
        '''
        split_pad_id:
        '''
        content = origin_id.split("_")
        res = []
        for i in range(len(content)):
            while len(content[i]) < int(step_length_dict_str[i]):
                content[i] = "0" + content[i]
            res.append(content[i])
        return res
    
    if pad_0:
        return seperator.join(split_pad_id(origin_docid, step_length[docid_ty]))
    else:
        return seperator.join(origin_docid.split("_"))


def get_rid_of_same_prefix_part(atomicId_docid_dict):
    '''
    get_rid_of_same_prefix_part:
    '''
    docid_list = list(atomicId_docid_dict.values())
    docid_list.sort()
    prefix = os.path.commonprefix(docid_list)
    if prefix == "":
        return atomicId_docid_dict
    else:
        res = {}
        for atomicId, docid in atomicId_docid_dict.items():
            res[atomicId] = docid[len(prefix):]
        return res


def truncate_or_append_eos(dim1_tensor:torch.Tensor, eos_token_id:int):

    eos_indices = (dim1_tensor == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_indices) > 0:
        first_eos_index = eos_indices[0]
        truncated_tensor = dim1_tensor[:first_eos_index+1]
        return truncated_tensor
    else:
        eos_tensor = torch.tensor([eos_token_id])
        result = torch.cat((dim1_tensor, eos_tensor))
        return result


class PrepareData:
    '''
    
    attribute:
        
        (for ELAM) 
        self.dataset_name
        self.docid_ty
        self.data_prepare_config
        self.atomicIds
        self.atomicId_labels_dict #  force labels_size
            - self.golden_labels
            - self.extensive_labels
        self.atomicId_labels_dict_metrics # for metrics 
            - self.golden_labels
            - self.extensive_labels    
        self.labels_size
        slef.atomicId_docid_dict
        self.test_set
        
        --------------------------
        
        (for LeCaRDv2)
        self.dataset_name
        self.docid_ty
        self.data_prepare_config
        self.semanticId_pid_dict # to pid
        self.multi_charge_test_set #
        self.qids
        self.pids
        self.pid_docids_dict 
        self.qid_relevantNewId_dict #  force labels_size
        self.qid_relevantNewId_dict_metrics # for metrics
        self.qid_labels_dict_metrics # 
    '''
    
    
    def __init__(self,config, dataset_name, tree_pad_token_id=0):
        self.dataset_name = dataset_name
        self.data_prepare_config = config[dataset_name+'_data_prepare']
        self.labels_size = self.data_prepare_config["labels_size"] # for train
        # with open(self.data_prepare_config['atomicIds_path'] ,'r') as f: # all old ids
        #     self.atomicIds = json.load(f)
        
        # atomic id, labels and test_set
        if dataset_name == "ELAM":
            
            with open(self.data_prepare_config['atomicIds_path'] ,'r') as f: # all old ids
                self.atomicIds = json.load(f)
            
            with open(self.data_prepare_config['extensive_labels_path'] ,'r') as f: # extensive_labels
                self.extensive_labels = json.load(f)
            with open(self.data_prepare_config['labels_path'],'r') as f: # golden_labels
                self.golden_labels = json.load(f)
            if self.data_prepare_config['labels_ty']['golden']:
                self.atomicId_labels_dict = self.golden_labels
            elif self.data_prepare_config['labels_ty']['extensive']:
                self.atomicId_labels_dict = self.extensive_labels
            for atomic_id, label_list in self.atomicId_labels_dict.items():
                self.atomicId_labels_dict[atomic_id] = label_list[:self.labels_size]
            self.atomicId_labels_dict_metrics = copy.deepcopy(self.atomicId_labels_dict)
            self._truncate_cp(self.atomicId_labels_dict,self.labels_size) 
            

            degree1_as_test_set_path = \
                self.data_prepare_config['test_set_path']
            with open(degree1_as_test_set_path,'r') as f: 
                self.test_set = json.load(f)

            
        elif dataset_name == 'LeCaRDv2':
            with open(self.data_prepare_config['atomicIds_path']['p'],'r') as f:
                self.pids = json.load(f)
            # with open(self.data_prepare_config['atomicIds_path']['q'],'r') as f:
            #     self.qids = json.load(f)
            with open(self.data_prepare_config['labels_path'],'r') as f:
                self.qid_labels_dict_metrics = json.load(f)
            with open(self.data_prepare_config['test_set_path'],'r') as f:
                self.test_set = json.load(f)
            with open(self.data_prepare_config['multi_charge_test_set_path'] ,'r') as f:
                self.multi_charge_test_set = json.load(f)
            self.test_set = list(map(str,self.test_set))
            self.multi_charge_test_set = list(map(str,self.multi_charge_test_set)) 
            del_qids = []
            for qid, pair_list in self.qid_labels_dict_metrics.items():
                self.qid_labels_dict_metrics[qid] = [[pid,rank] for pid,rank in pair_list if rank >= 2]
                if len(self.qid_labels_dict_metrics[qid]) == 0:
                    del_qids.append(qid)
            # self.qids = [qid for qid in self.qids if qid not in del_qids]
            self.test_set = [qid for qid in self.test_set if qid not in del_qids]  
            self.multi_charge_test_set = [qid for qid in self.multi_charge_test_set if qid not in del_qids]          
            self.qid_labels_dict_metrics = {qid:pair_list for qid,pair_list in self.qid_labels_dict_metrics.items() if qid not in del_qids}
            # self.qid_labels_dict_metrics = {qid:pair_list for qid,pair_list in self.qid_labels_dict_metrics.items() if len(pair_list) > 0}
            self.qids = list(self.qid_labels_dict_metrics.keys())
        elif dataset_name in ['LeCaRD_ver2','LeCaRD_version2']:
            with open(self.data_prepare_config['atomicIds_path']['p'],'r') as f:
                self.pids = json.load(f)
            with open(self.data_prepare_config['labels_path'],'r') as f: 
                self.qid_labels_dict_metrics = json.load(f) # 
            with open(self.data_prepare_config['test_set_path'],'r') as f:
                self.test_set = json.load(f)
            # with open(self.data_prepare_config['multi_charge_test_set_path'] ,'r') as f:
            #     self.multi_charge_test_set = json.load(f)
            self.test_set = list(map(str,self.test_set))
            # self.multi_charge_test_set = list(map(str,self.multi_charge_test_set)) 
            self.qids = list(self.qid_labels_dict_metrics.keys())
            
        
        self.root = None # defualt none


    def prepare_data(self,config,ex_id):
        '''
        call different f for different dataset
        '''
        if self.dataset_name == 'ELAM':
            return self.prepare_data_ELAM(config,ex_id)
        elif self.dataset_name.startswith('LeCaRD'):
            return self.prepare_data_LeCaRDv2(config,ex_id)
        else:
            raise ValueError("Error: dataset_name == {}".format(self.dataset_name))  
        
    def decode_tree_encode(self, docid:str, return_ty="pt")->Union[torch.Tensor, List[int]]:
        '''
        encode one id like '0_5_155_1' (can use it only when docid_ty startswith 'decode_tree' )
        return_ty: 'list' or 'pt'
        '''
        docid_encoding =  list(map(int,docid.split('_')))
        if return_ty == 'list':
            return docid_encoding
        elif return_ty == 'pt':
            return torch.tensor(docid_encoding,dtype=torch.long) 
        else:
            raise ValueError("Error: return_ty == {}".format(return_ty))
        
    def decode_tree_decode(self, docid_encoding:Union[torch.Tensor, List[int]])->str:
        '''
        decode one id, return a str like '0_5_155_1' 
        (can use it only when docid_ty startswith 'decode_tree' )
        
        '''
        if isinstance(docid_encoding,torch.Tensor):
            docid_encoding = docid_encoding.tolist()
            return "_".join(map(str,docid_encoding))
        elif isinstance(docid_encoding,list):
            return "_".join(map(str,docid_encoding))
    
    def decode_tree_encode_batch(self, docid_list:list, padding_value:int=0)->torch.Tensor:
        """ 
        encode a batch of docid, return a pt tensor (can use it only when docid_ty startswith 'decode_tree' )

        Args:
            docid_list (list): ['0_5_155_1','0_5_156_1',...]
            padding_value ...
        """
        
        return pad_sequence(
            [self.decode_tree_encode(target_str,return_ty='pt') 
                for target_str in docid_list], 
            batch_first=True, 
            padding_value=padding_value
        )
    
    def decode_tree_decode_batch(self, docid_encoding_batch:torch.Tensor, eos_token_id:int=1)->List[str]:
        assert isinstance(docid_encoding_batch, torch.Tensor) 
        docid_encoding_batch = docid_encoding_batch.cpu()
        ret = []
        for row in docid_encoding_batch:
            trimmed_tensor = truncate_or_append_eos(row, eos_token_id)
            # yield self.decode_tree_decode(trimmed_tensor.tolist())
            ret.append(self.decode_tree_decode(trimmed_tensor.tolist()))
        return ret


    
    
    def get_sentences_selected(self, sentence_list, sentence_score_dict, max_len=512, original_order=False):
        ''' QWC
        select_sentences_by_score:
            Select sentences from the list of sentences in text_list based on a score dictionary, and concatenate them into a long string, ensuring that the string's length does not exceed max_len.

        Parameters:
            text_list (list) - List of sentences.
            score_dict (dict) - Dictionary of sentence scores {0: 0.5, ...}.
            max_len (int) - Maximum length.
            original_order (bool) - Whether to maintain the original order.

        Returns:
            selected_sentences (list of str) - Output list of selected sentences.

        Note:
        '''
        selected_text = ""
        selected_sentences = []
        selected_sentence_index = []
        last_index = 0 
        last_sentence_len = 0 
        sorted_scores = sorted(sentence_score_dict.items(), key=lambda x: x[1], reverse=True)
        for index,score in sorted_scores:
            index = int(index)
            if len(selected_text)+len(sentence_list[index]) < max_len: 
                last_sentence_len = len(sentence_list[index])
                last_index = index
                selected_text += sentence_list[index]
                selected_sentence_index.append(index)
            else:
                last_sentence_len = max_len - len(selected_text)
                last_index = index
                selected_text += sentence_list[index][:last_sentence_len]
                selected_sentence_index.append(index)
                break

        if original_order:
            selected_sentence_index.sort()
        
        for index in selected_sentence_index:
            if index == last_index:
                selected_sentences.append(sentence_list[index][:last_sentence_len])
            else:
                selected_sentences.append(sentence_list[index])

        return selected_sentences
    
    
    
    def prepare_data_LeCaRDv2(self, config, ex_id, seperate_train=False):
        '''

        '''
        
        max_input_len = config['max_len']
        data_prepare_config = self.data_prepare_config
        QGen_mode = data_prepare_config["QGen_mode"]
        words1_QGen = data_prepare_config["words1_QGen"] if "words1_QGen" in data_prepare_config else False
        words2_QGen = data_prepare_config["words2_QGen"] if "words2_QGen" in data_prepare_config else False
        
        original_order = data_prepare_config["original_order"]
        
        # pq 2 options
        sentences_path = data_prepare_config["sentences_path"] 
        QGen_scores_path = data_prepare_config["QGen_scores_path"][QGen_mode] if QGen_mode in ["jieba","lac"] else None # None means no scoring
        
        atomicId_sentences_pq_dict = {}
        with open(sentences_path['p'],'r') as f:
            atomicId_sentences_pq_dict['p'] = json.load(f)
        with open(sentences_path['q'],'r') as f:
            atomicId_sentences_pq_dict['q'] = json.load(f)

        atomicId_score_seg_pq_dict = None
        if QGen_scores_path is not None:
            atomicId_score_seg_pq_dict = {}
            with open(QGen_scores_path['p'],'r') as f:
                atomicId_score_seg_pq_dict['p'] = json.load(f)
            with open(QGen_scores_path['q'],'r') as f:
                atomicId_score_seg_pq_dict['q'] = json.load(f)

        atomicId_sentencesSelected_pq_dict = {'p':{},'q':{}} # {ty:{atomic_id:[sentences]}}
        if QGen_mode in ["jieba","lac"]:
            for ty,atomicId_sentences in  atomicId_sentences_pq_dict.items(): # ty in ['q','p']
                print(f"Selecting {ty}'s sentences....")
                for atomic_id, sentence_list in tqdm(atomicId_sentences.items()):
                    sentence_score_dict = atomicId_score_seg_pq_dict[ty][atomic_id]
                
                    atomicId_sentencesSelected_pq_dict[ty][atomic_id] = self.get_sentences_selected(sentence_list, 
                                                                                    sentence_score_dict, 
                                                                                    max_len=max_input_len, 
                                                                                    original_order=original_order)
  
            
        elif QGen_mode == "no":
            for ty,atomicId_sentences in  atomicId_sentences_pq_dict.items(): # ty in ['q','p']
                print(f"Selecting {ty}'s sentences (first k tokens)....")
                for atomic_id, sentence_list in tqdm(atomicId_sentences.items()):
                    sent_num = len(sentence_list)
                    sentence_score_dict = {str(i):float(sent_num-i) for i in range(sent_num)} 

                    atomicId_sentencesSelected_pq_dict[ty][atomic_id] = self.get_sentences_selected(sentence_list, 
                                                                                    sentence_score_dict, 
                                                                                    max_len=max_input_len, 
                                                                                    original_order=original_order)

        else:
            raise ValueError("Error: QGen_mode == {}".format(QGen_mode))
        
        if words1_QGen or words2_QGen:
            with open(data_prepare_config["charges_path"],"r")as f:
                pid_charges = json.load(f)
            with open(data_prepare_config["total_crimes_path"],"r")as f:
                total_crimes = json.load(f)
            atomicId_words_pq_dict = {"p": defaultdict(list), "q": defaultdict(list)}
            if words1_QGen:
                with open(data_prepare_config["crime1_words_path"],"r")as f:
                    crime_words_dict = json.load(f)
                k = data_prepare_config["words1_QGen_k"]
                for id_str,sent_list in atomicId_sentences_pq_dict["p"].items():
                    charges = pid_charges[id_str]
                    words = list(set([word for charge in charges for word in crime_words_dict[charge]]))
                    orignal_s = "".join(sent_list)
                    counter = Counter()
                    for w in words:
                        counter[w] = orignal_s.count(w)
                    most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                    atomicId_words_pq_dict["p"][id_str] += most_common_words
                all_words = list(set([word for charge in total_crimes for word in crime_words_dict[charge]]))
                # all_words =  list(set([word for ,words in crime1_words_dict.items() for word in words]))
                for id_str,sent_list in atomicId_sentences_pq_dict["q"].items():
                    orignal_s = "".join(sent_list)
                    counter = Counter()
                    for w in all_words:
                        counter[w] = orignal_s.count(w)
                    most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                    atomicId_words_pq_dict["q"][id_str] += most_common_words
                
                
            if words2_QGen:
                with open(data_prepare_config["crime2_words_path"],"r")as f:
                    crime2_words_dict = json.load(f)
                k = data_prepare_config["words2_QGen_k"]
                if words1_QGen:
                    for c,words in crime2_words_dict.items():
                        crime2_words_dict[c] = list(set(words) - set(crime_words_dict[c]))
                crime_words_dict = crime2_words_dict
            
                for id_str,sent_list in atomicId_sentences_pq_dict["p"].items():
                    charges = pid_charges[id_str]
                    words = list(set([word for charge in charges for word in crime_words_dict[charge]]))
                    orignal_s = "".join(sent_list)
                    counter = Counter()
                    for w in words:
                        counter[w] = orignal_s.count(w)
                    most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                    atomicId_words_pq_dict["p"][id_str] += most_common_words
                all_words = list(set([word for charge in total_crimes for word in crime_words_dict[charge]]))
                # all_words =  list(set([word for ,words in crime1_words_dict.items() for word in words]))
                for id_str,sent_list in atomicId_sentences_pq_dict["q"].items():
                    orignal_s = "".join(sent_list)
                    counter = Counter()
                    for w in all_words:
                        counter[w] = orignal_s.count(w)
                    most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                    atomicId_words_pq_dict["q"][id_str] += most_common_words

            for ty,atomicId_sentences in  atomicId_sentencesSelected_pq_dict.items():
                for atomic_id, sentence_list in atomicId_sentences.items():
                    atomicId_sentencesSelected_pq_dict[ty][atomic_id] = \
                                    remove_duplicates(atomicId_words_pq_dict[ty][atomic_id]) + sentence_list


        self.docid_ty = data_prepare_config['docid_ty']
        if self.docid_ty in ["decode_tree_multi_crimes","decode_tree","kmeans","decode_tree_kmeans","decode_tree_Ultron_2"]:  
            with open(data_prepare_config["docid_paths"][self.docid_ty]["toSeq"],'r') as f:
                self.pid_docids_dict = json.load(f)  
            with open(data_prepare_config["docid_paths"][self.docid_ty]["toAto"],'r') as f:
                self.semanticId_pid_dict = json.load(f)
            # get decode tree
            tree_path = data_prepare_config["docid_paths"][self.docid_ty]["decode_tree"]
            print("getting decode tree from {}".format(tree_path))
            with open(tree_path, 'rb') as f:
                self.root = pickle.load(f)
        else:
            raise ValueError("Error: docid_ty == {}".format(self.docid_ty))
    
        
        final_train_pair_data = defaultdict(list)
        final_train_list_data = {} # {atomic_id: [text, [docid1, docid2, ...]], ...}
        final_test_data = {} # {atomic_id: [text, [docid1, docid2, ...]], ...}
        self.qid_relevantNewId_dict = {}
        for qid in self.qids:
            relevant_new_sematic_ids_list = self.get_relevant_new_multiIds_list(qid)
            self.qid_relevantNewId_dict[qid] = [id for new_ids in relevant_new_sematic_ids_list for id in new_ids]
        self.qid_relevantNewId_dict_metrics = copy.deepcopy(self.qid_relevantNewId_dict)
        self._truncate_cp(self.qid_relevantNewId_dict,self.labels_size)

        for qid in self.qids:

            sentence_list = atomicId_sentencesSelected_pq_dict['q'][qid]
            if qid not in self.test_set:
                for new_sematic_id in self.qid_relevantNewId_dict[qid]:
                    final_train_pair_data[qid].append([sentence_list,new_sematic_id]) 
                final_train_list_data[qid] = [sentence_list,self.qid_relevantNewId_dict[qid]]
            else:
                final_test_data[qid] = [sentence_list,self.qid_relevantNewId_dict[qid]]

        extra_input = defaultdict(list) 
        if "+qg" in data_prepare_config and data_prepare_config["+qg"]:              # add QG
            with open(data_prepare_config["+qg_path"],'r') as f:
                caseId_qgList_dict = json.load(f)
        
        for pid, sent_list in atomicId_sentencesSelected_pq_dict['p'].items():
            new_ids = self.get_new_multiIds(pid)
            for new_id in new_ids:
                extra_input["pid "+pid].append([sent_list,new_id])
                if "+qg" in data_prepare_config and data_prepare_config["+qg"]:    
                    num = len(caseId_qgList_dict[pid])
                    for j in range(1,1+num):
                        extra_input["pid "+pid+" fakeQ{}".format(j)].append([[caseId_qgList_dict[pid][j-1]],new_id])  # add QG

        final_model_input = {}
        if seperate_train:
            final_model_input['train_pair'] = final_train_pair_data
            final_model_input['train_list'] = final_train_list_data
            # self.each_qurey_max_rl_times = None 
        else: 
            final_model_input['train'] = final_train_pair_data
        
        final_model_input['test'] = final_test_data

        
        
        return final_model_input,extra_input
        
        
        
    
    def prepare_data_ELAM(self,config,ex_id,seperate_train=False):
        r'''QWC
        
        prepare_data_ELAM_ranking_loss:
            1. Select sentences based on scores.
            2. Split the data into training and testing documents.
            3. Prepare model input pairs.
            4. Save model inputs to the 'model_input' directory.

        Parameters:
            config (dict) - Configuration dictionary.
            ex_id (int) - Experiment ID.
            separate_train (bool) - Whether to split the training set into pairs and lists.

        Returns:
            final_model_input, atomicId_labels_dict, atomicId_docid_dict
            
            final_model_input:
            if seperate_train == true
            {  
                "train_pair":
                    {
                        atomic_id: [[text, own_docid1],[text, relevant_docid1],[text, relevant_docid2],...]
                        ...
                    }
                "train_list":
                    {
                        atomic_id: [text, [relevant_docid1, relevant_docid2,...]]
                    }
                "test":
                    {
                        atomic_id: [text, [relevant_docid1, relevant_docid2,...]]
                        ...
                    }
            }
            elif seperate_train == False
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
        
        max_input_len = config['max_len']
        data_prepare_config = self.data_prepare_config
        QGen_mode = data_prepare_config["QGen_mode"]
        original_order = data_prepare_config["original_order"]
        sentences_path = data_prepare_config["sentences_path"]
        QGen_scores_path = data_prepare_config["QGen_scores_path"][QGen_mode] if QGen_mode in ["jieba","lac"] else None 
        
        words1_QGen = data_prepare_config["words1_QGen"] if "words1_QGen" in data_prepare_config else False
        words2_QGen = data_prepare_config["words2_QGen"] if "words2_QGen" in data_prepare_config else False
        
        
        start_time = time.time()
        with open(sentences_path,'r') as f:
            atomicId_sentences_dict = json.load(f)
        if QGen_scores_path is not None:
            with open(QGen_scores_path,'r') as f:
                atomicId_QGenScore_seg_dict = json.load(f)

        atomicId_sentencesSelected_dict = {}
        
        if QGen_mode in ["jieba","lac"]:
            for atomic_id, sentence_list in tqdm(atomicId_sentences_dict.items()):
                sentence_score_dict = atomicId_QGenScore_seg_dict[atomic_id]  
                atomicId_sentencesSelected_dict[atomic_id] = self.get_sentences_selected(sentence_list, 
                                                                                    sentence_score_dict, 
                                                                                    max_len=max_input_len, 
                                                                                    original_order=original_order)
            
      
        elif QGen_mode == "no": 
            for atomic_id, sentence_list in tqdm(atomicId_sentences_dict.items()):
                sent_num = len(sentence_list)
                sentence_score_dict = {str(i):float(sent_num-i) for i in range(sent_num)} 
                atomicId_sentencesSelected_dict[atomic_id] = self.get_sentences_selected(sentence_list, 
                                                                                    sentence_score_dict, 
                                                                                    max_len=max_input_len, 
                                                                                    original_order=original_order) 
            
        elif QGen_mode == "title": 
            with open(data_prepare_config['title_path'],'r') as f:
                atomicId_title_dict = json.load(f)
            atomicId_sentencesSelected_dict = { id:[title] for id, title in atomicId_title_dict.items()}

        if words1_QGen or words2_QGen:
            with open(data_prepare_config["charges_path"],"r")as f:
                pid_charges = json.load(f)
            with open(data_prepare_config["total_crimes_path"],"r")as f:
                total_crimes = json.load(f)
            atomicId_words_dict = defaultdict(list)
            if words1_QGen:
                with open(data_prepare_config["crime1_words_path"],"r")as f:
                    crime_words_dict = json.load(f)
                k = data_prepare_config["words1_QGen_k"]
                all_words = list(set([word for charge in total_crimes for word in crime_words_dict[charge]]))
                for id_str,sent_list in atomicId_sentences_dict.items():
                    if id_str not in self.test_set:
                        charges = pid_charges[id_str]
                        words = list(set([word for charge in charges for word in crime_words_dict[charge]]))
                        orignal_s = "".join(sent_list)
                        counter = Counter()
                        for w in words:
                            counter[w] = orignal_s.count(w)
                        most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                        atomicId_words_dict[id_str] += most_common_words
                    else:
                        all_words = list(set([word for charge in total_crimes for word in crime_words_dict[charge]]))
                        # all_words =  list(set([word for ,words in crime1_words_dict.items() for word in words]))
                        
                        orignal_s = "".join(sent_list)
                        counter = Counter()
                        for w in all_words:
                            counter[w] = orignal_s.count(w)
                        most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                        atomicId_words_dict[id_str] += most_common_words
                
                
            if words2_QGen:
                with open(data_prepare_config["crime2_words_path"],"r")as f:
                    crime2_words_dict = json.load(f)
                k = data_prepare_config["words2_QGen_k"]
                if words1_QGen:
                    for c,words in crime2_words_dict.items():
                        crime2_words_dict[c] = list(set(words) - set(crime_words_dict[c]))
                crime_words_dict = crime2_words_dict
                all_words = list(set([word for charge in total_crimes for word in crime_words_dict[charge]]))

                for id_str,sent_list in atomicId_sentences_dict.items():
                    if id_str not in self.test_set:
                        charges = pid_charges[id_str]
                        words = list(set([word for charge in charges for word in crime_words_dict[charge]]))
                        orignal_s = "".join(sent_list)
                        counter = Counter()
                        for w in words:
                            counter[w] = orignal_s.count(w)
                        most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                        atomicId_words_dict[id_str] += most_common_words
                    else:
                # all_words =  list(set([word for ,words in crime1_words_dict.items() for word in words]))
                        orignal_s = "".join(sent_list)
                        counter = Counter()
                        for w in all_words:
                            counter[w] = orignal_s.count(w)
                        most_common_words = [word for word, times in counter.most_common(k) if times > 0]
                        atomicId_words_dict[id_str] += most_common_words
            
            for id_str,sent_list in atomicId_sentences_dict.items():
                atomicId_sentencesSelected_dict[atomic_id] = \
                                remove_duplicates(atomicId_words_dict[atomic_id]) + sentence_list    
     
        
        print("------TIME------->",time.time()-start_time)

        
        pad_0 = data_prepare_config["pad_0"] 
        id_internal_separator = data_prepare_config["id_internal_separator"] 
        
        self.docid_ty = data_prepare_config['docid_ty']
        if self.docid_ty in ["decode_tree","kmeans","decode_tree_kmeans","decode_tree_Ultron_1"]: 
            with open(data_prepare_config["docid_paths"][self.docid_ty]["toSeq"],'r') as f:
                self.atomicId_docid_dict = json.load(f)  
            # for k,v in self.atomicId_docid_dict.items():
            #     self.atomicId_docid_dict[k] = v.split("_")
            # get decode tree
            tree_path = data_prepare_config["docid_paths"][self.docid_ty]["decode_tree"]
            print("getting decode tree from {}".format(tree_path))
            with open(tree_path, 'rb') as f:
                self.root = pickle.load(f)

        elif self.docid_ty == "original_atomic": 
            self.atomicId_docid_dict = { atomic_id : atomic_id 
                        for atomic_id in atomicId_sentencesSelected_dict} 
        elif self.docid_ty == "atomic":
            # atomicId_docid_dict = { atomic_id : atomic_id 
            #                for atomic_id in atomicId_sentencesSelected_dict} 
            with open(data_prepare_config["docid_paths"]["random_seq"]["toSeq"],"r") as f:
                self.atomicId_docid_dict = json.load(f)
                
            step_length = {"atomic":'4'}
            for atomic_id, docid in self.atomicId_docid_dict.items():
                pad_docid = change_docid(docid,self.docid_ty,
                                        seperator=id_internal_separator,
                                        pad_0=pad_0,
                                        step_length=step_length)
                self.atomicId_docid_dict[atomic_id] = pad_docid
        
        elif self.docid_ty in ["random_exact",
                        "atomic_random_exact",
                        "atomic_tree_kmeans",
                        "atomic_tree_atomic",
                        "random_seq",
                        "tree_atomic",
                        "tree_kmeans",
                        "XingFaCrime_random_exact",
                        "atomic_XingFaCrime_random_exact",
                        "kmeans"]:
            with open(data_prepare_config["docid_paths"][self.docid_ty]["toSeq"],'r') as f:
                self.atomicId_docid_dict = json.load(f)  
            
            step_length = {
                "tree_atomic":"11212",
                "tree_kmeans":"112121",
                "random_seq":"4",
                "atomic_tree_atomic":"11212",
                "atomic_tree_kmeans":"112121",
                "atomic_random_exact":"4",
                "random_exact":"4",
                "atomic_XingFaCrime_random_exact":"1123",
                "XingFaCrime_random_exact":"1123",
                "kmeans":"222"
            }
            
            for atomic_id, docid in self.atomicId_docid_dict.items():
                pad_docid = change_docid(docid,self.docid_ty,
                                        seperator=id_internal_separator,
                                        pad_0=pad_0,
                                        step_length=step_length)
                self.atomicId_docid_dict[atomic_id] = pad_docid
                
            if data_prepare_config["get rid of the same prefix part"]:
                self.atomicId_docid_dict = get_rid_of_same_prefix_part(self.atomicId_docid_dict)

        elif self.docid_ty in ["CrimeTxt_QGenTxt_exact","CrimeTxt_kw_exact"]: 
            with open(data_prepare_config["docid_paths"][self.docid_ty]["toSeq"],'r') as f:
                self.atomicId_docid_dict = json.load(f) 
        elif self.docid_ty in ["CrimeTxt_random_exact"]: 
            with open(data_prepare_config["docid_paths"][self.docid_ty]["toSeq"],'r') as f:
                self.atomicId_docid_dict = json.load(f) 
        else:
            raise ValueError("Error: docid_ty=={}".format(self.docid_ty))
        
      
        ownTxt_to_ownDocId = defaultdict(list)
        ownTxt_to_relevantDocId = defaultdict(list)
        final_train_pair_data = {} # {atomic_id: [[text1,docid],[text2,docid],...]}
        extra_input = defaultdict(list) # add QG
        if "+qg" in data_prepare_config and data_prepare_config["+qg"]:              # add QG
            with open(data_prepare_config["+qg_path"],'r') as f:
                caseId_qgList_dict = json.load(f)
        
        for atomic_id in self.atomicId_labels_dict:
            if atomic_id in self.test_set: 
                continue
            sent_list = atomicId_sentencesSelected_dict[atomic_id]
            new_id = self.get_new_id(atomic_id)
            ownTxt_to_ownDocId[atomic_id].append([sent_list, new_id])
            if "+qg" in data_prepare_config and data_prepare_config["+qg"]: 
                num = len(caseId_qgList_dict[atomic_id])
                for j in range(1,1+num):
                    extra_input[atomic_id+" fakeQ{}".format(j)].append([[caseId_qgList_dict[atomic_id][j-1]],new_id]) # add QG  
        
        for atomic_id,relevant_atomicId_list in self.atomicId_labels_dict.items():
            if atomic_id in self.test_set: 
                continue
            for relevant_atomicId in relevant_atomicId_list:
                if relevant_atomicId in self.test_set: 
                    continue
                sent_list = atomicId_sentencesSelected_dict[atomic_id] 
                ownTxt_to_relevantDocId[atomic_id].append([sent_list, self.get_new_id(relevant_atomicId)])
                
        used_atomicIds1 = set(ownTxt_to_ownDocId.keys())
        used_atomicIds2 = set(ownTxt_to_relevantDocId.keys())
        check = len(used_atomicIds1) == len(used_atomicIds2)
        used_atomicIds = used_atomicIds1 | used_atomicIds2    

        ownTxt_to_ownDocId_coefficient,relevantTxt_to_ownDocId_coefficient = \
            data_prepare_config['data_coefficient']['ownTxt_to_ownDocId'],\
            data_prepare_config['data_coefficient']['relevantTxt_to_ownDocId']
        for atomic_id in used_atomicIds:
            final_train_pair_data[atomic_id] = ownTxt_to_ownDocId[atomic_id] * ownTxt_to_ownDocId_coefficient \
                                    + ownTxt_to_relevantDocId[atomic_id] * relevantTxt_to_ownDocId_coefficient
                                    

        final_train_list_data = {} # {atomic_id: [text, [docid1, docid2, ...]], ...}
        final_test_data = {} # {atomic_id: [text, [docid1, docid2, ...]], ...}
        
        for atomic_id in self.atomicId_labels_dict:
            if atomic_id in self.test_set: 
                final_test_data[atomic_id] = [
                    atomicId_sentencesSelected_dict[atomic_id], # sent list
                    [ self.get_new_id(relevant_atomic_id) # self.atomicId_docid_dict[relevant_atomic_id]
                        for relevant_atomic_id in self.atomicId_labels_dict[atomic_id]]
                ]
            else:
                final_train_list_data[atomic_id] = [
                    atomicId_sentencesSelected_dict[atomic_id], # sent list
                    [self.get_new_id(relevant_atomic_id) # self.atomicId_docid_dict[relevant_atomic_id] 
                        for relevant_atomic_id in self.atomicId_labels_dict[atomic_id]]
                ] 
        
        final_model_input = {}
        if seperate_train:
            final_model_input['train_pair'] = final_train_pair_data
            final_model_input['train_list'] = final_train_list_data
            # self.each_qurey_max_rl_times = None 
        else: 
            final_model_input['train'] = final_train_pair_data
        final_model_input['test'] = final_test_data

            

        return final_model_input, extra_input

    # def _random_sample_neg_labels(self, label_dict_hier, force_labels_size):
    #     '''
    #     Discarded....
    #     '''
    #     for atomic_id, label_list in label_dict_hier.items():
    #         if len(label_list) < force_labels_size:
    #             missing_num = force_labels_size - len(label_list)
    #             label_dict_hier[atomic_id] = label_list + random.sample(self.atomicIds, missing_num)
    #         else:
    #             label_dict_hier[atomic_id] = label_list[ : force_labels_size]
    #     return label_dict_hier

    def _truncate_cp(self, list_dict, force_list_size):
        # for a dict of list, truncate each list if ...
        # for k in list_dict:
        #     list_dict[k] = list_dict[k][:force_list_size]
        for key in list_dict:
            length = len(list_dict[key])
            if length > force_list_size:
                list_dict[key] = list_dict[key][:force_list_size]
            elif length < force_list_size:
                times = force_list_size // length
                remainder = force_list_size % length
                list_dict[key] = list_dict[key] * times + list_dict[key][:remainder]
        return list_dict
  
    
    
    ###### ELAM
    def get_new_id(self,old_id:str):
        try:
            new_id = self.atomicId_docid_dict[old_id]
        except KeyError:
            raise KeyError("Error: old_id({}) not in atomicId_docid_dict".format(old_id))
        return new_id

    def get_relevant_new_id_list(self, atomic_id:str):
        relevant_atomic_id_list = self.atomicId_labels_dict[atomic_id]
        relevant_new_id_list = [self.get_new_id(relevant_atomic_id) for relevant_atomic_id in relevant_atomic_id_list]
        return relevant_new_id_list
    
    ###### LeCaRDv2
    def get_new_multiIds(self,old_pid:str):
        try:
            new_ids = self.pid_docids_dict[old_pid]
        except KeyError:
            raise KeyError("Error: old_pid({}) not in atomicId_docid_dict".format(old_pid))
        return new_ids
    
    def get_relevant_new_multiIds_list(self, qid:str):
        relevant_pid_pair_list = self.qid_labels_dict_metrics[qid] # [[pid,rank],....]
        relevant_pid_list = [relevant_atomic_id[0] for relevant_atomic_id in relevant_pid_pair_list]
        relevant_new_pids_list = [self.get_new_multiIds(relevant_pid) for relevant_pid in relevant_pid_list]
        return relevant_new_pids_list # [[new_pid1,new_pid2,...],...]
    
    # def rlttpefeq(self):
    #     # ranking_loss_train_times_per_epoch_for_each_query 
    #     if hasattr(self,"each_qurey_max_rl_times"):
    #         if self.each_qurey_max_rl_times == None:
    #             print("\n    Seperate_train!   \n")
    #             return None
    #         return self.each_qurey_max_rl_times.copy()
    #     else:
    #         raise AttributeError("Error: no attribute each_qurey_max_rl_times, please prepare_data first!")
    
if __name__ == '__main__':
    
    pass

