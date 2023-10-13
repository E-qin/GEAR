import numpy as np
import torch
from rouge import Rouge

rouge = Rouge()
# metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l', "acc_atomic",...]


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def compute_rouge(source, target, unit='word'):
    """ for rouge-1、rouge-2、rouge-l
    """
    # if unit == 'word':
    #     source = jieba.cut(source, HMM=False)
    #     target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """ all metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics

def rouge_1(source, target):
    source, target = ' '.join(source), ' '.join(target)
    scores = rouge.get_scores(hyps=source, refs=target)
    return scores[0]['rouge-1']['f']

def rouge_2(source, target):
    source, target = ' '.join(source), ' '.join(target)
    scores = rouge.get_scores(hyps=source, refs=target)
    return scores[0]['rouge-2']['f']

def rouge_l(source, target):
    source, target = ' '.join(source), ' '.join(target)
    scores = rouge.get_scores(hyps=source, refs=target)
    return scores[0]['rouge-l']['f']

def acc_atomic(source, target):
    cnt = 0
    for d1,d2 in zip (source,target):
        if d1 == d2:
            cnt += 1
    return cnt/len(source)
    # return scores[0]['rouge-l']['f']
    
# def hit_ratio(source:list, target:list):
#     hit_at_1 = 0.
#     hit_at_10 = 0.
#     hit_at_20 = 0.
#     hit_at_30 = 0.
    
#     return 



def _recall(target_lists:list,pred_lists:list,at_k:int):
    """QWC
    recall@k
    Parameters:
        target_lists: list of list of target str
        pred_lists: list of list of pred str
        at_k: top k of pred list are considered
    """
    def _recall_per_user(t_list,p_list):
        return len(set(t_list) & set(p_list))/len(t_list)
    assert len(target_lists) == len(pred_lists) # user num
    recall_value = 0.0
    for target_list,pred_list in zip(target_lists,pred_lists):
        recall_value += _recall_per_user(target_list, pred_list[:at_k])   
    return recall_value/len(target_lists)

def _MRR(target_lists:list,pred_lists:list,mode='multi_labels'):
    """QWC
    ref to https://blog.csdn.net/weixin_44110392/article/details/123319189
    
    Parameters:
        target_lists: list of list of target str
        pred_lists: list of list of pred str
        mode: 'multi_labels' or 'single_label'
    """
    assert len(target_lists) == len(pred_lists) # user num
    
    if mode=='multi_labels':    
        mrr = 0.0
        for t_list,p_list in zip(target_lists,pred_lists):
            t_set=set(t_list)
            p_true = [1  if p in t_set else 0 for p in p_list]
            if sum(p_true) == 0:
                continue
            rr_score = p_true / (np.arange(len(p_true)) + 1)
            mrr += np.sum(rr_score) / np.sum(p_true) 
        return mrr/len(target_lists)
    elif mode=='single_label':
        mrr = 0.0
        for t_list,p_list in zip(target_lists,pred_lists):
            # assert len(t_list) == 1
            target = t_list[0]
            if target not in p_list:
                continue
            mrr += 1/(p_list.index(target)+1)
        return mrr/len(target_lists)
    else:
        raise ValueError('ERROR: MRR mode must be multi_labels or single_label')

def _hit_ratio(target_lists:list,pred_lists:list,at_k:int):
    '''QWC
    hit_ratio@k
    
    Parameters:
        target_lists: list of list of target str
        pred_lists: list of list of pred str
        at_k: top k of pred list are considered
    '''
    def _indicator(x:bool):
        return int(x) # return 1 if x is True else 0
    def _hit_per_user(t_list,p_list):
        return _indicator( len(set(t_list) & set(p_list)) > 0 )
    
    assert len(target_lists) == len(pred_lists) # user num
    hr_value = 0.0
    for target_list,pred_list in zip(target_lists,pred_lists):
        hr_value += _hit_per_user(target_list, pred_list[:at_k])   
    return hr_value/len(target_lists)


# ##########################################################################################
# # special 
# ##########################################################################################

# def coverage(target_lists:list,pred_lists:list,at_k:int):
#     '''QWC
#     coverage
#     '''
#     pass
#     # def _coverage_per_user(t_list,p_list):
#     #     return len(set(t_list) & set(p_list))/len(set(t_list))
#     # assert len(target_lists) == len(pred_lists) # user num
#     # coverage_value = 0.0
#     # for target_list,pred_list in zip(target_lists,pred_lists):
#     #     coverage_value += _coverage_per_user(target_list, pred_list)   
#     # return coverage_value/len(target_lists)





##############################################################
# optional eval-functions used 
##############################################################

def recall_at_1(target_lists:list, pred_lists:list):
    return _recall(target_lists, pred_lists, 1)

def recall_at_5(target_lists:list, pred_lists:list):
    return _recall(target_lists, pred_lists, 5)

def recall_at_10(target_lists:list, pred_lists:list):
    return _recall(target_lists, pred_lists, 10)

def recall_at_20(target_lists:list, pred_lists:list):
    return _recall(target_lists, pred_lists, 20)    
    
def recall_at_30(target_lists:list, pred_lists:list):
    return _recall(target_lists, pred_lists, 30)  

def MRR_mls(target_lists:list, pred_lists:list):
    return _MRR(target_lists, pred_lists, 'multi_labels')

def MRR_sl(target_lists:list, pred_lists:list):
    return _MRR(target_lists, pred_lists, 'single_label')

def hit_at_1(target_lists:list, pred_lists:list):
    return _hit_ratio(target_lists, pred_lists, 1)

def hit_at_5(target_lists:list, pred_lists:list):
    return _hit_ratio(target_lists, pred_lists, 5)

def hit_at_10(target_lists:list, pred_lists:list):
    return _hit_ratio(target_lists, pred_lists, 10)

def hit_at_20(target_lists:list, pred_lists:list):
    return _hit_ratio(target_lists, pred_lists, 20)

def hit_at_30(target_lists:list, pred_lists:list):
    return _hit_ratio(target_lists, pred_lists, 30)
