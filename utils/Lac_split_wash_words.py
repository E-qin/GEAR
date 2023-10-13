import re
import pdb
from collections import defaultdict
from LAC import LAC
import json


stop_ty = ['r','PER','nr','ns','nt','ORG','LOC','TIME','w','m','q']

def remove_stop_ty(tagwordlist,stopkeys):
    res = []
    for w, ty,_rank in tagwordlist:
        if ty not in stop_ty:
            _stop = False 
            for stopkey in stopkeys:  
                if stopkey in w:
                    _stop = True
                    break
            if _stop == False:
                res.append((w,ty,_rank)) 
    return res


def lac_wash(text ,stopwords=[], stopkeys=[], lac=None, least_rank=2, DBG=False, collect=0, least_len=2):
    ''' QWC

    Args:
        text: One-dimensional list where each element is a sentence.
        stopwords: One-dimensional list where each element is a stopword.
        stopkeys: One-dimensional list where each element is a stop character.
        lac: LAC tokenizer instance. You can find it [here](https://gitcode.net/mirrors/baidu/lac?utm_source=csdn_github_accelerator).
        least_rank: Word importance level (rank) for segmentation.
        DBG: Whether to enable debugging.
        collect: Whether to collect word statistics based on part of speech and rank. 0 means no collection, 1 means collect.

    Return:
        list of words: [[w1, w2, ...], ...] Corresponding with text.

    '''
    words_list =[]
    leng = len(text)
    col1 = defaultdict(set)
    col2 = defaultdict(set)
    # col3 = defaultdict(set)
    # col4 = defaultdict(set)
    for i in range(leng):
        if text[i] == '': 
            words_list.append([])
            continue

        try:
            lac_result = lac.run(text[i])
            lac_result = list(zip(*lac_result))

            if collect==1:
                for w,ty,_rank in lac_result:
                    col1[ty].add(w)
                    col2[_rank].add(w)

            lac_result = [(w,ty,_rank) for w,ty,_rank in lac_result if (len(w)>=least_len and not re.match('^[a-z|A-Z|0-9|.]*$',w) and w not in stopwords )]
        except Exception as e:
            print("!!Exception!!") 
            print(str(e))
            print(text[i])
            print(lac.run(text[i]))
            print(lac_result)

        lac_result = remove_stop_ty(lac_result,stopkeys) 
        # if collect==1:
        #     for w,ty,_rank in lac_result:
        #         col3[ty].add(w)
        #         col4[_rank].add(w)

        words = [w for w,_,_rank in lac_result if _rank >= 2] 
        words_list.append(words)
        if DBG:
            print('{} : {}'.format(i,str(words)))
            pdb.set_trace()

    if collect>0:
        return words_list, col1, col2 #,col3, col4
    else:
        return words_list

        



# test: the using of lac_wash
if __name__ == '__main__':
    pass