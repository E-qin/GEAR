import re
import jieba.posseg as psg


stop_ty = ['r','nr','ns','t','w','m','q','un']
def remove_stop_ty(tagwordlist,stopkeys):
    res = []
    for w, ty in tagwordlist:
        if ty not in stop_ty:
            _stop = False 
            for stopkey in stopkeys:  
                if stopkey in w:
                    _stop = True
                    break
            if _stop == False:
                res.append((w,ty)) 
    return res



def jieba_wash(text ,stopwords=[], stopkeys=[],least_len=2):
    ''' QWC
    jieba_wash: segment, clean each sentence in Chinese.

    Args:
        text: One-dimensional list where each element is a sentence.
        stopwords: One-dimensional list where each element is a stopword.
        stopkeys: One-dimensional list where each element is a stop character.

    Return:
        list of words: [[w1, w2, ...], ...] Corresponding to text one by one.

    '''
    words_list =[]
    leng = len(text)
    for i in range(leng):
        if text[i] == '': 
            words_list.append([])
            continue

        try:
            psg_result = psg.cut(text[i])
            
            psg_result = [(w,ty) for w,ty in psg_result if (len(w)>=least_len and 
                                    not re.match('^[a-z|A-Z|0-9|.]*$',w) and w not in stopwords )]

            psg_result = [(w,ty) for w,ty in psg_result if not re.match('[=,.?!@#$%^&*()_+:"<>/\[\]\\`~——，。、《》？；’：“【】、{}|·！￥…（）-]', w)]

        except Exception as e:
            print("!!Exception!!") 
            print(str(e))
            print(text[i])
            print(psg.cut(text[i]))
            print(psg_result)

        psg_result = remove_stop_ty(psg_result,stopkeys)

        words = [w for w,_ in psg_result] 
        words_list.append(words)
    return words_list