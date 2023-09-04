import json
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import unicodedata
import re

# s = "您好，欢迎来到我的博客：https://blog.csdn.net/weixin_44799217,,,###,,,我的邮箱是：535646343@qq.com. Today is 2021/12/21. It is a wonderful DAY!"
# s = re.sub(r'[\.\!\/_,$%^*(+\"\'):-]+|[+——()?【】“”！，。？、~@#￥%……&*（）：-]+', '+', s)
# print(s)

# with open('/home/wangsh/workspace/wzh/punc/punc_self/punc_result/2023-08-31_17:05:18/infer_result.json', 'r') as fin:
#     infer_result = json.load(fin)
#     fin.close()

# new_infer_result = []
# for result_unit in tqdm(infer_result):
#     utt = result_unit['utt']
#     ref = result_unit['origin']
#     ernie = result_unit['ernie-finetune']
#     ct = result_unit['ct-transformers']
#     new_infer_result.append({'utt':utt, 'renference':ref, 'ernie-finetune':ernie, 'ct-transformers':ct})
# # print(new_infer_result)

# with open('/home/wangsh/workspace/wzh/punc/punc_self/punc_result/2023-08-31_17:05:18/new_infer_result.json', 'w', encoding='utf-8') as fout:
#     json.dump(new_infer_result, fout, indent=4, ensure_ascii=False)
#     fout.close()

# from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# preds = [0, 1, 2, 1, 0, 2, 0, 2, 0, 0, 2, 1, 2, 0, 1]
# trues = [0, 1, 2, 1, 1, 2, 0, 2, 0, 0, 2, 1, 2, 1, 2]

# # labels为指定标签类别，average为指定计算模式

# # macro-precision
# macro_p = precision_score(trues, preds, labels=[1], average='macro')
# # macro-recall
# macro_r = recall_score(trues, preds, labels=[1], average='macro')
# # macro f1-score
# macro_f1 = f1_score(trues, preds, labels=[1], average='macro')

# print(macro_p, macro_r, macro_f1) # 0.8055555555555555 0.8111111111111112 0.7919191919191918
# # print(macro_p)

a = {'a':1, 'b':2, 'c':3}
print(a.setdefault('a', 10))