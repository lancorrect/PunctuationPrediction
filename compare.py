import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ErnieForTokenClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import argparse
import warnings
warnings.filterwarnings("ignore")  # 忽视计算精度时的UndefinedMetricWarning
from com_data_utils import EvalDataset, EvalCollate, AlibabaDataset, post_process
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

punc_name_dict = {'，':'comma', '、':'pause', '？':'question mark', '。':'period'}

def eval_metrics(labels, pred, id2punc):
    eval_result = {}
    if isinstance(labels, list):
        pass
    else:
        labels = labels.cpu().numpy().tolist()
        pred = pred.cpu().numpy().tolist()
    
    for punc_id, punc in id2punc.items():
        if punc_id == 0:
            continue  # 不考虑空格
        f1 = f1_score(labels, pred, labels=[punc_id],average="macro")
        recall = recall_score(labels, pred, labels=[punc_id],average="macro")
        precision = precision_score(labels, pred, labels=[punc_id],average="macro")
        punc_name = punc_name_dict[punc]
        eval_result[punc_name] = {'precision': precision, 'recall':recall, 'f1 score':f1}
    
    return eval_result

def average_metrics_by_punc(eval_punc):
    eval_pre = []
    eval_re = []
    eval_f1 = []

    for punc_metrics in eval_punc:
        pre, re, f1 = punc_metrics.values()
        eval_pre.append(pre)
        eval_re.append(re)
        eval_f1.append(f1)
    
    eval_precision_scores_avg = sum(eval_pre) / len(eval_pre)
    eval_recall_scores_avg = sum(eval_re) / len(eval_re)
    eval_f1_scores_avg = sum(eval_f1) / len(eval_f1)

    return eval_precision_scores_avg, eval_recall_scores_avg, eval_f1_scores_avg

def evaluate_ernie(data_loader, model, punc2id):
    eval_punc = {}
    
    id2punc = {k: v for (v, k) in punc2id.items()}
    device = model.device
    model.eval()
    num = 0

    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_masks = attention_masks.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_masks, 
                                labels=labels)
            loss, logits = output[:2]
            pred = torch.argmax(logits, axis=-1).reshape(-1)

            labels = labels.reshape(-1)
            all_punc_metrics = eval_metrics(labels, pred, id2punc)

            for punc_name in punc_name_dict.values():
                cur_punc_result = eval_punc.setdefault(punc_name, [])
                cur_punc_result.append(all_punc_metrics[punc_name])
            
    print("ernie\'s result")
    output_str_list = []
    for punc_name, metrics in eval_punc.items():
        eval_precision_scores_avg, eval_recall_scores_avg, eval_f1_scores_avg = average_metrics_by_punc(metrics)
        output_str = "{}| precision: {:.2f}, recall: {:.2f}, f1_score: {:.2f}".format(punc_name,
                                                                            eval_precision_scores_avg,
                                                                            eval_recall_scores_avg,
                                                                            eval_f1_scores_avg)
        print(output_str)
        output_str_list.append(output_str)
    with open('./log/ernie_eval_metrics.txt', 'w', encoding='utf-8') as fout:
        for out_str in output_str_list:
            fout.write(out_str+'\n')
        fout.close()

def evaluate_ali(data, labels, model, punc2id):
    eval_punc = {}
    id2punc = {k: v for (v, k) in punc2id.items()}
    num = 0
    
    for idx in tqdm(range(len(data))):
        text = data[idx]
        label = labels[idx]

        ali_result = model(text_in=text)['text']
        pred = post_process(ali_result, punc2id)

        all_punc_metrics = eval_metrics(label, pred, id2punc)

        for punc_name in punc_name_dict.values():
            cur_punc_result = eval_punc.setdefault(punc_name, [])
            cur_punc_result.append(all_punc_metrics[punc_name])

    print("ct-transformers\'s result")
    output_str_list = []
    for punc_name, metrics in eval_punc.items():
        eval_precision_scores_avg, eval_recall_scores_avg, eval_f1_scores_avg = average_metrics_by_punc(metrics)
        output_str = "{}| precision: {:.2f}, recall: {:.2f}, f1_score: {:.2f}".format(punc_name,
                                                                            eval_precision_scores_avg,
                                                                            eval_recall_scores_avg,
                                                                            eval_f1_scores_avg)
        print(output_str)
        output_str_list.append(output_str)
    with open('./log/ali_eval_metrics.txt', 'w', encoding='utf-8') as fout:
        for out_str in output_str_list:
            fout.write(out_str+'\n')
        fout.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # data args
    # parser.add_argument('--model_name', default='ct', type=str, required=False, help='ernie模型名称')
    parser.add_argument('--model_path', default='./models/2023-08-31_17:05:18/checkpoint', type=str, required=False, help='ernie模型名称')
    parser.add_argument('--train_data_path', default='./short_videos/train.json', type=str, required=False, help='原始训练语料路径')
    parser.add_argument('--dev_data_path', default='./short_videos/dev.json', type=str, required=False, help='验证集路径')
    parser.add_argument('--test_data_path', default='./dataset/test.txt', type=str, required=False, help='测试集路径')
    parser.add_argument('--punc_path', default='./short_videos/punc_vocab_golden', type=str, required=False, help='标点符号字典路径')
    parser.add_argument('--max_len', default=256, type=int, required=False, help='每条数据最大长度')
    parser.add_argument('--cache_data_path', default='./cache_self', type=str, required=False, help='预处理后语料存放位置')

    # train args
    parser.add_argument('--epochs', default=30, type=int, required=False, help='训练代数')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.0e-5, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=5, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default='./models/', type=str, required=False, help='模型保存路径')
    parser.add_argument('--plot_dir', default='./fig/', type=str, required=False, help='损失曲线保存路径')
    parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    print("loading dev dataset")
    dev_dataset = EvalDataset(args, args.dev_data_path)
    collate_fn = EvalCollate(args.max_len, dev_dataset.tokenizer.pad_token_id)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print("loading ernie model")
    model = ErnieForTokenClassification.from_pretrained(args.model_path, num_labels=len(dev_dataset.punc2id))
    model.cuda()
    
    # ernie pretrain| precision: 0.20, recall: 0.17, f1_score: 0.07
    # ernie finetune
    evaluate_ernie(dev_loader, model, dev_dataset.punc2id)

    print("loading dev dataset again")
    alibaba_dataset = AlibabaDataset(args, args.dev_data_path)
    dev_ori = alibaba_dataset.ori
    dev_dataset = alibaba_dataset.data
    dev_labels = alibaba_dataset.labels
    punc2id = alibaba_dataset.punc2id

    inference_pipeline = pipeline(
        task=Tasks.punctuation,
        model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        model_revision="v1.1.7",
        device='cuda:0'
    )

    # precision: 0.63, recall: 0.62, f1_score: 0.62
    evaluate_ali(dev_dataset, dev_labels, inference_pipeline, punc2id)