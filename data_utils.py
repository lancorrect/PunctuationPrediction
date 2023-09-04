import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from logger import setup_logger

logger = setup_logger(__name__)

class PuncDataset:
    def __init__(self, args, data_path):
        split_name = Path(data_path).stem
        self.cache_data_path = Path(args.cache_data_path) / (split_name + '.json')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_len = args.max_len
        self.punc2id = self.load_vocab(args.punc_path, extra_word_list=[" "])
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        self.preprocess(data_path)
    
    @staticmethod
    def load_vocab(vocab_path, extra_word_list=[]):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab
    
    def preprocess(self, data_path):
        self.ori = []
        self.data = []
        self.labels = []
        if not self.cache_data_path.exists():
            logger.info(f'{self.cache_data_path}不存在，正在重新生成，时间比较长，请耐心等待...')
            with open(data_path, 'r', encoding='utf-8') as fin:
                txt_seqs = json.load(fin)
                fin.close()
            # txt_seqs = open(data_path, encoding='utf-8').readlines()
            indices = list(range(len(txt_seqs)))
            random.shuffle(indices)
            for idx in tqdm(indices):
                text = txt_seqs[idx]
                self.ori.append(text)
                txt = list(text.replace('\n', ''))  # 在中文中，如果想像英文一样把每个字分开，必须使用list()，使用split()没有效果
                if txt[-1] not in self.punc2id.keys(): txt += ' '
                label, input_data = [], []
                for i in range(len(txt) - 1):
                    # 获取输入数据
                    word = txt[i]
                    if word in self.punc2id.keys():
                        continue  # repo里的数据集是两个字中间有空格，所以如果碰到空格或者标点的话直接跳过，因为这不是要输入的内容
                    token = self.tokenizer.convert_tokens_to_ids(word)
                    input_data += [token]
                    # 获取标签数据
                    punc = txt[i + 1]
                    for _ in range(len([token]) - 1):
                        label.append(self.punc2id[" "])  # 一个字在经过tokenizer后可能由很多个元素组成，除了最后一个位置，其他都填上空格，表示没有标点
                    if punc not in self.punc2id:
                        label.append(self.punc2id[" "])  # 如果该字后面没有标点，则标签在该位置上直接填上空格对应的id
                    else:
                        label.append(self.punc2id[punc])  # 如果该字后面有标点，则标签在位置上直接填上标点对应的id
                if len(input_data) != len(label):
                    continue
                self.data.append(input_data)
                self.labels.append(label)
            data = {'inputs_data': self.data, 'labels': self.labels, 'ori': self.ori}
            with open(self.cache_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            logger.info(f'正在加载：{self.cache_data_path}')
            # 读取之前制作好的数据，如果是更换了数据集，需要删除这几个缓存文件
            with open(self.cache_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data = data['inputs_data']
                self.labels = data['labels']

        if len(self.data) != len(self.labels):
            assert 'error: length input_data != label'

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_select = self.data[index][:self.max_len]
        label_select = self.labels[index][:self.max_len]
        return data_select, label_select

class PuncCollate:
    def __init__(self, max_len, pad_id):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch):
        sents = []
        labels = []
        attention_masks = []

        lengths = [len(t) for t,_ in batch]
        max_length = min(max(lengths), self.max_len)

        for sent,label in batch:
            sent_len = len(sent)
            pad_len = max_length - sent_len
            new_sent = sent + [self.pad_id] * pad_len
            new_label = label + [self.pad_id] * pad_len
            attention_mask = [1] * sent_len + [0] * pad_len

            sents.append(new_sent)
            labels.append(new_label)
            attention_masks.append(attention_mask)
        
        sents = torch.tensor(sents, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        return sents, labels, attention_masks