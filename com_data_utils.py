import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

class EvalDataset:
    def __init__(self, args, data_path):
        split_name = Path(data_path).stem
        self.cache_data_path = Path(args.cache_data_path) / (split_name + '.json')
        self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
        self.max_len = args.max_len
        self.punc2id = self.load_vocab(args.punc_path, extra_word_list=[" "])

        with open(self.cache_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.ori = data['ori']
            self.data = data['inputs_data']
            self.labels = data['labels']
            f.close()
    
    @staticmethod
    def load_vocab(vocab_path, extra_word_list=[]):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_select = self.data[index][:self.max_len]
        label_select = self.labels[index][:self.max_len]
        return data_select, label_select

class EvalCollate:
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

class AlibabaDataset:
    def __init__(self, args, data_path):
        self.punc2id = self.load_vocab(args.punc_path, extra_word_list=[" "])

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
        with open(data_path, 'r', encoding='utf-8') as fin:
            txt_seqs = json.load(fin)
            fin.close()
        for text in txt_seqs:
            self.ori.append(text)
            txt = list(text.replace('\n', ''))
            if txt[-1] not in self.punc2id.keys(): txt += ' '
            label, input_data = [], []
            for i in range(len(txt) - 1):
                # 获取输入数据
                word = txt[i]
                if word in self.punc2id.keys():
                    continue
                token = word
                input_data += [token]
                # 获取标签数据
                punc = txt[i + 1]
                for _ in range(len([token]) - 1):
                    label.append(self.punc2id[" "])
                if punc not in self.punc2id:
                    label.append(self.punc2id[" "])
                else:
                    label.append(self.punc2id[punc])
            if len(input_data) != len(label):
                continue
            self.data.append(''.join(input_data))
            self.labels.append(label)
        # data = {'inputs_data': self.data, 'labels': self.labels, 'ori': self.ori}
        # with open(self.cache_data_path, 'w', encoding='utf-8') as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)

        if len(self.data) != len(self.labels):
            assert 'error: length input_data != label'

def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def post_process(text, punc2id):
    pred = []
    if text[-1] not in punc2id.keys(): text += ' '
    for idx in range(len(text)-1):
        word = text[idx]
        if word in punc2id.keys():
            continue
        next_token = text[idx+1]
        label = punc2id.get(next_token, 0)
        pred.append(label)
    return pred