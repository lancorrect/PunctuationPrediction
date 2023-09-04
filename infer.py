import json
from pathlib import Path
import time
import torch
from transformers import ErnieForTokenClassification, AutoTokenizer
from tqdm import tqdm
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import re

class Inferer:
    def __init__(self, args, dev_dataset):
        self.args = args
        self.dataset = dev_dataset
        self.punc2id = self.load_vocab(args.punc_path, extra_word_list=[" "]) # 标点字典中没有空格
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        print("loading ernie finetune model")
        self.ernie_ft = ErnieForTokenClassification.from_pretrained(Path(args.model_path)/'checkpoint', num_labels=len(self.punc2id))
        self.ernie_ft.cuda()
        
        print("loading ernie pretrain model")
        self.ernie_pt = ErnieForTokenClassification.from_pretrained('nghuyong/ernie-3.0-base-zh', num_labels=len(self.punc2id))
        self.ernie_pt.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
        
        # self.ct = None
        print("loading ct-transformers")
        self.ct = inference_pipeline = pipeline(
            task=Tasks.punctuation,
            model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            model_revision="v1.1.7",
        )

        args.punc_result_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_vocab(vocab_path, extra_word_list=[]):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab
    
    def _clean_text(self, text):
        text = re.sub(r'#.*?#', '', text)
        text_with_punc=re.sub(r'[\s+\/_$%^*(+\"\'):-·!?,.]+|[+——()！《》【】“”~@#￥%……&*（）：-]+', '', text)
        text=re.sub(r'[\s+\/_$%^*(+\"\'):-·!?,.]+|[+——()？！。，、《》【】“”~@#￥%……&*（）：-]+', '', text)
        return text_with_punc, text

    # 预处理文本
    def preprocess(self, text: str):
        text_with_punc, clean_text = self._clean_text(text)
        if len(clean_text) == 0: return None, None, None
        
        label = []
        punc_num = 0
        if text_with_punc[-1] not in self.punc2id: text_with_punc += ' '
        for i in range(len(text_with_punc)-1):
            punc = text_with_punc[i+1]
            if punc not in self.punc2id:
                label.append(' ')
            else:
                punc_num += 1
                label.append(punc)

        input_ids = self.tokenizer.convert_tokens_to_ids(list(clean_text))
        # assert len(text_with_punc) - len(input_ids) == punc_num
        return clean_text, input_ids, label

    def infer(self, model, input_ids: list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(input_ids=input_ids)
            logits = output.logits
        
        preds = torch.argmax(logits, axis=-1).squeeze(0)
        return preds.cpu().numpy()

    def infer_multi_model(self, clean_text, input_ids):
        # ernie finetune result
        preds_ft = self.infer(self.ernie_ft, input_ids)

        # ernie pretrain result
        preds_pt = self.infer(self.ernie_pt, input_ids)

        # ct-transformers result
        preds_ct = self.ct(text_in=clean_text)['text']

        return preds_ft, preds_pt, preds_ct

    # 后处理识别结果
    def postprocess(self, input_ids, preds):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        labels = preds.tolist()
        assert len(tokens) == len(labels)

        text = ''
        for t, l in zip(tokens, labels):
            text += t
            if l != 0:
                text += self.id2punc[l]
        return text

    def __call__(self, input_data) -> str:
        # 数据batch处理
        if isinstance(input_data, str):
            text = input_data
            clean_text, input_ids, label = self.preprocess(text)
            preds = self.infer(input_ids=input_ids)
            if len(preds.shape) == 2:
                preds = preds[0]
            output = self.postprocess(input_ids, preds)
        else:
            output = []
            print('Start Infering')
            for display, info in tqdm(input_data.items()):
                text = info['text']
                if len(text) <= 3: continue  # 文本长度小于等于3的就不打上标点了
                output_dict = {'utt':info['utt'], 'origin':text}

                clean_text, input_ids, label = self.preprocess(text)
                if input_ids is None : continue  # 清洗完文本成空的了就跳过
                # preds = self.infer(input_ids=input_ids)
                preds_ft, preds_pt, preds_ct = self.infer_multi_model(clean_text, input_ids)
                # if len(preds.shape) == 2:
                #     preds = preds[0]
                output_ft = self.postprocess(input_ids, preds_ft)
                output_pt = self.postprocess(input_ids, preds_pt)

                output_dict['label'] = label
                output_dict['ernie-pretrain'] = output_pt
                output_dict['ernie-finetune'] = output_ft
                output_dict['ct-transformers'] = preds_ct
                
                output.append(output_dict)
            output_file = self.args.punc_result_dir / 'infer_result.json'
            print('Infer over and start saving')
            with open(output_file, 'w', encoding='utf-8') as fout:
                json.dump(output, fout, indent=4, ensure_ascii=False)
                fout.close()
            print('Saving completed')
        return output


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--model_name', default='ernie', type=str, required=False, help=['ernie', 'ali'])
    parser.add_argument('--model_path', default='./models/2023-08-31_17:05:18', type=str, required=False, help='ernie模型名称')
    parser.add_argument('--dev_data_path', default='/data_sunty/wangzh/share/short_videos_d0002.json', type=str, required=False, help='验证集路径')
    parser.add_argument('--punc_path', default='./short_videos/punc_vocab_golden', type=str, required=False, help='标点符号字典路径')

    # infer args
    parser.add_argument('--punc_result_dir', default='./punc_result', type=str, required=False, help='打标点后的结果')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    args.punc_result_dir = Path(args.punc_result_dir) / 'pytorch_model_' + Path(args.model_path).name

    print("loading dev dataset")
    with open(args.dev_data_path, 'r', encoding='utf-8') as fin:
        dev_dataset = json.load(fin)
        fin.close()
    
    inferer = Inferer(args, dev_dataset)
    result = inferer(dev_dataset)
    if isinstance(result, str):
        print(result)
