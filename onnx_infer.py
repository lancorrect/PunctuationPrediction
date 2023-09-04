import json
from pathlib import Path
import time
from tqdm import tqdm
import argparse
import re
import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer


class Onnx_Inferer:
    def __init__(self, args):
        self.punc2id = self.load_vocab(args.punc_path, extra_word_list=[" "]) # 标点字典中没有空格
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}
        
        self.punc_result_dir = args.punc_result_dir
        args.punc_result_dir.mkdir(parents=True, exist_ok=True)

        print("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
        
        print("loading ernie finetune onnx model")
        ernie_ft = onnx.load_model(args.model_path)
        self.sess = ort.InferenceSession(bytes(ernie_ft.SerializeToString()), providers=['CPUExecutionProvider'])
        
    @staticmethod
    def load_vocab(vocab_path, extra_word_list=[]):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    # 预处理文本
    def preprocess(self, text: str):
        text = re.sub(r'#.*?#', '', text)
        clean_text=re.sub(r'[\s+\/_$%^*(+\"\'):-·!?,.]+|[+——()？！。，、《》【】“”~@#￥%……&*（）：-]+', '', text)
        if len(clean_text) == 0: return None, None
        
        input_ids = self.tokenizer.convert_tokens_to_ids(list(clean_text))
        input_ids = np.array([input_ids], dtype=np.int64)
        
        return clean_text, input_ids

    def onnx_infer(self, input_ids: list):
        preds = self.sess.run(
            output_names=None,
            input_feed={"input": input_ids}
        )[0]
        preds = np.argmax(preds, axis=-1)
        return preds

    # 后处理识别结果
    def postprocess(self, input_ids, preds):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = preds[0].tolist()
        assert len(tokens) == len(labels)

        text = ''
        for t, l in zip(tokens, labels):
            text += t
            if l != 0:
                text += self.id2punc[l]
        return text

    def __call__(self, input_data) -> str:
        if isinstance(input_data, str):
            print(f"reference: {input_data}")

            clean_text, input_ids = self.preprocess(input_data)
            print(f"clean text: {clean_text}")
            if input_ids is None: 
                print("text is empty after preprocessing")
                return None
            
            start = time.time()
            preds = self.onnx_infer(input_ids)
            then = time.time()
            # print("Time required to infer once: {:.3f}".format(then-start))
            
            text = self.postprocess(input_ids, preds)
            print(f"onnx infer result: {text}")
        else:
            raise NotImplementedError
        
        # 数据batch处理
        # if isinstance(input_data, str):
        #     text = input_data
        #     clean_text, input_ids, label = self.preprocess(text)
        #     preds = self.infer(input_ids=input_ids)
        #     if len(preds.shape) == 2:
        #         preds = preds[0]
        #     output = self.postprocess(input_ids, preds)
        # else:
        #     output = []
        #     print('Start Infering')
        #     for display, info in tqdm(input_data.items()):
        #         text = info['text']
        #         if len(text) <= 3: continue  # 文本长度小于等于3的就不打上标点了
        #         output_dict = {'utt':info['utt'], 'origin':text}

        #         clean_text, input_ids, label = self.preprocess(text)
        #         if input_ids is None : continue  # 清洗完文本成空的了就跳过
        #         # preds = self.infer(input_ids=input_ids)
        #         preds_ft, preds_pt, preds_ct = self.infer_multi_model(clean_text, input_ids)
        #         # if len(preds.shape) == 2:
        #         #     preds = preds[0]
        #         output_ft = self.postprocess(input_ids, preds_ft)
        #         output_pt = self.postprocess(input_ids, preds_pt)

        #         output_dict['label'] = label
        #         output_dict['ernie-pretrain'] = output_pt
        #         output_dict['ernie-finetune'] = output_ft
        #         output_dict['ct-transformers'] = preds_ct
                
        #         output.append(output_dict)
        #     output_file = self.args.punc_result_dir / 'infer_result.json'
        #     print('Infer over and start saving')
        #     with open(output_file, 'w', encoding='utf-8') as fout:
        #         json.dump(output, fout, indent=4, ensure_ascii=False)
        #         fout.close()
        #     print('Saving completed')
        # return output


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # data args
    # parser.add_argument('--dev_data_path', default='/data_sunty/wangzh/share/short_videos_d0002.json', type=str, required=False, help='验证集路径')
    parser.add_argument('--punc_path', default='./short_videos/punc_vocab_golden', type=str, required=False, help='标点符号字典路径')

    # model args
    parser.add_argument('--model_path', default='./onnx/ernie_ft.onnx', type=str, required=False, help='ernie模型名称')

    # infer args
    parser.add_argument('--punc_result_dir', default='./punc_result', type=str, required=False, help='打标点后的结果')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    args.punc_result_dir = Path(args.punc_result_dir) / 'onnx'

    sent = '几千斤重的工程锤，威力巨大的加农炮，男人们想尽了各种方法，却都被聪明的女巫一一化解。'
    
    print("start infer")
    onnx_inferer = Onnx_Inferer(args)
    result = onnx_inferer(sent)
    print("infer over")
