import torch
from transformers import ErnieForTokenClassification, AutoTokenizer

text = "我听说你是农村出来的孩子"
tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
tokens = tokenizer.convert_tokens_to_ids(list(text))
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
print(tokens)

print('loading saved model')
ernie = ErnieForTokenClassification.from_pretrained('./models/2023-08-31_17:05:18/checkpoint', num_labels=5)

print("start export")
torch.onnx.export(ernie, tokens, "./onnx/ernie_ft.onnx",opset_version=10,
                    input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {1: 'seq_len'}})  # 第一维可变，第0维默认维batch
print("export ends")
