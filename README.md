# Chinese punctuation prediction

The goal of this model is to predict suitable punctuation for the sentence without punctuation. We only predict four kinds of punctuation, that is "，", "。", "？", "、". You can think of this model as the pytorch version of this [repo](https://github.com/jiangnanboy/punctuation_prediction.git), which means our model is also built by Ernie. However, we use [ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh) as the pretained model instead of ernie-1.0.

# Dataset

We train our model in our internal data, but you could use the dataset in this [repo](https://github.com/jiangnanboy/punctuation_prediction.git) or your own data. Please put your dataset in `./dataset`.

Although our punctuation vocabulary is different from this [repo](https://github.com/jiangnanboy/punctuation_prediction.git), you can just change the punctuation vocabulary to run our codes successfully.

# How to train

After `python train.py`, the finetuned model would be saved in `./models` and the loss curve would be saved in `./fig`.

In `./models`,  it contains three kinds of model files, including train step checkpoints, epoch checkpoints and best model checkpoints.

# How to infer

### pytorch infer

`python infer.py`

### onnx infer:

First, you should export our model into onnx format with `python export_onnx.py`, and onnx model would be saved in `./onnx`.

Then, you could infer by onnx model with `python onnx_infer.py`.

### Infer demo

```bash
reference: 几千斤重的工程锤，威力巨大的加农炮，男人们想尽了各种方法，却都被聪明的女巫一一化解。
clean text: 几千斤重的工程锤威力巨大的加农炮男人们想尽了各种方法却都被聪明的女巫一一化解
infer result: 几千斤重的工程锤，威力巨大的加农炮，男人们想尽了各种方法，却都被聪明的女巫一一化解。
```

# Compare

We also compare our model with [CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary) built by Alibaba DAMO Academy. The evaluation metrics contain precision, recall and f1 score. Note that CT-Transformer was not finetuned in our dataset.

`python compare.py` 

Compare result:

```
CT-Transformer
comma| precision: 0.47, recall: 0.43, f1_score: 0.43
pause| precision: 0.01, recall: 0.01, f1_score: 0.01
question mark| precision: 0.11, recall: 0.11, f1_score: 0.11
period| precision: 0.30, recall: 0.33, f1_score: 0.31

Our model
comma| precision: 0.67, recall: 0.66, f1_score: 0.65
pause| precision: 0.02, recall: 0.02, f1_score: 0.02
question mark| precision: 0.11, recall: 0.11, f1_score: 0.11
period| precision: 0.22, recall: 0.22, f1_score: 0.22
```

This result shows that our model is better than CT-Transformer when punctuation is comma and pause, and they have same performence in question mark. However, our model is worse than CT-Transformer in period. Perhaps there are fewer examples with period in our dataset than the conterpart used by CT-Transformer. 

# Contact

Please feel free to contact me.  It's my honor to make progress with you.

You could open an issue or send an email to zhihaolancorrect@gmail.com.
