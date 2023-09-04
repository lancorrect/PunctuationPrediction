import argparse
import time
import os 
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import ErnieForTokenClassification

from logger import setup_logger
cur_time = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
log_dir = './log/' + cur_time + '.log'
logger = setup_logger(__name__, output=log_dir)

from data_utils import PuncDataset, PuncCollate
from trainer import Trainer

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--model_name', default='nghuyong/ernie-3.0-base-zh', type=str, required=False, help='ernie模型名称')
    parser.add_argument('--train_data_path', default='./short_videos/train.json', type=str, required=False, help='原始训练语料路径')
    parser.add_argument('--dev_data_path', default='./short_videos/dev.json', type=str, required=False, help='验证集路径')
    parser.add_argument('--test_data_path', default='./dataset/test.txt', type=str, required=False, help='测试集路径')
    parser.add_argument('--punc_path', default='./short_videos/punc_vocab_golden', type=str, required=False, help='标点符号字典路径')
    parser.add_argument('--max_len', default=256, type=int, required=False, help='每条数据最大长度')
    parser.add_argument('--cache_data_path', default='./cache_self', type=str, required=False, help='预处理后语料存放位置')

    # train args
    parser.add_argument('--epochs', default=1, type=int, required=False, help='训练代数')
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.0e-5, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')
    
    # save args
    parser.add_argument('--output_dir', default='./models_test/', type=str, required=False, help='模型保存路径')
    parser.add_argument('--plot_dir', default='./fig/', type=str, required=False, help='损失曲线保存路径')
    parser.add_argument('--save_step', default=1000, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')

    args = parser.parse_args()
    args.cur_time = cur_time  # time tag for folder name
    args.output_dir = Path(args.output_dir) / args.cur_time
    logger.info('args:\n' + args.__repr__())

    logger.info(f'loading dataset')
    if not Path(args.cache_data_path).exists():
        Path(args.cache_data_path).mkdir(parents=True, exist_ok=True)
    train_dataset = PuncDataset(args, args.train_data_path)
    dev_dataset = PuncDataset(args, args.dev_data_path)
    # test_dataset = PuncDataset(args, args.test_data_path)
    collate_fn = PuncCollate(args.max_len, train_dataset.tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    logger.info(f'loading model')
    model = ErnieForTokenClassification.from_pretrained(args.model_name, num_labels=len(train_dataset.punc2id))
    model.cuda()

    trainer = Trainer(args, model, train_loader, dev_loader)
    trainer.train()
    # trainer.evaluate(split='test')