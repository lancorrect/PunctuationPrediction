import torch
from logger import setup_logger
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from copy import deepcopy
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from matplotlib import pyplot as plt

logger = setup_logger(__name__)

class Trainer:
    def __init__(self, args, model, train_loader, dev_loader, test_loader=None):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
        self.optimizer.zero_grad()

        total_steps = int(len(train_loader) * args.epochs / args.gradient_accumulation)
        logger.info(f"total train steps are {total_steps}")
        self.scheduler_linear = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = int(total_steps * 0.3), num_training_steps = total_steps)
        # self.scheduler_cosine = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = total_steps)
    
    def train(self):
        logger.info("Start training")

        cur_step = 1
        loss_step_scalar = 0
        loss_list = []
        best_loss = 1e3
        device = self.model.device

        for epoch in range(self.args.epochs):
            logger.info("="*60)
            logger.info(f"epoch {epoch+1}")
            logger.info("Train")
            self.model.train()
            loss_epoch_scalar = 0

            for input_ids, labels, attention_masks in tqdm(self.train_loader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_masks = attention_masks.to(device)

                output = self.model(input_ids=input_ids, attention_mask=attention_masks, 
                                    labels=labels)
                loss, logits = output[:2]
                
                if self.args.gradient_accumulation > 1:
                    loss = loss / self.args.gradient_accumulation
                
                loss_step_scalar += loss.item()
                loss_list.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                if cur_step % self.args.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.scheduler_linear.step()
                    # self.scheduler_cosine.step()
                    self.optimizer.zero_grad()

                    loss_epoch_scalar += loss_step_scalar
                    # logger.info(f"epoch: {epoch+1}, current step: {cur_step}, train loss: {loss_step_scalar}")
                    loss_step_scalar = 0
                
                if cur_step % self.args.save_step == 0:
                    # 保存一定训练步后的模型
                    self.save(self.model, self.args.output_dir, f'{cur_step}_checkpoint')
                
                cur_step += 1
            
            logger.info(f"Average train loss in {epoch+1}-th epoch: {loss_epoch_scalar/len(self.train_loader)}")
            loss_epoch_scalar = 0
            
            # 使用dev评测一下并记录最好模型
            eval_loss = self.evaluate()
            if best_loss is None or eval_loss < best_loss:
                # 保存最好模型
                logger.info(f"Saving best model")
                self.save(self.model, self.args.output_dir, 'best_model')

                best_loss = eval_loss
                logger.info(f"best loss: {best_loss}, best model is recorded")
            
            # 保存每代模型
            logger.info(f"Saving {epoch+1}-th epoch\'s model")
            self.save(self.model, self.args.output_dir, f"{epoch+1}_epoch")
        
        logger.info("="*60)
        logger.info("Training ends")
        
        # 画loss图
        self.plot_loss_curve(loss_list, self.args.plot_dir, self.args.cur_time)

    def evaluate(self, split='dev'):
        eval_loss = 0
        self.model.eval()
        device = self.model.device
        eval_acc = []
        eval_pre = []
        eval_re = []
        eval_f1 = []

        if split == 'dev':
            logger.info("Dev")
            eval_loader = self.dev_loader
        else:
            logger.info("Eval")
            eval_loader = self.test_loader

        for input_ids, labels, attention_masks in tqdm(eval_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_masks = attention_masks.to(device)

            output = self.model(input_ids=input_ids, attention_mask=attention_masks, 
                                labels=labels)
            
            loss, logits = output[:2]
            eval_loss += loss.item()
            accracy, precision, recall, f1 = self.eval_metrics(labels, logits)
            eval_acc.append(accracy)
            eval_pre.append(precision)
            eval_re.append(recall)
            eval_f1.append(f1)
        
        logger.info("Average {} loss: {}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
                    split, eval_loss/len(eval_loader), sum(eval_acc)/len(eval_acc), sum(eval_pre)/len(eval_pre),
                    sum(eval_re)/len(eval_re), sum(eval_f1)/len(eval_f1)
        ))
        return eval_loss/len(eval_loader)
    
    def save(self, model, output_dir, folder_name):    
        save_dir = Path(output_dir) / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_dir)
    
    def plot_loss_curve(self, loss_list, plot_dir, cur_time):
        logger.info("plot loss curve")
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        plt.cla()
        x = list(range(len(loss_list)))
        y = loss_list
        plt.title('Train loss vs. steps', fontsize=20)
        plt.plot(x, y)
        plt.xlabel('steps', fontsize=15)
        plt.ylabel('Train loss', fontsize=15)
        plt.grid()
        plt.savefig(Path(plot_dir) / (cur_time + "_train_loss.png"))
        plt.show()

        logger.info("plot completed")

    def eval_metrics(self, labels, logits):
        labels = labels.reshape(-1).cpu().numpy().tolist()
        pred = torch.argmax(logits, axis=-1)
        pred = pred.reshape(-1).cpu().numpy().tolist()

        accuracy = accuracy_score(labels, pred)
        precision = precision_score(labels, pred, average="macro")
        recall = recall_score(labels, pred, average="macro")
        f1 = f1_score(labels, pred, average="macro")

        return accuracy, precision, recall, f1
        