import time

import numpy as np
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from logger import logger
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizer, AdamW

from BertMatch.config import BaseOptions
from BertMatch.dataset import MyDataset
from BertMatch.model import BertMatchModel


def set_seed(seed, use_cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train(config, model, train_iter, eval_iter):
    if config.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(config.device)
        if len(config.device_ids) > 1:
            logger.info("Use multi GPU", config.device_ids)
            model = torch.nn.DataParallel(model, device_ids=config.device_ids)  # use multi GPU
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.epsilon)
    num_training_examples = len(train_iter)
    prev_best_score = 0
    es_cnt = 0
    logger.info("Start Training ......")
    model.train()
    for epoch in trange(0, config.n_epochs, desc="Epoch"):
        # global_step = (epoch + 1) * len(train_iter)
        for i, inputs in tqdm(enumerate(train_iter),desc="Training Iteration",total=num_training_examples):
            global_step = epoch * num_training_examples + i
            loss, _ = model(inputs['model_inputs'])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            config.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            config.writer.add_scalar("Train/Loss", float(loss), global_step)
        to_write = config.train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch, loss_str=loss)
        with open(config.train_log_filepath, "a", encoding='utf-8') as f:
            f.write(to_write)

        if eval_iter is not None:
            eval_acc, eval_loss = evaluate(model, eval_iter)
            to_write = config.eval_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch, eval_acc=eval_acc,eval_loss=eval_loss)
            with open(config.eval_log_filepath, 'a', encoding='utf-8') as f:
                f.write(to_write)
            stop_score = eval_loss
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score
                checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch}
                torch.save(checkpoint, config.ckpt_filepath)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if config.max_es_cnt != -1 and es_cnt > config.max_es_cnt:
                    with open(config.train_log_filepath, 'a', encoding='utf-8') as f:
                        f.write("Early Stop at epoch {}".format(epoch))
                    logger.info("Early stop at {}".format(epoch))
                    break
        else:
            checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch}
            torch.save(checkpoint, config.ckpt_filepath)
    config.writer.close()

def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    logger.info('Evaluating .....')
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(data_iter), desc="Evaluating Iteration", total=len(data_iter)):
            loss, output = model(inputs['model_inputs'])
            loss_total += loss
            labels = inputs['labels'].data.numpy()
            predict = torch.max(output.data, 1)[1].numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter)

def start_training():
    logger.info('Start Loading Tokenizer')
    tokenizer = BertTokenizer("../bert-pretrained/vocab.txt")

    config = BaseOptions().parse()
    config.writer = SummaryWriter(config.tensorboard_log_dir)
    config.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    config.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [acc] {eval_acc} [Loss] {eval_loss}\n"

    set_seed(config.seed)
    train_data = MyDataset(config.train_path, tokenizer)
    eval_data = MyDataset(config.dev_path, tokenizer)
    train_iter = DataLoader(train_data,batch_size=config.batch_size,shuffle=True)
    eval_iter = DataLoader(eval_data,batch_size=config.batch_size,shuffle=True)
    model = BertMatchModel(config)
    train(config, model, train_iter, eval_iter)

if __name__ == '__main__':
    start_training()
    logger.info("\n\n\nFINISHED TRAINING!!!")