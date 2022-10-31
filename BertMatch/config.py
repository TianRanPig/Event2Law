import argparse
import os
import time

import torch


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--model_name", type=str, default="bert")
        self.parser.add_argument("--results_root", type=str, default="results")
        self.parser.add_argument("--seed", type=int, default=2022, help="random seed")
        self.parser.add_argument("--device", type=int, default=-1, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--data_ratio", default=1)

        self.parser.add_argument("--train_path", type=str, default="../data/train.json")
        self.parser.add_argument("--dev_path", type=str, default="../data/dev.json")
        # self.parser.add_argument("--test_path", type=str, default="../data/test.txt")

        self.parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop, use -1 to disable early stop")
        self.parser.add_argument("--batch_size", type=int, default=4, help="mini-batch size")
        self.parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
        self.parser.add_argument("--epsilon", type=float, default=1e-6, help="AdamW epsilon")
        self.parser.add_argument("--hidden_size", type=int, default=768)

        self.parser.add_argument("--max_length", type=int, default=512)
        self.parser.add_argument("--bert_path", type=str, default="bert-base-chinese")

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        opt.results_dir = os.path.join(opt.results_root, "-".join([opt.model_name, time.strftime("%Y_%m_%d_%H_%M_%S")]))
        mkdirp(opt.results_dir)
        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        self.opt = opt
        return opt


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)