import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import BertTokenizer

from BertMatch.config import BaseOptions
from BertMatch.dataset import MyDataset, load_dataset

# tokenizer = BertTokenizer(vocab_file="../bert-pretrained/vocab.txt")
# examples = ["我爱北京天安门", "天安门广场"]
#
# res = tokenizer(examples,
#                 padding=True,
#                 truncation=True,
#                 max_length=12,
#                 return_tensors="pt",
#                 return_length=True)
# print(res['length'][0])

config = BaseOptions().parse()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y%m%d %H:%M:%S %p")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logfile = config.train_log_filepath
file_handler = logging.FileHandler(logfile, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



