import copy
import json
import random

event2Law_path = '../data/event2Law.json'
all_path = '../data/all.json'
rights_path = '../data/市直部门权责清单.json'

def generate_dataset():
    maxEventLen = 0
    maxLawLen = 0
    fw = open(all_path, 'a', encoding='utf-8')
    with open(event2Law_path,'r',encoding='utf-8') as f1, open(rights_path,'r',encoding='utf-8') as f2:
        rights = json.load(f2)
        for line in f1.readlines():
            line_json = json.loads(line)
            line_neg = copy.deepcopy(line_json)
            line_json['label'] = 1
            line_neg['label'] = 0
            depIndex = random.randint(0,len(rights)-1)
            while depIndex == 16:
                depIndex = random.randint(0,len(rights)-1)
            dataInfo = random.choice(rights[depIndex]['dataInfos'])
            if "childrenShowDataInfo" in dataInfo.keys():
                if type(dataInfo['childrenShowDataInfo']) == list:
                    dataInfo = random.choice(dataInfo['childrenShowDataInfo'])
            line_neg['law'] = dataInfo['zqyj']
            if len(line_json['event']) > maxEventLen:
                maxEventLen = len(line_json['event'])
            if len(line_neg['event']) > maxEventLen:
                maxEventLen = len(line_neg['event'])
            if len(line_json['law']) > maxLawLen:
                maxLawLen = len(line_json['law'])
            if len(line_neg['law']) > maxLawLen:
                maxLawLen = len(line_neg['law'])
            fw.write(json.dumps(line_json, ensure_ascii=False) + '\n')
            fw.write(json.dumps(line_neg, ensure_ascii=False) + '\n')
    print('event最长长度：{}', maxEventLen)
    print('law最长长度：{}', maxLawLen)

def dataset_split(ratio=0.8):
    f_train = open('../data/train.json','a',encoding='utf-8')
    f_dev = open('../data/dev.json', 'a', encoding='utf-8')
    with open(all_path, 'r', encoding='utf-8') as f:
        list = f.readlines()
        offset = int(len(list) * ratio) - 1
        train = list[:offset]
        dev = list[offset:]
        for line in train:
            f_train.write(line)
        for lin in dev:
            f_dev.write(lin)

def statistics():
    eventNum1 = 0
    lawNum1 = 0
    with open(all_path,'r', encoding='utf-8') as f:
        for line in f.readlines():
            res = json.loads(line)
            if len(res['event']) > 512:
                eventNum1 += 1
                print(res)
            if len(res['law']) > 512:
                lawNum1 += 1
    print('all中长度大于512的event有：{}',eventNum1)
    print('all中长度大于512的law有：{}', lawNum1)

    eventNum2 = 0
    lawNum2 = 0
    with open(event2Law_path,'r',encoding='utf-8') as fe:
        for line in fe.readlines():
            res = json.loads(line)
            if len(res['event']) > 512:
                eventNum2 += 1
                print(res)
            if len(res['law']) > 512:
                lawNum2 += 1
    print('event2Law中长度大于512的event有：{}', eventNum2)
    print('event2Law中长度大于512的law有：{}', lawNum2)

if __name__ == '__main__':
    # generate_dataset()
    # dataset_split()
    statistics()