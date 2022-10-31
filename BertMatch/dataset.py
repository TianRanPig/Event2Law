import json
from torch.utils.data import Dataset


def load_dataset(path):
    events = []
    laws = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            events.append(data['event'])
            laws.append(data['law'])
            labels.append(int(data['label']))
    return events, laws, labels

class MyDataset(Dataset):
    def __init__(self,path, tokenizer, max_length=512):
        self.events, self.laws, self.labels = load_dataset(path)
        self.events_feature = tokenizer(self.events, padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_length=True)
        self.laws_feature = tokenizer(self.laws, padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_length=True)

    def __getitem__(self, index):
        meta = dict()
        model_inputs = dict()
        meta['event'] = self.events[index]
        meta['law'] = self.laws[index]
        model_inputs['event_ids'] = self.events_feature['input_ids'][index]
        model_inputs['event_mask'] = self.events_feature['attention_mask'][index]
        model_inputs['event_length'] = int(self.events_feature['length'][index])
        model_inputs['law_ids'] = self.laws_feature['input_ids'][index]
        model_inputs['law_mask'] = self.laws_feature['attention_mask'][index]
        model_inputs['law_length'] = int(self.laws_feature['length'][index])
        model_inputs['labels'] = self.labels[index]

        return dict(meta=meta,model_inputs=model_inputs)

    def __len__(self):
        return len(self.events)
