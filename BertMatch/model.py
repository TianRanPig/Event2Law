import torch
from torch import nn
from transformers import BertModel


class BertMatchModel(nn.Module):
    def __init__(self, config):
        super(BertMatchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.loss_fct = nn.MSELoss()
        self.cos_score_transformation = nn.Identity()
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, model_inputs):
        event_bert_output = self.bert(input_ids=model_inputs['event_ids'],attention_mask=model_inputs['event_mask'])
        law_bert_output = self.bert(input_ids=model_inputs['law_ids'],attention_mask=model_inputs['law_mask'])
        event_pool_output = self.pooling(event_bert_output['last_hidden_state'], model_inputs['event_mask'])
        law_pool_output = self.pooling(law_bert_output['last_hidden_state'], model_inputs['law_mask'])
        output = self.cos_score_transformation(torch.cosine_similarity(event_pool_output, law_pool_output))
        loss = self.loss_fct(output.float(), model_inputs['labels'].view(-1).float())
        return loss, output




    def pooling(self, token_embeddings, attention_mask):
        output_vectors = []
        # [B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = torch.unsqueeze(attention_mask, -1).expand(token_embeddings.size()).float()
        # 这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        sum_embeddings = torch.sum(t,1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask,min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector
