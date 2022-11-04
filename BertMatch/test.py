from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# emb1 = model.encode("调漆房未进行密闭")
# emb2 = model.encode("应当在密闭空间或者设备中进行或者设备中进")
# cos_sim = util.cos_sim(emb1, emb2)
# print("Cosine-Similarity:", cos_sim)
from transformers import BertTokenizer

tokenizer = BertTokenizer("../bert-pretrained/vocab.txt")
input_text = ["我爱北京天安门", "广场吃炸鸡"]
tok_res = tokenizer(input_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt")
print(tok_res)



