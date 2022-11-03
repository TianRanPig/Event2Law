from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
emb1 = model.encode("调漆房未进行密闭")
emb2 = model.encode("应当在密闭空间或者设备中进行或者设备中进")
cos_sim = util.cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)




