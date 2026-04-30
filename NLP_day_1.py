import gensim.downloader as api

# Load a small pre-trained GloVe model
model = api.load("glove-wiki-gigaword-100")

# 1. Similarity
print(model.similarity('king', 'queen'))

# 2. Analogy: King - Man + Woman
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Analogy Result: {result}")