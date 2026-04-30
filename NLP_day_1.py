# This code demonstrates the use of pre-trained word embeddings (GloVe) to compute word similarity and perform analogy tasks.
import gensim.downloader as api

# Load a small pre-trained GloVe model
model = api.load("glove-wiki-gigaword-100")

# 1. Similarity
print(model.similarity('king', 'queen'))

# 2. Analogy: King - Man + Woman
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Analogy Result: {result}")

# This code demonstrates how to train a Doc2Vec model on a small dataset of documents and infer a vector for a new document.
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data = ["I love natural language processing", 
        "Deep learning is a subset of AI", 
        "NLP and AI are changing the world"]

# Tag each document with an ID
tagged_data = [TaggedDocument(words=d.split(), tags=[str(i)]) for i, d in enumerate(data)]

# Train Doc2Vec
model = Doc2Vec(vector_size=20, min_count=1, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Infer a vector for a new document
new_doc = "I like learning about AI".split()
vector = model.infer_vector(new_doc)
print(f"Vector for new doc: {vector[:5]}...")

# This code demonstrates how to train a FastText model on a small dataset of sentences and retrieve the vector for a word that was not in the training data, showcasing FastText's ability to handle out-of-vocabulary words through subword information.
from gensim.models import FastText

# Dummy data
sentences = [["the", "quick", "brown", "fox"], ["jumped", "over", "the", "lazy", "dog"]]
ft_model = FastText(sentences, vector_size=10, window=3, min_count=1)

# Word not in training: 'foxes'
print("Vector for 'foxes':", ft_model.wv['foxes'][:5]) # Works because of subwords!
