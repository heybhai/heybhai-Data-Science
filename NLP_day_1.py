"""Natural Language Processing with Word2Vec and Doc2Vec models along with RNNs."""
# # This code demonstrates the use of pre-trained word embeddings (GloVe) to compute word similarity and perform analogy tasks.
# import gensim.downloader as api

# # Load a small pre-trained GloVe model
# model = api.load("glove-wiki-gigaword-100")

# # 1. Similarity
# print(model.similarity('king', 'queen'))

# # 2. Analogy: King - Man + Woman
# result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
# print(f"Analogy Result: {result}")

# # This code demonstrates how to train a Doc2Vec model on a small dataset of documents and infer a vector for a new document.
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# data = ["I love natural language processing", 
#         "Deep learning is a subset of AI", 
#         "NLP and AI are changing the world"]

# # Tag each document with an ID
# tagged_data = [TaggedDocument(words=d.split(), tags=[str(i)]) for i, d in enumerate(data)]

# # Train Doc2Vec
# model = Doc2Vec(vector_size=20, min_count=1, epochs=50)
# model.build_vocab(tagged_data)
# model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# # Infer a vector for a new document
# new_doc = "I like learning about AI".split()
# vector = model.infer_vector(new_doc)
# print(f"Vector for new doc: {vector[:5]}...")

# # This code demonstrates how to train a FastText model on a small dataset of sentences and retrieve the vector for a word that was not in the training data, showcasing FastText's ability to handle out-of-vocabulary words through subword information.
# from gensim.models import FastText

# # Dummy data
# sentences = [["the", "quick", "brown", "fox"], ["jumped", "over", "the", "lazy", "dog"]]
# ft_model = FastText(sentences, vector_size=10, window=3, min_count=1)

# # Word not in training: 'foxes'
# print("Vector for 'foxes':", ft_model.wv['foxes'][:5]) # Works because of subwords!

# # This code demonstrates the training of both Word2Vec and Doc2Vec models on a sample paragraph, highlighting the differences in how they represent text. Word2Vec focuses on individual word relationships, while Doc2Vec captures the overall context of the entire paragraph using a unique Paragraph ID.
# from gensim.models import Word2Vec, Doc2Vec
# from gensim.models.doc2vec import TaggedDocument
# import numpy as np

# # 1. The Scenario: A large sample paragraph
# paragraph = """Natural language processing is a subfield of linguistics, computer science, 
# and artificial intelligence concerned with the interactions between computers and human language, 
# in particular how to program computers to process and analyze large amounts of natural language data."""

# # Preprocessing: Tokenize the paragraph into a list of words
# words = paragraph.lower().split()
# data = [words]

# # 2. Word2Vec Implementation (The Atomic View)
# # Learns vectors for individual words [cite: 59]
# w2v_model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1, workers=4)

# # 3. Doc2Vec Implementation (The Holistic View)
# # Learns a vector for the entire paragraph using a unique Paragraph ID [cite: 38, 64]
# tagged_data = [TaggedDocument(words=words, tags=["DOC_001"])]
# d2v_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=100)
# d2v_model.build_vocab(tagged_data)
# d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

# # --- Output Comparison ---

# print("--- Word2Vec Analysis ---")
# # Word2Vec ignores overall document context and focuses on word relationships [cite: 48, 72]
# print(f"Number of words processed: {len(words)}")
# print(f"Shape of vector for the word 'language': {w2v_model.wv['language'].shape}")
# # To get a document vector here, you would typically need to average all word vectors 

# print("\n--- Doc2Vec Analysis ---")
# # Doc2Vec uses the Paragraph ID as 'memory' for the whole text [cite: 40, 72]
# doc_vector = d2v_model.dv["DOC_001"]
# print(f"Number of documents processed: 1")
# print(f"Shape of the single paragraph vector: {doc_vector.shape}")

