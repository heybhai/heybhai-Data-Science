import gensim.downloader as api
import numpy as np

# 1. Load a small pre-trained Word2Vec model (Glove-wiki)
# This might take a minute to download the first time (~66MB)
print("Loading model...")
model = api.load("glove-wiki-gigaword-50") 

def calculate_closeness(w1, w2):
    # Get the vectors for each word
    vec1 = model[w1].reshape(1, -1)
    vec2 = model[w2].reshape(1, -1)
    
    # Calculate Cosine Similarity
    similarity = np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

# 2. Test the Distributional Hypothesis
word_pairs = [
    ("pizza", "burger"),   # High similarity (same context: food)
    ("pizza", "spaceship"),# Low similarity (different contexts)
    ("king", "queen"),     # High similarity (same context: royalty)
    ("doctor", "nurse"),    # High similarity (same context: hospital)
    ("run","running"),       # High similarity (same context: action)
    ("run","swim"),        # Low similarity (different contexts)
]

for w1, w2 in word_pairs:
    score = calculate_closeness(w1, w2)
    print(f"Similarity between '{w1}' and '{w2}': {score}")

# 3. The "Analogy" Magic
# King - Man + Woman = ?
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"\nAnalogy Result: King - Man + Woman = {result}")

'''
Similarity between 'pizza' and 'burger': [[0.7635947]]
Similarity between 'pizza' and 'spaceship': [[0.09771109]]
Similarity between 'king' and 'queen': [[0.7839042]]
Similarity between 'doctor' and 'nurse': [[0.79774964]]
Similarity between 'run' and 'running': [[0.88025516]]
Similarity between 'run' and 'swim': [[0.37359157]]

Analogy Result: King - Man + Woman = [('queen', 0.8523604273796082)]
'''