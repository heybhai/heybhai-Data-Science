import numpy as np

# 1. Hyperparameters & Vocabulary
vocab = {"NLP": 0, "is": 1, "future": 2, "AI": 3}
vocab_size = len(vocab)
embed_dim = 2  # We'll represent words in 2D space

# 2. Initialize Weights (The Embeddings)
# W1 is the "Input Vector", W2 is the "Context Vector"
W1 = np.random.randn(vocab_size, embed_dim)
W2 = np.random.randn(embed_dim, vocab_size)

def get_input_vector(word_idx):
    # This acts like a 'lookup' or One-Hot multiplication
    return W1[word_idx]

# 3. Training Step (Manual Forward Pass)
# Context: "NLP" (target) -> "is" (context)
target_idx = vocab["NLP"]
context_idx = vocab["is"]

# Forward pass
h = get_input_vector(target_idx) # Hidden layer (the embedding)
u = np.dot(h, W2)                # Output layer (unnormalized scores)
exp_u = np.exp(u)
probs = exp_u / np.sum(exp_u)    # Softmax to get probabilities

print(f"Probability of 'is' given 'NLP' before training: {probs[context_idx]:.4f}")
#Probability of 'is' given 'NLP' before training: 0.0244

# Backpropagation logic (simplified):
# We would calculate the error (probs - target) and update W1 and W2 
# using the chain rule we discussed in Day 1!
