import numpy as np
import torch
from scipy.optimize import linprog
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_distances

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_word_embeddings(sentence):
    """
    Tokenizes a sentence and extracts contextual word embeddings using BERT.
    """
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():  # No gradient computation
        outputs = model(**tokens)
    
    # Extract last hidden state (word embeddings)
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()

    # Remove CLS and SEP tokens (first and last)
    return embeddings[1:-1]  # Exclude special tokens

def compute_cost_matrix(embeddings1, embeddings2):
    """
    Computes the cost matrix based on cosine distance.
    """
    return cosine_distances(embeddings1, embeddings2)

def solve_optimal_transport(cost_matrix):
    """
    Solves the Optimal Transport problem using linear programming.
    """
    n, m = cost_matrix.shape
    C = cost_matrix.flatten()  # Flatten cost matrix into 1D vector

    # Uniform probability distributions
    weights1 = np.ones(n) / n
    weights2 = np.ones(m) / m

    # Constraints: Marginals must sum to weights
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):  # Row constraints
        A_eq[i, i * m:(i + 1) * m] = 1
    for j in range(m):  # Column constraints
        A_eq[n + j, j::m] = 1

    b_eq = np.hstack([weights1, weights2])  # Concatenate marginals

    # Bounds: Transport variables must be non-negative
    bounds = [(0, None)] * (n * m)

    # Solve LP
    result = linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        transport_plan = result.x.reshape(n, m)
        wasserstein_distance = np.sum(transport_plan * cost_matrix)  # Compute cost
        return wasserstein_distance, transport_plan
    else:
        raise ValueError("Optimal Transport LP did not converge.")

# Example sentences
sentence1 = "Machine learning is a subset of artificial intelligence"
sentence2 = "Deep learning is part of artificial intelligence"

# Compute BERT embeddings
embeddings1 = get_word_embeddings(sentence1)
embeddings2 = get_word_embeddings(sentence2)

# Compute cost matrix
cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

# Solve OT problem
wasserstein_distance, transport_plan = solve_optimal_transport(cost_matrix)

# Print results
print(f"Wasserstein Distance: {wasserstein_distance}")
print(f"Transport Plan:\n{transport_plan}")
