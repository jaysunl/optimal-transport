import numpy as np
import torch
from scipy.optimize import linprog
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_distances

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_word_embeddings(sentence):
    """
    Tokenizes a sentence and extracts contextual word embeddings using BERT.
    """
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad(): 
        outputs = model(**tokens)
    
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()

    return embeddings[1:-1]  

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
    C = cost_matrix.flatten()  

    weights1 = np.ones(n) / n
    weights2 = np.ones(m) / m

    A_eq = np.zeros((n + m, n * m))
    for i in range(n): 
        A_eq[i, i * m:(i + 1) * m] = 1
    for j in range(m):  
        A_eq[n + j, j::m] = 1

    b_eq = np.hstack([weights1, weights2])

    bounds = [(0, None)] * (n * m)

    result = linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        transport_plan = result.x.reshape(n, m)
        wasserstein_distance = np.sum(transport_plan * cost_matrix)  
        return wasserstein_distance, transport_plan
    else:
        raise ValueError("Optimal Transport LP did not converge.")

sentence1 = "Machine learning is a subset of artificial intelligence"
sentence2 = "Deep learning is part of artificial intelligence"

embeddings1 = get_word_embeddings(sentence1)
embeddings2 = get_word_embeddings(sentence2)

cost_matrix = compute_cost_matrix(embeddings1, embeddings2)

wasserstein_distance, transport_plan = solve_optimal_transport(cost_matrix)

print(f"Wasserstein Distance: {wasserstein_distance}")
print(f"Transport Plan:\n{transport_plan}")
