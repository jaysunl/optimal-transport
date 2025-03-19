import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_distances

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_word_embeddings(sentence):
    """
    Tokenizes a sentence and extracts contextual word embeddings using BERT.
    Returns embeddings excluding the [CLS] and [SEP] tokens.
    """
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    return embeddings[1:-1]  

def interior_point_ot(C, mu, nu, tol=1e-6, max_iter=100, alpha=0.95):
    """
    Solves the Optimal Transport problem using a simplified iterative Interior-Point method.
    
    Args:
        C (ndarray): Cost matrix (n x m)
        mu (ndarray): Source distribution (n,)
        nu (ndarray): Target distribution (m,)
        tol (float): Convergence tolerance
        max_iter (int): Maximum iterations
        alpha (float): Step size scaling factor (0 < alpha < 1)
    
    Returns:
        T (ndarray): Approximated optimal transport plan (n x m)
    """
    n, m = C.shape

    T = np.ones((n, m)) / (n * m)

    u = np.zeros(n)
    v = np.zeros(m)

    mu = mu.reshape(-1, 1)
    nu = nu.reshape(1, -1)

    for iteration in range(max_iter):
        residual_p = np.sum(T, axis=1) - mu.flatten() 
        residual_d = np.sum(T, axis=0) - nu.flatten() 

        slackness = C - u[:, None] - v[None, :]
        
        dT = T * slackness  

        mask = dT < 0
        if np.any(mask):
            max_step = np.min(-T[mask] / dT[mask])
        else:
            max_step = 1.0  
        
        step_size = alpha * max_step

        T = T + step_size * dT
        
        if np.linalg.norm(residual_p) < tol and np.linalg.norm(residual_d) < tol:
            print(f"Converged in {iteration+1} iterations.")
            break

    return T

def compute_sentence_similarity(sentence1, sentence2):
    emb1 = get_word_embeddings(sentence1)
    emb2 = get_word_embeddings(sentence2)
    
    C = cosine_distances(emb1, emb2)
    
    mu = np.ones(emb1.shape[0]) / emb1.shape[0]
    nu = np.ones(emb2.shape[0]) / emb2.shape[0]
    
    T = interior_point_ot(C, mu, nu)
    
    wasserstein_distance = np.sum(T * C)
    return wasserstein_distance, T

# Example sentences
sentence1 = "Machine learning is a subset of artificial intelligence."
sentence2 = "Deep learning is part of artificial intelligence."

distance, transport_plan = compute_sentence_similarity(sentence1, sentence2)
print(f"Wasserstein Distance: {distance}")
print("Transport Plan:")
print(transport_plan)
