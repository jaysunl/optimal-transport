import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to get word embeddings using BERT
def get_word_embeddings(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    return embeddings[1:-1]  # Exclude [CLS] and [SEP] tokens

# Interior-point OT solver (simplified)
def interior_point_ot(C, mu, nu, tol=1e-6, max_iter=100, alpha=0.95):
    n, m = C.shape
    T = np.ones((n, m)) / (n * m)  # Initialize transport plan
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
            break

    return T

# Function to compute sentence similarity using OT
def compute_sentence_similarity(sentence1, sentence2):
    emb1 = get_word_embeddings(sentence1)
    emb2 = get_word_embeddings(sentence2)
    
    C = cosine_distances(emb1, emb2)
    
    mu = np.ones(emb1.shape[0]) / emb1.shape[0]
    nu = np.ones(emb2.shape[0]) / emb2.shape[0]
    
    T = interior_point_ot(C, mu, nu)
    wasserstein_distance = np.sum(T * C)
    
    return wasserstein_distance

# Load the STS Benchmark dataset
dataset = load_dataset("stsb_multi_mt", "en")  # Use the test split for evaluation

# Calculate the OT distance for each sentence pair
predicted_distances = []
true_scores = []
train_data = dataset["train"]
for example in train_data:
    sentence1 = example['sentence1']
    sentence2 = example['sentence2']
    true_score = example['similarity_score']
    
    # Compute OT distance (Wasserstein distance)
    distance = compute_sentence_similarity(sentence1, sentence2)
    
    predicted_distances.append(distance)
    true_scores.append(true_score)
    break
# Compute Spearman correlation between the predicted distances and the true scores
print(predicted_distances)
print(true_scores)
spearman_corr, _ = spearmanr(predicted_distances, true_scores)

print(f"Spearman Rank Correlation: {spearman_corr}")
