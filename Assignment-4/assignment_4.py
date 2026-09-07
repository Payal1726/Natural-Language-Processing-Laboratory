"""
Assignment 4: Word Embeddings – Word2Vec & GloVe
- Train Word2Vec model on custom e-commerce product descriptions
- Explore semantic similarity & word vector analogies (e.g., king - man + woman ≈ queen)
- 2D embedding space visualization via PCA / t-SNE
- E-Commerce Semantic Product Recommendation Engine
"""

import os
import json
import re
import numpy as np
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA, TruncatedSVD

def tokenize(text: str) -> List[str]:
    """Lowercase tokenization that strips punctuation for consistent matches."""
    return re.findall(r"[a-z0-9]+", text.lower())

def train_word_vectors(corpus_tokens: List[List[str]], dim: int = 50) -> Dict[str, np.ndarray]:
    """
    Builds simple distributional word vectors from token co-occurrence.
    """
    vocab = sorted({w.lower() for doc in corpus_tokens for w in doc})
    if not vocab:
        return {}

    if len(vocab) == 1:
        return {vocab[0]: np.array([1.0])}

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    co_matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
    window_size = 3

    for doc in corpus_tokens:
        for i, word in enumerate(doc):
            word_idx = word_to_idx[word]
            left = max(0, i - window_size)
            right = min(len(doc), i + window_size + 1)

            for j in range(left, right):
                if i == j:
                    continue

                context_idx = word_to_idx[doc[j]]
                co_matrix[word_idx, context_idx] += 1.0 / abs(i - j)

    total = co_matrix.sum()
    if total == 0:
        return {word: np.eye(len(vocab))[idx] for word, idx in word_to_idx.items()}

    row_sums = co_matrix.sum(axis=1, keepdims=True)
    col_sums = co_matrix.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums

    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(np.log((co_matrix * total + 1e-9) / (expected + 1e-9)), 0.0)

    rank = min(dim, len(vocab) - 1)
    reduced = TruncatedSVD(n_components=rank, random_state=42).fit_transform(ppmi)
    return {word: reduced[word_to_idx[word]] for word in vocab}

def compute_analogy(vec_a: np.ndarray, vec_b: np.ndarray, vec_c: np.ndarray) -> np.ndarray:
    """Computes word vector analogy: A - B + C (e.g., King - Man + Woman)."""
    return vec_a - vec_b + vec_c

def recommend_products(query: str, products: List[dict], word_vectors: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    """Suggests semantically similar e-commerce products for a user search query."""
    q_words = tokenize(query)
    q_vecs = [word_vectors[w] for w in q_words if w in word_vectors]
    
    if not q_vecs:
        return []
    
    query_centroid = np.mean(q_vecs, axis=0)
    query_norm = np.linalg.norm(query_centroid)
    
    results = []
    for product in products:
        p_tokens = tokenize(f"{product['name']} {product['description']}")
        p_vecs = [word_vectors[w] for w in p_tokens if w in word_vectors]
        
        if p_vecs:
            doc_centroid = np.mean(p_vecs, axis=0)
            doc_norm = np.linalg.norm(doc_centroid)
            
            if query_norm > 0 and doc_norm > 0:
                sim = np.dot(query_centroid, doc_centroid) / (query_norm * doc_norm)
                results.append((product["name"], float(sim)))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    print("=" * 60)
    print("ASSIGNMENT 4: WORD EMBEDDINGS - WORD2VEC & GLOVE")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__), "ecommerce_products.json")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    corpus_tokens = [tokenize(f"{p['name']} {p['description']}") for p in products]
    vectors = train_word_vectors(corpus_tokens)

    # 1. Analogy Demonstration
    if all(w in vectors for w in ["king", "man", "woman"]):
        analogy_res = compute_analogy(vectors["king"], vectors["man"], vectors["woman"])
        print("\n--- Vector Analogy Execution ---")
        print("Expression: king - man + woman")
        print(f"Analogy Vector Shape: {analogy_res.shape}")
    else:
        print("\n--- Vector Analogy Execution ---")
        print("Skipped: analogy anchor words are not present in the e-commerce vocabulary.")

    # 2. PCA Dimensionality Reduction (2D Space)
    sample_words = list(vectors.keys())[:8]
    if len(sample_words) >= 2:
        sample_matrix = np.array([vectors[w] for w in sample_words])
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(sample_matrix)
        
        print("\n--- 2D Projection Coordinates (PCA) ---")
        for word, coord in zip(sample_words, pca_2d):
            print(f"Word: {word:<15} | 2D Coordinates: ({coord[0]:.3f}, {coord[1]:.3f})")
    else:
        print("\n--- 2D Projection Coordinates (PCA) ---")
        print("Skipped: not enough vocabulary terms for 2D projection.")

    # 3. Product Recommendation Query
    query = "headphone"
    recs = recommend_products(query, products, vectors)
    print(f"\n--- E-Commerce Product Recommendations for Query: '{query}' ---")
    if recs:
        for name, score in recs:
            print(f"Product: {name:<40} | Similarity Score: {score:.4f}")
    else:
        print("No recommendations found for the current query.")

if __name__ == "__main__":
    main()
