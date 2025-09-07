import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sacrebleu import corpus_bleu

# Load the dataset
df = pd.read_csv('detailed_analysis_with_radon.csv')

# Initialize CodeBERT model and tokenizer
print("Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def calculate_semantic_similarity(code1, code2):
    try:
        if not code1 or not code2 or pd.isna(code1) or pd.isna(code2):
            return None
        
        # Skip very short code snippets
        if len(str(code1).strip()) < 10 or len(str(code2).strip()) < 10:
            return None
        
        # Tokenize and encode the code snippets
        inputs1 = tokenizer(str(code1), return_tensors="pt", truncation=True, 
                           padding=True, max_length=512)
        inputs2 = tokenizer(str(code2), return_tensors="pt", truncation=True, 
                           padding=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
        
        # Use the [CLS] token representation as the sentence embedding
        embedding1 = outputs1.last_hidden_state[:, 0, :].numpy().flatten()
        embedding2 = outputs2.last_hidden_state[:, 0, :].numpy().flatten()
        
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(embedding1, embedding2)
        return max(0.0, min(1.0, similarity))  # Ensure value is between 0-1
    
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return None

def calculate_token_similarity(code1, code2):
    try:
        if not code1 or not code2 or pd.isna(code1) or pd.isna(code2):
            return None
        
        # Convert to string and clean
        code1_str = str(code1).strip()
        code2_str = str(code2).strip()
        
        # Handle very short code snippets
        if len(code1_str) < 10 or len(code2_str) < 10:
            return 0.0
        
        # BLEU expects references as list of strings and hypothesis as string
        # Format: references = [[ref1_string, ref2_string, ...]], hypothesis = hyp_string
        bleu_score = corpus_bleu([code2_str], [code1_str]).score / 100
        
        return max(0.0, min(1.0, bleu_score))  # Ensure value is between 0-1
    
    except Exception as e:
        print(f"Error calculating token similarity for codes of length {len(str(code1))} and {len(str(code2))}: {e}")
        return None

print("Calculating similarity metrics...")

# Initialize new columns
df['Semantic_similarity'] = None
df['Token_similarity'] = None

# Process each row
for idx, row in df.iterrows():
    if idx % 50 == 0:  # Print progress every 50 files
        print(f"Processing file {idx+1}/{len(df)}")
    
    before_code = row['Source Code (before)']
    after_code = row['Source Code (current)']
    
    # Calculate semantic similarity
    semantic_sim = calculate_semantic_similarity(before_code, after_code)
    df.at[idx, 'Semantic_similarity'] = semantic_sim
    
    # Calculate token similarity
    token_sim = calculate_token_similarity(before_code, after_code)
    df.at[idx, 'Token_similarity'] = token_sim

# Display summary statistics
semantic_scores = df['Semantic_similarity'].dropna()
token_scores = df['Token_similarity'].dropna()

print("\nSummary of Similarity Metrics:")
print()
print(f"Semantic similarity - Mean: {semantic_scores.mean():.3f}, "
      f"Std: {semantic_scores.std():.3f}, "
      f"Range: [{semantic_scores.min():.3f}, {semantic_scores.max():.3f}]")
print(f"Token similarity - Mean: {token_scores.mean():.3f}, "
      f"Std: {token_scores.std():.3f}, "
      f"Range: [{token_scores.min():.3f}, {token_scores.max():.3f}]")

# Check correlation between the two metrics
if len(semantic_scores) > 0 and len(token_scores) > 0:
    correlation = df[['Semantic_similarity', 'Token_similarity']].corr().iloc[0, 1]
    print(f"Correlation between semantic and token similarity: {correlation:.3f}")

# Save the updated dataframe
df.to_csv('analysis_with_similarity.csv', index=False)
print("\nAnalysis complete. Results saved to 'analysis_with_similarity.csv'")

# Show sample of results
print("\nSample of results (first 5 rows):")
print(df[['Hash', 'Filename', 'Semantic_similarity', 'Token_similarity']].head().to_string(index=False))