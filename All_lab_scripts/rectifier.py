from pydriller import Repository
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_name = "mamiksik/CommitPredictorT5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

rectifier_prompt = """
You are an expert software engineer reviewing a commit. Your task is to write the perfect commit message for a single file change.

Original commit message (may be vague or for multiple files): 
"{original_message}"

Code changes for this specific file:
{diff_text}

Instructions:
1. If the original message is already excellent and specific to this change, use it.
2. If it is vague, incorrect, or too general, rewrite it to be precise and descriptive.
3. The new message must clearly describe what changed in this file and why.
4. Be concise but specific. Use imperative mood (e.g., "Fix...", "Add...").

Write the perfect commit message for this file change:
"""

def generate_rectified_message(original_msg, diff_text, tokenizer, model):

    # Format the prompt with the actual original message and diff
    input_text = rectifier_prompt.format(original_message=original_msg, diff_text=diff_text)
    
    # Generate the rectified message with the LLM
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    generated_ids = model.generate(input_ids, max_length=128)
    rectified_msg = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return rectified_msg
    
def main():

    # Load the bug-fix commits we found earlier
    try:
        commits_df = pd.read_csv('bug_fix_commits.csv')
        hashes_to_analyze = commits_df['Hash'].tolist()
        print(f"Loaded {len(hashes_to_analyze)} bug-fix commits for analysis.")
    except FileNotFoundError:
        print("Error: bug_fix_commits.csv not found. Run miner.py first.")
        return

    # List to store our detailed data
    detailed_data = []
    counter=0

    # Iterate through each specific commit we care about
    for i, commit_hash in enumerate(hashes_to_analyze):
        print(f"Processing commit {i+1}/{len(hashes_to_analyze)}")
        
        # Use Pydriller to get the commit object for this specific hash
        for commit in Repository('.', single=commit_hash).traverse_commits():
            # Now iterate through each file in this specific commit
            for file in commit.modified_files:
                
                # Prepare the input for the LLM: the diff of the change
                diff_text = file.diff if file.diff else "No diff available."
                
                # RECTIFIER: Generate a context-aware, improved message
                llm_rectified_message = generate_rectified_message(commit.msg, diff_text, tokenizer, model)
                
                # Append all the data for this file change
                detailed_data.append({
                    'Hash': commit.hash,
                    'Message': commit.msg, # The original developer's message
                    'Filename': file.filename,
                    'Rectified Message': llm_rectified_message
                })
        
        counter+=1
        if counter>10:
            break

    # Create DataFrame and save
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('detailed_analysis_rectifier2.csv', index=False)
    print(f"Detailed analysis complete. Saved {len(detailed_df)} file changes to 'detailed_analysis_rectifier2.csv'.")

main()