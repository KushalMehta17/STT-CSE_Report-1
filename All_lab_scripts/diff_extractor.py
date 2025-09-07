from pydriller import Repository
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_name = "mamiksik/CommitPredictorT5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load commit hashes
try:
    commits_df = pd.read_csv('bug_fix_commits.csv')
    hashes_to_analyze = commits_df['Hash'].dropna().tolist()
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit()

# Prompt template for classification
classification_prompt = (
    "Classify the type of fix in this code change into one of the following categories:\n"
    "- Bug Fix\n- Feature Addition\n- Refactoring\n- Documentation\n- Performance Improvement\n"
    "- Style Fix\n- Dependency Update\n- Other\n\nCode Change:\n"
)

# Store output rows
results = []

# Loop through commit hashes
for i, commit_hash in enumerate(hashes_to_analyze):
    print(f"Processing commit {i+1}/{len(hashes_to_analyze)}")
    try:
        repo = Repository('.', single=commit_hash)
        for commit in repo.traverse_commits():
            for file in commit.modified_files:
                diff_text = file.diff or "No diff available."

                # Rectified Commit Message
                prompt_msg = "Generate a concise commit message for this code change: " + diff_text
                input_ids_msg = tokenizer(prompt_msg, return_tensors="pt", max_length=512, truncation=True).input_ids
                output_ids_msg = model.generate(input_ids_msg, max_length=100)
                rectified_message = tokenizer.decode(output_ids_msg[0], skip_special_tokens=True)

                # Fix Type Classification
                prompt_cls = classification_prompt + diff_text
                input_ids_cls = tokenizer(prompt_cls, return_tensors="pt", max_length=512, truncation=True).input_ids
                output_ids_cls = model.generate(input_ids_cls, max_length=50)
                fix_type = tokenizer.decode(output_ids_cls[0], skip_special_tokens=True)

                # Store Data
                results.append({
                    'Hash': commit.hash,
                    'Message': commit.msg,
                    'Filename': file.filename,
                    'Source Code (before)': file.source_code_before,
                    'Source Code (current)': file.source_code,
                    'Diff': diff_text,
                    'LLM Inference (fix type)': fix_type.strip(),
                    'Rectified Message': rectified_message.strip()
                })

    except Exception as e:
        print(f"[Error processing commit {commit_hash}]: {e}")
        continue

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('detailed_analysis.csv', index=False)
print("Analysis complete. Results saved to 'detailed_analysis.csv'")