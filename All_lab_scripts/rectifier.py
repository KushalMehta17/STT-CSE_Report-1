from pydriller import Repository
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the device to GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

def generate_rectified_message(original_msg, diff_text, tokenizer, model):

    messages = [
        {"role": "system", "content": "You are an expert software engineer. Your task is to write perfect, concise commit messages for individual file changes."},
        {"role": "user", "content": f"""
        Original commit message: \"{original_msg}\"

        Code changes for a specific file:
        {diff_text}

        Please rewrite the commit message to be specific to these changes. If the original message is good, improve it slightly. If it is vague, replace it completely.
        Focus on what was changed and why in this specific file. Use imperative mood and be concise.

        Output ONLY the rewritten commit message with no additional commentary:
        """}
        ]
    
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rectified_msg = generated_text.split("assistant\n")[-1].strip()
    
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