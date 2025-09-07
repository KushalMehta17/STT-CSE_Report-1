from pydriller import Repository
import pandas as pd
import subprocess
import tempfile
import os

# 1. Define Paths 
repo_path = './ragas'  
output_csv_path = './ragas_analysis.csv'

# 2. Helper Function to get a git diff
def get_git_diff(old_content, new_content, algorithm):

    old_content = old_content if old_content is not None else ''
    new_content = new_content if new_content is not None else ''
    
    # Create temporary files
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.old') as f_old:
            f_old.write(old_content)
            old_path = f_old.name
            
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.new') as f_new:
            f_new.write(new_content)
            new_path = f_new.name
        
        # Run git diff with proper flags 
        result = subprocess.run([
            'git', 'diff', 
            '--no-index',
            '-w',  
            f'--diff-algorithm={algorithm}',
            '--',
            old_path, new_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode in [0, 1]:
            return result.stdout
        else:
            print(f"Git diff failed with return code {result.returncode}: {result.stderr}")
            return "ERROR"
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "ERROR"
    finally:
        # Clean up temporary files
        os.unlink(old_path)
        os.unlink(new_path)

# 3. Helper Function to classify file type
def classify_file_type(file_path):
    if not file_path:
        return 'Other'
    file_path_lower = file_path.lower()
    if 'test' in file_path_lower or 'spec' in file_path_lower:
        return 'Test'
    elif file_path_lower.endswith('readme.md'):
        return 'README'
    elif 'license' in file_path_lower:
        return 'LICENSE'
    elif any(file_path_lower.endswith(ext) for ext in ['.py', '.java', '.js', '.ts', '.c', '.cpp', '.h', '.html', '.css', '.go', '.rs']):
        return 'Source'
    else:
        return 'Other'

# 4. Main Mining Logic
print(f"\nStarting to mine repository: {repo_path}")

# We will store each row as a dict in this list, then convert to DataFrame
data_rows = []
commitcount = 0

# Iterate through every commit in the repository
try:
    for commit in Repository(repo_path).traverse_commits():
        commitcount += 1
        if commitcount % 10 == 0:
            print(f"Processing commit {commitcount}") 
        
        for file in commit.modified_files:
            try:
                # Skip binary files
                if file.filename and any(file.filename.lower().endswith(ext) 
                                       for ext in ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.exe', '.dll', '.so']):
                    continue
                
                # Get the required metadata
                old_path = file.old_path
                new_path = file.new_path
                has_old = file.source_code_before is not None and len(file.source_code_before.strip()) > 0
                has_new = file.source_code is not None and len(file.source_code.strip()) > 0
                
                # Generate the two diffs
                diff_myers = get_git_diff(file.source_code_before, file.source_code, 'myers')
                diff_hist = get_git_diff(file.source_code_before, file.source_code, 'histogram')
                
                # Check for Discrepancy (Simple string comparison)
                discrepancy = 'Yes' if diff_myers != diff_hist else 'No'
                
                # Classify the File Type
                file_type = classify_file_type(new_path if new_path else old_path)
                
                # Create a dictionary for this row of data
                row_data = {
                    'old_file_path': old_path,
                    'new_file_path': new_path,
                    'commit_sha': commit.hash,
                    'parent_commit_sha': commit.parents[0] if commit.parents else None,
                    'commit_message': commit.msg,
                    'diff_myers': diff_myers,
                    'diff_hist': diff_hist,
                    'discrepancy': discrepancy,
                    'file_type': file_type
                }
                data_rows.append(row_data)
                
            except Exception as e:
                print(f"Error processing file {new_path or old_path} in commit {commit.hash[:8]}: {e}")
                continue

except Exception as e:
    print(f"Error during repository traversal: {e}")


print(f"\nFinished mining. Creating DataFrame and saving CSV...")
if data_rows:
    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    print(f"All done! Data saved to: {output_csv_path}")
    print(f"Total file modifications analyzed: {len(df)}")

else:
    print("No data collected.")