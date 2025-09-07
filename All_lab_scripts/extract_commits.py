from pydriller import Repository
import pandas as pd

# Defined the repository path
repo_path = '.'

# Defined the bug-fixing keyword list
bug_keywords = ['fix', 'bug', 'error', 'issue', 'resolve']

# List to store our mined data
bug_fix_commits = []

# 1. Iterate through each commit in the repository
for commit in Repository(repo_path).traverse_commits():
    
    # 2. Check if any keyword is in the commit message
    message_lower = commit.msg.lower()
    if any(keyword in message_lower for keyword in bug_keywords):
        
        # 3. Get the parent hashes. A commit can have multiple parents (e.g., in a merge).
        parent_hashes = [parent for parent in commit.parents]
        
        # 4. Check if it's a merge commit (has more than one parent?)
        is_merge = len(commit.parents) > 1
        
        # 5. Get the list of files modified in this commit
        modified_files = []
        for file in commit.modified_files:
            modified_files.append(file.filename)
        
        # 6. Append all the required data for this commit to our list
        bug_fix_commits.append({
            'Hash': commit.hash,
            'Message': commit.msg,
            'Hashes of parents': ", ".join(parent_hashes), # Join list into a string for CSV
            'Is a merge commit?': is_merge,
            'List of modified files': ", ".join(modified_files) # Join list into a string
        })

# 7. Convert the list of dictionaries to a Pandas DataFrame and save as CSV
df = pd.DataFrame(bug_fix_commits)
df.to_csv('bug_fix_commits.csv', index=False)
print(f"Found {len(df)} potential bug-fixing commits. Saved to 'bug_fix_commits.csv'.")