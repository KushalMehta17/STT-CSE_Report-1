import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def load_data():
    # Load the dataset from CSV file
    df = pd.read_csv('detailed_analysis.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    return df


def descriptive_stats(df):
    """Compute and report baseline descriptive statistics"""
    print("\n" + "="*60)
    print("TASK (b): BASELINE DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Total number of commits and files
    total_commits = df['Hash'].nunique()
    total_files = len(df)
    
    # Average number of modified files per commit
    files_per_commit = df.groupby('Hash').size()
    avg_files_per_commit = files_per_commit.mean()
    
    print(f"Total number of commits: {total_commits}")
    print(f"Total number of files: {total_files}")
    print(f"Average number of modified files per commit: {avg_files_per_commit:.2f}")
    
    fix_type_distribution = df['LLM Inference (fix type)'].value_counts()

    print("Fix Type Distribution:")
    for fix_type, count in fix_type_distribution.items():
        print(f"{fix_type}: {count}")
    
    # Most frequently modified extensions
    file_extensions = df['Filename'].apply(lambda x: re.search(r'\.(\w+)$', x).group(1) if re.search(r'\.(\w+)$', x) else 'None')
    top_extensions = file_extensions.value_counts().head(10)
    
    print("\nTop 10 File Extensions:")
    for ext, count in top_extensions.items():
        print(f"  .{ext}: {count}")
    
    plt.figure(figsize=(15, 5))
    
    # Fix type pie chart
    plt.subplot(1, 2, 1)
    plt.pie(fix_type_distribution.values(), labels=fix_type_distribution.keys(), autopct='%1.1f%%')
    plt.title('Distribution of Fix Types')
    
    # File extensions bar chart
    plt.subplot(1, 2, 2)
    top_extensions.plot(kind='bar')
    plt.title('Top 10 File Extensions')
    plt.xlabel('File Extension')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('lab3_task_b_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'total_commits': total_commits,
        'total_files': total_files,
        'avg_files_per_commit': avg_files_per_commit,
        'fix_type_distribution': fix_type_distribution,
        'top_extensions': top_extensions
    }


def main():
    
    # Load the data
    df = load_data()
    if df is None:
        return

    stats = descriptive_stats(df)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()