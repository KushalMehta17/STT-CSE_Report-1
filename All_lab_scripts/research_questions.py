import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def automated_score_message(message):
    """
    Scores a commit message for precision on a scale of 0-5.
    A score of 4 or 5 is considered a 'Hit'.
    """
    if not isinstance(message, str) or message.lower() == "no diff available.":
        return 0
    
    score = 0
    msg_lower = message.lower()
    words = message.split()
    
    if len(words) == 0:
        return 0
    
    # 1. Length (Max 2 points)
    score += min(2, len(words) / 5)  # 10 words -> 2 points, 5 words -> 1 point
    
    # 2. Specificity: Check for technical terms
    tech_terms = ['error', 'exception', 'fix', 'add', 'remove', 'update', 'refactor', 
                  'function', 'method', 'class', 'module', 'argument', 'parameter',
                  'validation', 'logic', 'calculation', 'initialization', 'null', 'none',
                  'bug', 'issue', 'resolve', 'defect', 'patch', 'fail']
    if any(term in msg_lower for term in tech_terms):
        score += 1
    
    # 3. Penalize Vague Words
    vague_words = ['stuff', 'things', 'update', 'fix', 'patch', 'changes', 'minor', 'tmp', 'temp']
    if any(vague in msg_lower for vague in vague_words):
        score -= 1  # Penalize for vagueness
    
    # 4. Imperative Mood Check
    imperative_verbs = ['fix', 'add', 'remove', 'update', 'refactor', 'implement', 
                        'correct', 'resolve', 'handle', 'improve', 'optimize',
                        'change', 'delete', 'create', 'adjust', 'modify']
    if words[0].lower() in imperative_verbs:
        score += 1
    
    # 5. Context Awareness
    context_indicators = ['in', 'for', 'to', 'with', 'on', 'when', 'during', 'while']
    if any(ctx in msg_lower for ctx in context_indicators):
        score += 1
    
    # Ensure score is within bounds
    return max(0, min(5, round(score, 2)))

def is_hit(score):
    # Returns 1 (Hit) if score is 4 or 5, else 0
    return 1 if score >= 4 else 0

def load_and_score_data():
    # Loads all CSV files and scores the commit messages
    print("Loading and scoring commit messages...")
    
    # Load the data
    dev_df = pd.read_csv('bug_fix_commits.csv')
    llm_df = pd.read_csv('detailed_analysis.csv')
    rectifier_df = pd.read_csv('detailed_analysis_rectifier.csv')
    
    # Score developer messages
    dev_df['Score'] = dev_df['Message'].apply(automated_score_message)
    dev_df['Hit'] = dev_df['Score'].apply(is_hit)
    
    # Score LLM messages (from detailed_analysis.csv)
    llm_df['Score'] = llm_df['Rectified Message'].apply(automated_score_message)
    llm_df['Hit'] = llm_df['Score'].apply(is_hit)
    
    # Score rectifier messages
    rectifier_df['Score'] = rectifier_df['Rectified Message'].apply(automated_score_message)
    rectifier_df['Hit'] = rectifier_df['Score'].apply(is_hit)
    
    return dev_df, llm_df, rectifier_df

def calculate_hit_rates(dev_df, llm_df, rectifier_df):
    # Calculates hit rates for all three approaches
    dev_hit_rate = (dev_df['Hit'].sum() / len(dev_df)) * 100
    llm_hit_rate = (llm_df['Hit'].sum() / len(llm_df)) * 100
    rectifier_hit_rate = (rectifier_df['Hit'].sum() / len(rectifier_df)) * 100
    
    return dev_hit_rate, llm_hit_rate, rectifier_hit_rate

def create_summary_table(dev_hit_rate, llm_hit_rate, rectifier_hit_rate):
    # Creates and displays a summary table
    summary_data = {
        'Approach': ['Developer (RQ1)', 'LLM (RQ2)', 'Rectifier (RQ3)'],
        'Hit Rate (%)': [dev_hit_rate, llm_hit_rate, rectifier_hit_rate],
        'Sample Size': [len(dev_df), len(llm_df), len(rectifier_df)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*50)
    print("SUMMARY TABLE - HIT RATES ANALYSIS")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)
    
    return summary_df

def plot_results(summary_df):
    """Creates bar plots of the results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot for hit rates
    approaches = summary_df['Approach']
    hit_rates = summary_df['Hit Rate (%)']
    
    bars = ax1.bar(approaches, hit_rates, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title('Hit Rate Comparison across Approaches')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, hit_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Sample size plot
    sample_sizes = summary_df['Sample Size']
    bars2 = ax2.bar(approaches, sample_sizes, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax2.set_ylabel('Number of Messages')
    ax2.set_title('Sample Size per Approach')
    
    # Add value labels on bars
    for bar, value in zip(bars2, sample_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('hit_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(dev_df, llm_df, rectifier_df):
    """Prints detailed statistical analysis"""
    print("\n" + "="*50)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*50)
    
    print(f"\nDeveloper Messages:")
    print(f"  - Total: {len(dev_df)}")
    print(f"  - Average Score: {dev_df['Score'].mean():.2f}/5")
    print(f"  - Hit Rate: {dev_df['Hit'].sum()}/{len(dev_df)} ({dev_df['Hit'].sum()/len(dev_df)*100:.1f}%)")
    
    print(f"\nLLM Messages:")
    print(f"  - Total: {len(llm_df)}")
    print(f"  - Average Score: {llm_df['Score'].mean():.2f}/5")
    print(f"  - Hit Rate: {llm_df['Hit'].sum()}/{len(llm_df)} ({llm_df['Hit'].sum()/len(llm_df)*100:.1f}%)")
    
    print(f"\nRectifier Messages:")
    print(f"  - Total: {len(rectifier_df)}")
    print(f"  - Average Score: {rectifier_df['Score'].mean():.2f}/5")
    print(f"  - Hit Rate: {rectifier_df['Hit'].sum()}/{len(rectifier_df)} ({rectifier_df['Hit'].sum()/len(rectifier_df)*100:.1f}%)")

if __name__ == "__main__":
    # Load data and score messages
    dev_df, llm_df, rectifier_df = load_and_score_data()
    
    # Calculate hit rates
    dev_hit_rate, llm_hit_rate, rectifier_hit_rate = calculate_hit_rates(dev_df, llm_df, rectifier_df)
    
    # Create and display summary table
    summary_df = create_summary_table(dev_hit_rate, llm_hit_rate, rectifier_hit_rate)
    
    # Print detailed analysis
    print_detailed_analysis(dev_df, llm_df, rectifier_df)
    
    # Generate plots
    plot_results(summary_df)
    
    print("\nAnalysis complete! Check 'hit_rate_analysis.png' for the visualizations.")