import pandas as pd

# Load the dataset with similarity metrics
df = pd.read_csv('analysis_with_similarity.csv')

# Define classification thresholds
SEMANTIC_THRESHOLD = 0.80  # ≥ 0.80 = Minor, < 0.80 = Major
TOKEN_THRESHOLD = 0.75     # ≥ 0.75 = Minor, < 0.75 = Major

print("Starting classification and agreement analysis...")

# Classify based on semantic similarity
df['Semantic_class'] = df['Semantic_similarity'].apply(
    lambda x: 'Minor' if x >= SEMANTIC_THRESHOLD else 'Major' if pd.notna(x) else None
)

# Classify based on token similarity  
df['Token_class'] = df['Token_similarity'].apply(
    lambda x: 'Minor' if x >= TOKEN_THRESHOLD else 'Major' if pd.notna(x) else None
)

# Check for agreement between classifications
df['Classes_Agree'] = df.apply(
    lambda row: 'YES' if (pd.notna(row['Semantic_class']) and 
                         pd.notna(row['Token_class']) and
                         row['Semantic_class'] == row['Token_class']) 
               else 'NO' if (pd.notna(row['Semantic_class']) and 
                            pd.notna(row['Token_class']))
               else None, 
    axis=1
)

# Calculate statistics
total_classifiable = len(df[df['Classes_Agree'].notna()])
agreement_count = (df['Classes_Agree'] == 'YES').sum()
disagreement_count = (df['Classes_Agree'] == 'NO').sum()
agreement_rate = (agreement_count / total_classifiable * 100) if total_classifiable > 0 else 0

# Display results
print("CLASSIFICATION AND AGREEMENT RESULTS")

print(f"\nClassification Distribution:")
print(f"Semantic - Major Fix: {(df['Semantic_class'] == 'Major').sum()}")
print(f"Semantic - Minor Fix: {(df['Semantic_class'] == 'Minor').sum()}")
print(f"Token - Major Fix: {(df['Token_class'] == 'Major').sum()}")  
print(f"Token - Minor Fix: {(df['Token_class'] == 'Minor').sum()}")

print(f"\nAgreement Analysis:")
print(f"Total classifiable commits: {total_classifiable}")
print(f"Agreements: {agreement_count} ({agreement_rate:.1f}%)")
print(f"Disagreements: {disagreement_count} ({100 - agreement_rate:.1f}%)")

print(f"\nFinal DataFrame Columns: {list(df.columns)}")

# Save the final results
final_columns = [
    'Hash', 'Message', 'Filename', 'Source Code (before)', 'Source Code (current)',
    'Diff', 'LLM Inference (fix type)', 'Rectified Message', 'MI_Change', 'CC_Change',
    'LOC_Change', 'Semantic_similarity', 'Token_similarity', 'Semantic_class',
    'Token_class', 'Classes_Agree'
]

# Keep only the required columns for the final output
final_df = df[final_columns]
final_df.to_csv('lab3_final_analysis.csv', index=False)

print(f"\nFinal results saved to 'lab3_final_analysis.csv'")
print(f"Dataset shape: {final_df.shape}")

# Generate a pie chart to visualize agreement vs disagreement
import matplotlib.pyplot as plt

# Create pie chart for agreement analysis
agree_count = (df['Classes_Agree'] == 'YES').sum()
disagree_count = (df['Classes_Agree'] == 'NO').sum()

# Data for pie chart
sizes = [agree_count, disagree_count]
labels = ['Agreement', 'Disagreement']
colors = ['#66b3ff', '#ff9999']
explode = (0.1, 0)  # explode the 1st slice

# Create pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Agreement Between Semantic and Token-based Classifications', fontsize=14, pad=20)

# Add annotation with total counts
plt.text(0, -1.2, f'Total Commits: {len(df)}\nAgreements: {agree_count}\nDisagreements: {disagree_count}', 
         ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('classification_agreement_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Pie chart saved as 'classification_agreement_pie_chart.png'")