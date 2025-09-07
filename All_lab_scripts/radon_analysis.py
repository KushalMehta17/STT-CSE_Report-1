import pandas as pd
from radon import metrics, raw, complexity

# Load the dataset
df = pd.read_csv('detailed_analysis.csv')

def calculate_radon_metrics(code):
    try:
        if not code or pd.isna(code):
            return None, None, None
        
        # Calculate Maintainability Index
        mi = metrics.mi_visit(code, multi=True)
        
        # Calculate Cyclomatic Complexity
        cc_result = complexity.cc_visit(code)
        total_cc = sum([func.complexity for func in cc_result]) if cc_result else 0
        
        # Calculate Lines of Code
        loc_analysis = raw.analyze(code)
        loc = loc_analysis.loc
        
        return mi, total_cc, loc
    except Exception as e:
        print(f"Error calculating radon metrics: {e}")
        return None, None, None

# Calculate metrics for the 'before' code
df['MI_Before'] = df['Source Code (before)'].apply(
    lambda x: calculate_radon_metrics(x)[0] if calculate_radon_metrics(x) else None
)
df['CC_Before'] = df['Source Code (before)'].apply(
    lambda x: calculate_radon_metrics(x)[1] if calculate_radon_metrics(x) else None
)
df['LOC_Before'] = df['Source Code (before)'].apply(
    lambda x: calculate_radon_metrics(x)[2] if calculate_radon_metrics(x) else None
)

# Calculate metrics for the 'after' code
df['MI_After'] = df['Source Code (current)'].apply(
    lambda x: calculate_radon_metrics(x)[0] if calculate_radon_metrics(x) else None
)
df['CC_After'] = df['Source Code (current)'].apply(
    lambda x: calculate_radon_metrics(x)[1] if calculate_radon_metrics(x) else None
)
df['LOC_After'] = df['Source Code (current)'].apply(
    lambda x: calculate_radon_metrics(x)[2] if calculate_radon_metrics(x) else None
)

# Calculate the changes
df['MI_Change'] = df['MI_After'] - df['MI_Before']
df['CC_Change'] = df['CC_After'] - df['CC_Before']
df['LOC_Change'] = df['LOC_After'] - df['LOC_Before']

mi_changes = df['MI_Change'].dropna()


# Display summary statistics of the changes
print("Summary of Structural Metric Changes:")
print()
print(f"Files successfully analyzed: {len(mi_changes)}/{len(df)}")
print(f"Maintainability Index Change: Mean = {df['MI_Change'].mean():.2f}")
print(f"Cyclomatic Complexity Change: Mean = {df['CC_Change'].mean():.2f}")
print(f"Lines of Code Change: Mean = {df['LOC_Change'].mean():.2f}")

# Remove intermediate columns if desired (keeping only the change columns)
df = df.drop(['MI_Before', 'MI_After', 'CC_Before', 'CC_After', 'LOC_Before', 'LOC_After'], axis=1)

# Save the updated dataframe
df.to_csv('detailed_analysis_with_radon.csv', index=False)
print("\nAnalysis complete. Results saved to 'detailed_analysis_with_radon.csv'")