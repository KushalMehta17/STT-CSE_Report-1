import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('./ragas_analysis.csv')

# Display basic info about the dataset
print(f"Total file modifications analyzed: {len(df)}")

# 2. Basic Statistics
total_modifications = len(df)
total_discrepancies = df['discrepancy'].eq('Yes').sum()
discrepancy_percentage = (total_discrepancies / total_modifications) * 100

print("Overall Discrepancy Statistics:")
print(f"  - Total File Modifications: {total_modifications}")
print(f"  - Total Discrepancies (Yes): {total_discrepancies}")
print(f"  - Overall Discrepancy Rate: {discrepancy_percentage:.2f}%")
print("\n" + "="*50 + "\n")

# 3. Statistics by File Type 
print("Discrepancy Statistics by File Type:")

grouped_stats = df.groupby('file_type')['discrepancy'].apply(
    lambda x: pd.Series({
        'Total_Modifications': len(x),
        'Discrepancies': x.eq('Yes').sum(),
        'Discrepancy_Rate': (x.eq('Yes').sum() / len(x)) * 100
    })
).unstack()

# Print the statistics in a clean table
print(grouped_stats.round(2))
print("\n" + "="*50 + "\n")


# 4. Prepare Data for Plots
mismatch_counts = df[df['discrepancy'] == 'Yes']['file_type'].value_counts()
required_categories = ['Source', 'Test', 'README', 'LICENSE']
plot_series = pd.Series({cat: mismatch_counts.get(cat, 0) for cat in required_categories})

# 5. Generate the Plot
print("Generating plot...")
plt.figure(figsize=(10, 6))
bars = plt.bar(plot_series.index, plot_series.values, color='skyblue')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom')

plt.title('Number of Diff Algorithm Mismatches by File Type', fontsize=14)
plt.xlabel('File Type')
plt.ylabel('Number of Mismatches (Discrepancy = "Yes")')
plt.tight_layout() 

# Save the plot to the data directory
plot_path = './mismatches_plot_ragas.png'
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")

plt.show()

print("\nAnalysis complete!")