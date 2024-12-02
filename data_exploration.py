import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset/accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv')

# View the first few rows
print(data.head())

# Understand the data
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Set plot size and style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plot histogram
sns.histplot(data['loan_amnt'], bins=30, color="skyblue", kde=True)

# Add titles and labels
plt.title("Loan Amount Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Loan Amount ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Save the plot as an image for your portfolio
plt.savefig("Loan_Amount_Distribution.png", dpi=300)
plt.show()


# Filter numerical columns only
numerical_data = data.select_dtypes(include=["Float64","int64"])

# Calculate correlation matrix
correlation_matrix = numerical_data.corr()

# Set plot size and style
plt.figure(figsize=(18, 14))
sns.set_style("whitegrid")

# Create heatmap with annotations
sns.heatmap(
    correlation_matrix,
    annot=False,  # Set to True if you want numbers on the heatmap
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Correlation Coefficient'}
)

# Add titles and labels
plt.title("Correlation Heatmap", fontsize=18, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and align x-axis labels to the right
plt.yticks(fontsize=10)  # Keep y-axis labels unchanged
plt.tight_layout()  # Automatically adjust padding to prevent overlap

# Save the plot as an image for your portfolio
plt.savefig("Correlation_Heatmap.png", dpi=300)
plt.show()