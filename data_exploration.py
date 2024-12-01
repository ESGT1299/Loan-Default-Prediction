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

# Visualize distribution of important columns
sns.histplot(data['loan_amnt'],bins=30)
plt.title("Loan Amount Distribution")
plt.show()

# Filter numerical columns only
numerical_data = data.select_dtypes(include=["Float64","int64"])

# Correlation heatmap
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=False,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()