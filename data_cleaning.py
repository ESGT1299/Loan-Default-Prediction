import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
import joblib

def preprocess_data(file_path, target_column="loan_status", output_file="dataset/cleaned_data.csv"):
    """
    Cleans and preprocesses the dataset
    1. Drops irrelevant columns and those with excessive missing data.
    2. Encodes categorical variables (dynamically handles high-cardinality columns).
    3. Selects numerical features with significant correlation to the target variable.
    4. Scales numerical features.

    Args:
    - file_path: Path to the raw dataset file
    - target_column: Name of the target column for correlation-based feature selection.
    - output_file: Path to save the cleaned dataset.

    Returns:
    - data: Cleaned and preprocessed dataset as a Pandas DataFrame
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path, low_memory=False)

    print(f"Columns in the dataset: {data.columns.tolist()}")

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    target_data = data[target_column]

    # Drop columns with more than 50% missing values
    print("Dropping columns with excessive missing data...")
    data = data.dropna(axis=1, thresh=0.5 * len(data))

    # Ensure target column isn't dropped
    if target_column not in data.columns:
        data[target_column] = target_data

    print("Dropping constant or near-constant columns...")
    constant_columns = [col for col in data.columns if data[col].nunique() <= 1 and col != target_column]
    data = data.drop(columns=constant_columns)

    print("Handling high-cardinality categorical columns...")
    high_cardinality_columns = [col for col in data.columns if data[col].nunique() > 50]
    for col in high_cardinality_columns:
        print(f"Encoding high-cardinality column: {col}")
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    print("Applying one-hot encoding to other categorical columns...")
    categorical_columns = [col for col in data.select_dtypes(include=['object']).columns if col not in high_cardinality_columns and col != target_column]
    if categorical_columns:
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    print("Filling missing values...")
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].mean())
        elif data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])

    print("Encoding target column 'loan_status' into numerical values...")
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    # Save label mapping for later use
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    joblib.dump(label_mapping, "label_mapping.pkl")
    print("Label mapping saved:", label_mapping)

    print("Selecting significant features based on correlation with target...")
    correlation_matrix = data.corr()
    significant_features = correlation_matrix[target_column][correlation_matrix[target_column].abs() > 0.1].index
    data = data[significant_features]

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    print("Scaling numerical features...")
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    data = X.copy()
    data[target_column] = y

    print(f"Saving cleaned data to {output_file}...")
    data.to_csv(output_file, index=False)
    print("Data cleaning and preprocessing complete.")

    return data

# Run the cleaning process
raw_file_path = "dataset/accepted_2007_to_2018q4.csv/accepted_2007_to_2018q4.csv"
cleaned_file_path = "dataset/cleaned_data.csv"
target_column_name = "loan_status"

preprocess_data(raw_file_path, target_column=target_column_name, output_file=cleaned_file_path)
