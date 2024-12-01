from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib

# Load the cleaned dataset
data = pd.read_csv("dataset/cleaned_data.csv")
#data_sampled = data.sample(frac=0.2, random_state=42) # To reduce the size of the dataset to test the funcionality

# Define features and target
# X = data_sampled.drop('loan_status', axis=1)
# y = data_sampled['loan_status']

X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE with adjusted parameters
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Compute class weights
classes = y.unique()
class_weights = compute_class_weight('balanced', classes=classes, y=y)
weights_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

# Train a weighted Random Forest model
model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=weights_dict)
model.fit(X_train_resampled, y_train_resampled)

# Get feature importance and select top features
feature_importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Select top N features (e.g., 10)
top_features = feature_importances["Feature"].head(10).tolist()
X_train_reduced = X_train_resampled[top_features]
X_test_reduced = X_test[top_features]

# Retrain the model with reduced features
model_reduced = RandomForestClassifier(random_state=42)
model_reduced.fit(X_train_reduced, y_train_resampled)

# Evaluate the model
y_pred_reduced = model_reduced.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred_reduced)
print(f"Accuracy with reduced features: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred_reduced))

# Save the reduced feature set and model
with open("top_features.pkl", "wb") as f:
    joblib.dump(top_features, f)

joblib.dump(model_reduced, "loan_default_model_reduced.pkl")
print("Model and features saved.")

