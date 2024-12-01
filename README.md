# **Loan Default Prediction Project**

## **Overview**
This project predicts the likelihood of loan defaults based on a dataset of historical loan data. The dataset was sourced from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/wordsforthewise/lending-club)), and the project uses **Python**, **Scikit-learn**, and **Streamlit** to preprocess data, build predictive models, and create an interactive dashboard.

---

## **Table of Contents**
1. [Project Objectives](#project-objectives)
2. [Dataset Description](#dataset-description)
3. [Key Methodologies](#key-methodologies)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Interactive Dashboard](#interactive-dashboard)
7. [Project Structure](#project-structure)
8. [How to Run the Project](#how-to-run-the-project)
9. [Future Enhancements](#future-enhancements)


---

## **Project Objectives**
- Analyze and preprocess loan data to identify significant features.
- Build a predictive model to classify loans as "Fully Paid," "Charged Off," or other categories.
- Deploy an interactive dashboard using **Streamlit**.

---

## **Dataset Description**
The dataset contains over 2 million records with details on loans issued by LendingClub from 2007 to 2018. Key features include:
- `loan_amnt`: The amount of the loan.
- `int_rate`: The interest rate of the loan.
- `loan_status`: The status of the loan, such as "Fully Paid" or "Charged Off."

### **Source**: [Kaggle LendingClub Dataset]([https://www.kaggle.com/](https://www.kaggle.com/datasets/wordsforthewise/lending-club))

---

## **Key Methodologies**
1. **Data Cleaning**:
   - Removed columns with more than 50% missing values.
   - Imputed missing values using mean and mode for numerical and categorical features, respectively.
2. **Feature Engineering**:
   - Encoded categorical variables using one-hot encoding and label encoding.
   - Selected the top features based on correlation with the target variable.
3. **Model Training**:
   - Used **SMOTE** for handling class imbalance.
   - Trained a **Random Forest Classifier** for predictions.
4. **Interactive Dashboard**:
   - Created a dashboard using **Streamlit** for user interaction.

---

## **Data Preprocessing**
Key preprocessing steps:
1. Dropped irrelevant and sparse columns.
2. Imputed missing values.
3. Encoded categorical variables:
   - Label Encoding for high-cardinality features.
   - One-Hot Encoding for low-cardinality features.
4. Scaled numerical features using `StandardScaler`.
5. Selected features with correlation > 0.1 with the target column (`loan_status`).

---

## **Model Training and Evaluation**
### **Model Selection**:
- **Random Forest Classifier** was chosen for its robustness and ability to handle mixed data types.

### **Evaluation Metrics**:
- Accuracy: 97.16%
- Precision, Recall, and F1-Score: Evaluated per class.

### **Feature Importance**:
The top 10 features contributing to the model:
- `total_pymnt`
- `int_rate`
- `loan_amnt`
- ...

---

## **Interactive Dashboard**
The dashboard allows users to:
1. Enter values for the top features.
2. Predict the loan's status.
3. View results as "Fully Paid," "Charged Off," or other categories.

---

## **Project Structure**
Proyect 3/ 
├── dataset/
    ├── accepted_2007_to_2018Q4.csv 
├── data_cleaning.py 
├── data_exploration.py 
├── model_training.py 
├── app.py 
├── README.md 
├── requirements.txt 
└── venv/

## **How to Run the Project**
### **Step 1**: Clone the Repository
```bash
git clone https://github.com/ESGT1299/Loan-Default-Prediction.git
cd Loan-Default-Prediction
```

### **Step 2**: Install Dependencies
```bash
pip install -r requirements.txt
```

### **Step 3**: Run the Streamlit Dashboard
```bash
streamlit run app.py
```

### **Step 4**: Interact with the dashboard

Provide input values and view predictions

## **Future Enhancements**
- Integrate SQL for database management.
- Create visualizations using Tableau.
- Improve model performance by experimenting with XGBoost or LightGBM.
