import streamlit as st # type: ignore
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
import pandas as pd # type: ignore
from sklearn.impute import KNNImputer # type: ignore

def clean_testdata(data):
    # Convert object columns to numeric, handle non-convertible values with NaN
    for feature in data.select_dtypes(include=['object']).columns:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
    
    # Drop rows with non-convertible string values
    data = data.dropna()
    
    # Impute missing values using KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    return data

def detect_fraud(new_data, trained_model):
    # Preprocess the new dataset similarly to the training data
    scaler = StandardScaler()
    new_data['Time'] = scaler.fit_transform(new_data[['Time']])
    
    minmaxscaler = MinMaxScaler()
    new_data['Amount'] = minmaxscaler.fit_transform(new_data[['Amount']])
    
    new_data = clean_testdata(new_data)  # Ensure this function doesn't drop/alter necessary columns

    # Extract features for prediction (excluding any non-feature columns like 'Class')
    features = new_data.drop(columns=['Class'], errors='ignore')  # Drop 'Class' if present

    # Apply PCA transformation using the same number of components as during training
    pca = PCA(n_components=20)  
    pca_features = pca.fit_transform(features)

    # Convert the PCA-transformed features to a DataFrame with column names
    pca_columns = [f'PCA{i+1}' for i in range(20)]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns)

    # Predict anomalies using the trained model
    predictions = trained_model.predict(pca_df)
    
    # Convert predictions (-1 for anomaly, 1 for normal) to binary classification (1 for fraud, 0 for normal)
    fraud_predictions = pd.Series(predictions, index=new_data.index).apply(lambda x: 1 if x == -1 else 0)
    
    # Add the first two PCA features and predictions to the original data for visualization
    new_data[['PCA1', 'PCA2']] = pca_df[['PCA1', 'PCA2']]
    new_data['Fraud'] = fraud_predictions

    # Return a DataFrame of the fraudulent transactions
    fraud_transactions = new_data[new_data['Fraud'] == 1]
    
    return fraud_transactions

# Load the trained model (make sure to train and save your model beforehand)
with open('if_model_str.pkl', 'rb') as f:
    trained_model = pickle.load(f)

def main():
    st.title("Credit Card Fraud Detection App")
    
    # Upload a file
    uploaded_file = st.file_uploader("Upload a CSV or Excel file with credit card transactions", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Check file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            new_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            new_data = pd.read_excel(uploaded_file, sheet_name='creditcard_test')
        
        # Display the uploaded file
        st.write("Uploaded Dataset:")
        st.write(new_data.head())
        
        # Run fraud detection
        fraud_transactions = detect_fraud(new_data, trained_model)
        
        # Display the results
        st.write("Detected Fraudulent Transactions:")
        st.write(fraud_transactions)
        
        # Visualization (scatter plot for detected anomalies)
        if not fraud_transactions.empty:
            st.write("Visualization of Detected Anomalies:")
            st.scatter_chart(fraud_transactions[['PCA1', 'PCA2']])

if __name__ == '__main__':
    main()
