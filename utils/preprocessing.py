import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def load_and_clean_data(file_path):
    # Load the data
    df = pd.read_excel(file_path)

    # Handle missing values in Income by filling with the median
    df['Income'].fillna(df['Income'].median(), inplace=True)

    # Convert Dt_Customer to datetime format and calculate customer tenure
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Customer_Since'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days

    # Feature engineering: Total children
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']

    # Drop irrelevant columns
    df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], inplace=True)

    return df

def encode_and_scale(df):
    # Label Encoding for categorical variables
    le = LabelEncoder()
    df['Education'] = le.fit_transform(df['Education'])
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

    # Scaling numerical features
    scaler = StandardScaler()
    df[['Income', 'Customer_Since', 'Total_Children', 'MntWines', 'MntFruits', 'MntMeatProducts', 
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']] = scaler.fit_transform(
        df[['Income', 'Customer_Since', 'Total_Children', 'MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']])

    return df
