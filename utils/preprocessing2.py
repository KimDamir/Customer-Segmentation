import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def load_and_clean_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Handle missing values in Income by filling with the median
    df['Income'].fillna(df['Income'].median(), inplace=True)

    # Convert Dt_Customer to datetime format and calculate customer tenure

    # Feature engineering: Total children

    # Drop irrelevant columns
    df.drop(columns=['ID'], inplace=True)

    return df

def encode_and_scale(df):
    # Label Encoding for categorical variables
    le = LabelEncoder()
    df['Education_level'] = le.fit_transform(df['Education_level'])
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

    # Scaling numerical features
    scaler = StandardScaler()
    df[['Age', 'Income', 'Total_Spend',
                    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'Total_Campaigns_Accepted']] = scaler.fit_transform(
        df[['Age', 'Income', 'Total_Spend',
                    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'Total_Campaigns_Accepted']])


    return df