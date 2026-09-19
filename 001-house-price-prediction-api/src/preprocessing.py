import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def create_preprocessing_pipeline():
    """
    Create and return preprocessing pipeline
    """

    # Define column types
    cat_cols = ["waterfront", "zipcode", "view", "condition", "grade"]
    nominal_cols =["waterfront", "zipcode"]
    ord_cols = ["view", "condition", "grade"]
    num_cols = ["bedrooms", "bathrooms", "floors", "sqft_living","sqft_lot", "sqft_basement", "sqft_above", "sqft_living15", "sqft_lot15", "lat", "long", "yr_built", "yr_renovated"]

    # Numerical pipeline
    num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
    ("standardize", StandardScaler()),
    ])

    # categorical pipeline
    encoder = ColumnTransformer([
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
    unknown_value=-1), ord_cols),
    ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols),
    ])  

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent").set_output(transform="pandas")),
        ("encode", encoder)
    ])

    # full pipeline
    preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, ord_cols + nominal_cols),
    ]).set_output(transform="pandas")

    return preprocessing, num_cols, cat_cols

def prepare_data(data_path="../data/raw/kc_house_data.csv", test_size=0.2, random_state=42):
    """Load data, split, and fit preprocessing pipeline"""

    # Load data
    df = pd.read_csv(data_path)
    
    # Split features and target
    y = df["price"]
    X = df.drop("price", axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create and fit preprocessing pipeline
    preprocessing, num_cols, cat_cols = create_preprocessing_pipeline()
    X_train_prepared = preprocessing.fit_transform(X_train)
    X_test_prepared = preprocessing.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_prepared': X_train_prepared,
        'X_test_prepared': X_test_prepared,
        'preprocessing': preprocessing,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }

if __name__ == "__main__":
    # Prepare data
    data = prepare_data()
    
    # Export preprocessing pipeline
    joblib.dump(data['preprocessing'], '../models/preprocessing.pkl')
    print("✓ Preprocessing pipeline saved to ../models/preprocessing.pkl")
    
    # Export column info (useful for API validation)
    column_info = {
        'num_cols': data['num_cols'],
        'cat_cols': data['cat_cols']
    }
    joblib.dump(column_info, '../models/column_info.pkl')
    print("✓ Column info saved to ../models/column_info.pkl")
    
    print(f"\nData shapes:")
    print(f"X_train_prepared: {data['X_train_prepared'].shape}")
    print(f"X_test_prepared: {data['X_test_prepared'].shape}")
