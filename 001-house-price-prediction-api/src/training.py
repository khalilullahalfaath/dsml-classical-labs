# training.py
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing pipeline
from preprocessing import create_preprocessing_pipeline


def train_model(data_path="../data/raw/kc_house_data.csv", 
                test_size=0.2, 
                random_state=42):
    """Train Random Forest model with preprocessing pipeline"""
    
    print("=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape}")
    
    # 2. Split features and target
    print("\n[2/7] Splitting features and target...")
    y = df["price"]
    X = df.drop("price", axis=1)
    
    # 3. Train-test split
    print("\n[3/7] Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✓ Train set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # 4. Create preprocessing pipeline
    print("\n[4/7] Creating preprocessing pipeline...")
    preprocessing, num_cols, cat_cols = create_preprocessing_pipeline()
    
    # 5. Create full pipeline (preprocessing + model)
    print("\n[5/7] Training Random Forest model...")
    rf_pipeline = make_pipeline(
        preprocessing,
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    )
    
    rf_pipeline.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # 6. Evaluate model
    print("\n[6/7] Evaluating model...")
    y_train_pred = rf_pipeline.predict(X_train)
    y_test_pred = rf_pipeline.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        "model": "RandomForestRegressor",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_metrics": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2)
        },
        "test_metrics": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        },
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": random_state
        }
    }
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTrain Metrics:")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE:  ${train_mae:,.2f}")
    print(f"  R²:   {train_r2:.4f}")
    print(f"\nTest Metrics:")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE:  ${test_mae:,.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # 7. Feature importances
    print("\n[7/7] Generating feature importances...")
    feature_names = rf_pipeline[:-1].get_feature_names_out()
    importances = rf_pipeline[-1].feature_importances_
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_imp_df.head(10).to_string(index=False))
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)
    
    # Save model
    model_path = "../models/rf_model.pkl"
    joblib.dump(rf_pipeline, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save metrics
    metrics_path = "../models/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save feature importances
    feature_imp_path = "../models/feature_importances.csv"
    feature_imp_df.to_csv(feature_imp_path, index=False)
    print(f"✓ Feature importances saved: {feature_imp_path}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    # Plot 1: Feature Importances
    plt.figure(figsize=(10, 8))
    top_n = 15
    top_features = feature_imp_df.head(top_n)
    plt.barh(range(top_n), top_features['Importance'])
    plt.yticks(range(top_n), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../models/feature_importances.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importances plot saved: ../models/feature_importances.png")
    plt.close()
    
    # Plot 2: Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('../models/residual_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Residual plot saved: ../models/residual_plot.png")
    plt.close()
    
    # Plot 3: Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Prices')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../models/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Predicted vs Actual plot saved: ../models/predicted_vs_actual.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return rf_pipeline, metrics, feature_imp_df


if __name__ == "__main__":
    model, metrics, feature_importances = train_model()