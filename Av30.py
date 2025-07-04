import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import catboost as cb
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import holidays
import warnings
import os
import gc
from joblib import parallel_backend
from tqdm import tqdm

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)
USE_LOG_TRANSFORM = True
MODEL_PARALLEL = True  # Set to False if memory constrained

# Load data with optimized dtypes
def load_data():
    """Load and validate data files with proper error handling"""
    dtypes = {
        'srcid': 'float32', 'destid': 'float32',
        'cumsum_seatcount': 'float32', 'cumsum_searchcount': 'float32',
        'final_seatcount': 'float32', 'dbd': 'float32',
        'srcid_region': 'object', 'destid_region': 'object',
        'srcid_tier': 'object', 'destid_tier': 'object'
    }
    
    print("Loading data...")
    
    # Check if files exist
    data_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    required_files = [
        os.path.join(data_dir, 'train.csv'),
        os.path.join(data_dir, 'test.csv'),
        os.path.join(data_dir, 'transactions.csv')
    ]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file {file} not found")
    
    # Define columns to load
    train_cols = ['doj', 'route_key', 'srcid', 'destid', 'cumsum_seatcount', 'cumsum_searchcount',
                  'final_seatcount', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
    test_cols = ['doj', 'route_key', 'srcid', 'destid', 'cumsum_seatcount', 'cumsum_searchcount',
                 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
    transactions_cols = ['doj', 'srcid', 'destid', 'dbd']

    # Load with error handling
    try:
        train = pd.read_csv(os.path.join(data_dir, "train.csv"), dtype=dtypes, usecols=train_cols)
        test = pd.read_csv(os.path.join(data_dir, "test.csv"), dtype=dtypes, usecols=test_cols)
        transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), dtype={'dbd': 'float32'}, usecols=transactions_cols)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Validate data
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Transactions shape: {transactions.shape}")
    
    # Check for required columns
    if 'final_seatcount' not in train.columns:
        raise ValueError("Target variable 'final_seatcount' not found in training data")
    
    return train, test, transactions

# Improved feature engineering with better missing value handling
def create_features(df, transactions_df, is_train=True, label_encoders=None):
    """Create features with improved missing value handling and validation"""
    print(f"Creating features for {'train' if is_train else 'test'} data...")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    df = df.copy()  # Avoid modifying original dataframe
    
    # Ensure doj is datetime
    df["doj"] = pd.to_datetime(df["doj"], errors='coerce')
    transactions_df["doj"] = pd.to_datetime(transactions_df["doj"], errors='coerce')
    
    # Check for invalid dates
    if df["doj"].isnull().any():
        print(f"Warning: {df['doj'].isnull().sum()} invalid dates found and will be handled")
    
    # Merge transactions with better error handling
    transactions_30 = transactions_df[transactions_df["dbd"] <= 30].copy()
    
    # Check if merge keys exist
    merge_cols = ["doj", "srcid", "destid"]
    for col in merge_cols:
        if col not in df.columns or col not in transactions_30.columns:
            print(f"Warning: Column {col} missing for merge operation")
    
    merged = pd.merge(df, transactions_30[['doj', 'srcid', 'destid', 'dbd']], 
                     on=merge_cols, how="left")

    # Date features with null handling
    merged["day"] = merged["doj"].dt.day.fillna(15).astype('int8')  # Fill with median day
    merged["month"] = merged["doj"].dt.month.fillna(6).astype('int8')  # Fill with median month
    merged["weekday"] = merged["doj"].dt.weekday.fillna(3).astype('int8')  # Fill with Wednesday
    merged["is_weekend"] = merged["weekday"].isin([5, 6]).astype('int8')
    merged["quarter"] = merged["doj"].dt.quarter.fillna(2).astype('int8')

    # Holiday features with better handling
    try:
        in_holidays = holidays.India(years=range(2020, 2026))  # Extended range
        # Get unique dates from both datasets
        all_dates_train = merged['doj'].dropna().unique()
        
        # Load test dates if not training
        if not is_train:
            try:
                test_temp = pd.read_csv(os.path.join(data_dir, "test.csv"), usecols=['doj'])
                test_temp["doj"] = pd.to_datetime(test_temp["doj"], errors='coerce')
                all_dates = np.concatenate([all_dates_train, test_temp['doj'].dropna().unique()])
            except:
                all_dates = all_dates_train
        else:
            all_dates = all_dates_train
        
        holiday_dates = {date for date in all_dates if pd.notna(date) and date.date() in in_holidays}
        merged["is_holiday"] = merged["doj"].isin(holiday_dates).astype('int8')
    except Exception as e:
        print(f"Warning: Holiday feature creation failed: {e}")
        merged["is_holiday"] = 0
    
    merged["season"] = ((merged["month"] % 12 + 3) // 3).astype('int8')

    # IMPROVED categorical features handling
    categorical_cols = ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
    
    for col in categorical_cols:
        if col in merged.columns:
            # Better missing value handling
            merged[col] = merged[col].fillna('Unknown')
            # Create missing indicator
            merged[f'{col}_is_missing'] = merged[col].eq('Unknown').astype('int8')
        else:
            merged[col] = 'Unknown'
            merged[f'{col}_is_missing'] = 1

    # Combined categorical features
    merged["src_dest_region"] = (merged["srcid_region"].astype(str) + "_" + 
                                merged["destid_region"].astype(str))
    merged["src_dest_tier"] = (merged["srcid_tier"].astype(str) + "_" + 
                              merged["destid_tier"].astype(str))

    # Route statistics - calculated only on training data
    route_cache_file = "route_stats.parquet"
    
    if is_train and 'final_seatcount' in merged.columns:
        print("Calculating route statistics...")
        # Remove outliers before calculating stats
        q1 = merged['final_seatcount'].quantile(0.25)
        q3 = merged['final_seatcount'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        clean_data = merged[(merged['final_seatcount'] >= lower_bound) & 
                           (merged['final_seatcount'] <= upper_bound)]
        
        route_stats = clean_data.groupby(['srcid', 'destid'])['final_seatcount'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        route_stats.columns = ['srcid', 'destid', 'route_mean', 'route_median', 
                              'route_std', 'route_min', 'route_max', 'route_count']
        
        # Fill NaN std with 0
        route_stats['route_std'] = route_stats['route_std'].fillna(0)
        route_stats = route_stats.astype({col: 'float32' for col in route_stats.columns if col not in ['srcid', 'destid']})
        
        # Save route stats
        route_stats.to_parquet(route_cache_file)
        print(f"Route statistics saved to {route_cache_file}")

    # Load and merge route stats
    if os.path.exists(route_cache_file):
        route_stats_cached = pd.read_parquet(route_cache_file)
        merged = merged.merge(route_stats_cached, on=['srcid', 'destid'], how='left')
        
        # Fill missing route stats with global statistics
        if is_train and 'final_seatcount' in merged.columns:
            global_mean = merged['final_seatcount'].mean()
            global_std = merged['final_seatcount'].std()
        else:
            global_mean = merged['cumsum_seatcount'].mean() if 'cumsum_seatcount' in merged.columns else 50
            global_std = merged['cumsum_seatcount'].std() if 'cumsum_seatcount' in merged.columns else 20
        
        merged['route_mean'] = merged['route_mean'].fillna(global_mean)
        merged['route_median'] = merged['route_median'].fillna(global_mean)
        merged['route_std'] = merged['route_std'].fillna(global_std)
        merged['route_min'] = merged['route_min'].fillna(0)
        merged['route_max'] = merged['route_max'].fillna(global_mean * 2)
        merged['route_count'] = merged['route_count'].fillna(1)
    else:
        print("Warning: Route statistics file not found. Creating dummy features.")
        for col in ['route_mean', 'route_median', 'route_std', 'route_min', 'route_max', 'route_count']:
            merged[col] = 0.0

    # Additional engineered features
    if 'cumsum_seatcount' in merged.columns:
        max_capacity = merged['cumsum_seatcount'].quantile(0.99)  # Use 99th percentile instead of max
        merged['seats_to_capacity_ratio'] = (merged['cumsum_seatcount'] / (max_capacity + 1e-6)).clip(0, 2)
        merged['search_to_seat_ratio'] = (merged['cumsum_searchcount'] / (merged['cumsum_seatcount'] + 1e-6)).clip(0, 10)
    else:
        merged['seats_to_capacity_ratio'] = 0.0
        merged['search_to_seat_ratio'] = 0.0

    # Fill remaining NaN values
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    categorical_cols_all = merged.select_dtypes(include=['object']).columns
    merged[categorical_cols_all] = merged[categorical_cols_all].fillna('Unknown')

    return merged

# Improved model training functions with better error handling
def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM with optimized parameters"""
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'min_child_samples': 20,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1
    }
    
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(100)
        ]
    )
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with optimized parameters"""
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    return model

def train_catboost(X_train, y_train, X_val, y_val, cat_features):
    """Train CatBoost with optimized parameters"""
    model = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        cat_features=cat_features
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model

def train_sklearn_models(X_train, y_train):
    """Train multiple sklearn models with error handling"""
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
    }

    # Only include models that can handle the data size
    if X_train.shape[0] < 10000:
        models.update({
            'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            'SVR': SVR(C=1.0, kernel='rbf'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), early_stopping=True, random_state=42, max_iter=500)
        })

    trained_models = {}
    for name, model in tqdm(models.items(), desc="Training sklearn models"):
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Failed to train {name}: {e}")
    
    return trained_models

def validate_route_key(df, column_name='route_key'):
    """Check for potential data leakage in route_key"""
    if column_name not in df.columns:
        print(f"Warning: {column_name} not found in data")
        return
    
    sample_keys = df[column_name].dropna().head(10).tolist()
    print(f"Sample route_key values: {sample_keys}")
    
    # Check for date patterns
    date_patterns = ['-', '/', '2020', '2021', '2022', '2023', '2024', '2025']
    potential_leakage = any(any(pattern in str(key) for pattern in date_patterns) for key in sample_keys)
    
    if potential_leakage:
        print("⚠️  WARNING: route_key may contain temporal information that could cause data leakage!")
        print("Consider removing route_key from features or ensuring it's time-independent")
    else:
        print(" route_key appears to be time-independent")

# Main execution with comprehensive error handling
def main():
    try:
        print("Starting ML pipeline...")
        train, test, transactions = load_data()

        # Only keep transactions where dbd == 15
        transactions = transactions[transactions['dbd'] == 15]

        # (Optional) If train has dbd column, filter for dbd == 15
        if 'dbd' in train.columns:
            train = train[train['dbd'] == 15]

        validate_route_key(train)
        train_merged = create_features(train, transactions, is_train=True)
        test_merged = create_features(test, transactions, is_train=False)
        
        # Validate target variable
        if 'final_seatcount' not in train_merged.columns:
            raise ValueError("Target variable 'final_seatcount' not found after feature engineering")
        
        # Check for and handle outliers in target
        target_col = 'final_seatcount'
        q1 = train_merged[target_col].quantile(0.01)
        q99 = train_merged[target_col].quantile(0.99)
        
        print(f"Target variable range: {train_merged[target_col].min():.2f} to {train_merged[target_col].max():.2f}")
        print(f"Target variable 1st-99th percentile: {q1:.2f} to {q99:.2f}")
        
        # Clip extreme outliers
        train_merged[target_col] = train_merged[target_col].clip(lower=q1, upper=q99)
        
        # Target transformation
        if USE_LOG_TRANSFORM:
            # Ensure no negative values before log transform
            min_val = train_merged[target_col].min()
            if min_val <= 0:
                train_merged[target_col] = train_merged[target_col] + abs(min_val) + 1
            y = np.log1p(train_merged[target_col])
        else:
            y = train_merged[target_col]

        # Feature selection and encoding
        categorical_cols = [
            "srcid_region", "destid_region", "srcid_tier", "destid_tier",
            "src_dest_region", "src_dest_tier", "season", "is_weekend", "is_holiday", "quarter"
        ]
        
        # Label encode categorical features
        label_encoders = {}
        for col in categorical_cols:
            if col in train_merged.columns and col in test_merged.columns:
                le = LabelEncoder()
                
                # Combine train and test for consistent encoding
                combined_values = pd.concat([
                    train_merged[col].astype(str),
                    test_merged[col].astype(str)
                ])
                
                le.fit(combined_values)
                train_merged[col] = le.transform(train_merged[col].astype(str))
                test_merged[col] = le.transform(test_merged[col].astype(str))
                label_encoders[col] = le

        # Define features (excluding route_key to prevent data leakage)
        base_features = [
            "srcid", "destid", "cumsum_seatcount", "cumsum_searchcount",
            "day", "month", "weekday", "is_weekend", "is_holiday", "season", "quarter"
        ]
        
        categorical_features = [col for col in categorical_cols if col in train_merged.columns]
        route_features = [
            "route_mean", "route_median", "route_std", "route_min", "route_max", "route_count"
        ]
        engineered_features = [
            "seats_to_capacity_ratio", "search_to_seat_ratio"
        ]
        
        # Missing indicators
        missing_indicators = [col for col in train_merged.columns if col.endswith('_is_missing')]
        
        features = base_features + categorical_features + route_features + engineered_features + missing_indicators
        
        # Filter features that exist in both datasets
        features = [f for f in features if f in train_merged.columns and f in test_merged.columns]
        
        print(f"Using {len(features)} features: {features}")

        # Time-based split for validation
        train_merged = train_merged.sort_values('doj').reset_index(drop=True)
        split_date = train_merged['doj'].quantile(0.8)
        train_mask = train_merged['doj'] < split_date

        X_train = train_merged.loc[train_mask, features]
        y_train = y[train_mask]
        X_val = train_merged.loc[~train_mask, features]
        y_val = y[~train_mask]
        
        print(f"Train set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")

        # Prepare categorical features indices for CatBoost
        cat_features_idx = [features.index(col) for col in categorical_features if col in features]

        # Train models
        all_models = {}
        
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)

        try:
            print("\nTraining LightGBM...")
            all_models['LightGBM'] = train_lightgbm(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"LightGBM training failed: {e}")

        try:
            print("\nTraining XGBoost...")
            all_models['XGBoost'] = train_xgboost(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"XGBoost training failed: {e}")

        try:
            print("\nTraining CatBoost...")
            all_models['CatBoost'] = train_catboost(X_train, y_train, X_val, y_val, cat_features_idx)
        except Exception as e:
            print(f"CatBoost training failed: {e}")

        print("\nTraining sklearn models...")
        sklearn_models = train_sklearn_models(X_train, y_train)
        all_models.update(sklearn_models)

        if not all_models:
            print("Model weights:")
            for i, (name, weight) in enumerate(zip(valid_models.keys(), weights)):
                print(f"{name:15s}: {weight:.3f}")

        # Generate test predictions
        test_preds = []
        X_test = test_merged[features]
        
        for i, (name, model) in enumerate(valid_models.items()):
            try:
                if name == 'LightGBM':
                    pred = model.predict(X_test, num_iteration=model.best_iteration)
                else:
                    pred = model.predict(X_test)

                if USE_LOG_TRANSFORM:
                    pred = np.expm1(pred)
                
                test_preds.append(pred * weights[i])
                print(f"Generated predictions for {name}")
            except Exception as e:
                print(f"Prediction failed for {name}: {e}")

        if not test_preds:
            raise ValueError("No valid test predictions generated!")

        # Final ensemble prediction
        final_pred = np.sum(test_preds, axis=0)

        # Post-processing: clip predictions to reasonable range
        if USE_LOG_TRANSFORM:
            min_seats = np.expm1(y.min())
            max_seats = np.expm1(y.max())
        else:
            min_seats = y.min()
            max_seats = y.max()
        
        final_pred = np.clip(final_pred, min_seats * 0.1, max_seats * 2.0)  # Allow some extrapolation
        
        print(f"\nFinal predictions range: {final_pred.min():.2f} to {final_pred.max():.2f}")

        # Create submission file
        submission = pd.DataFrame({
            "route_key": test_merged["route_key"],
            "final_seatcount": final_pred
        })
        
        # Validate submission
        if submission.isnull().any().any():
            print("Warning: Submission contains null values!")
            submission = submission.fillna(submission['final_seatcount'].median())
        
        submission.to_csv("submission.csv", index=False)
        print(f"\n Submission file created successfully with {len(submission)} predictions!")
        print(f"Submission saved as: submission.csv")
        
        # Display summary statistics
        print(f"\n{'='*50}")
        print("SUMMARY")
        print("="*50)
        print(f"Best individual model RMSE: {min(model_scores[name]['RMSE'] for name in model_scores):.2f}")
        print(f"Models trained successfully: {len(valid_models)}")
        print(f"Features used: {len(features)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test predictions: {len(final_pred)}")

    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

