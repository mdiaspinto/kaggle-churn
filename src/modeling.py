"""
Modeling functions for churn prediction.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb


def prepare_training_data(df_train, target_col='churn_status', exclude_cols=None):
    """
    Prepare training data by separating features and target.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with features and target column
    target_col : str
        Name of the target column
    exclude_cols : list, optional
        Columns to exclude from features (e.g., ['userId', 'date', 'churn_status'])
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target series
    feature_cols : list
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['userId', 'date', target_col]
    
    # Drop rows with NaN in target
    df_model = df_train.dropna(subset=[target_col]).copy()
    print(f"Training data shape after dropping NaN targets: {df_model.shape}")
    
    # Get feature columns
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]

    # One-hot encode categorical features and fill remaining NaNs
    obj_cols = [col for col in feature_cols if df_model[col].dtype == 'object']
    X = pd.get_dummies(df_model[feature_cols], columns=obj_cols, drop_first=True).fillna(0)
    feature_cols = X.columns.tolist()
    y = df_model[target_col].astype(int)
    
    print(f"\nFeatures: {len(feature_cols)} columns")
    print(f"Feature columns: {feature_cols}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    return X, y, feature_cols


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, 
                        random_state=42, class_weight='balanced', n_jobs=-1, **kwargs):
    """
    Train a Random Forest classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum depth of trees
    random_state : int
        Random seed
    class_weight : str or dict
        'balanced' to adjust weights inversely proportional to class frequencies
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    **kwargs : additional arguments to pass to RandomForestClassifier
    
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        n_jobs=n_jobs,
        class_weight=class_weight,
        **kwargs
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        Test target
    
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print("\n=== Model Performance ===")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    results = {
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def get_feature_importance(model, feature_cols, top_n=10):
    """
    Get feature importance from trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_cols : list
        List of feature names
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    feature_importance : pd.DataFrame
        Dataframe with features and their importance scores
    """
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance.head(top_n))
    
    return feature_importance


def prepare_test_data(df_test_agg, feature_cols, last_date=None):
    """
    Prepare test data for prediction.
    
    Parameters:
    -----------
    df_test_agg : pd.DataFrame
        Aggregated test dataframe
    feature_cols : list
        List of feature columns used in training
    last_date : date, optional
        If provided, use most recent data up to this date
    
    Returns:
    --------
    X_test : pd.DataFrame
        Test features in correct order and format
    df_test_final : pd.DataFrame
        Test dataframe with predictions metadata (userId, etc.)
    """
    if last_date is None:
        last_date = df_test_agg['date'].max()
    
    print(f"Last date in test data: {last_date}")
    
    # Get most recent record for each user before or on last_date
    df_test_final = df_test_agg[df_test_agg['date'] <= last_date].sort_values('date').groupby('userId', as_index=False).tail(1)
    
    print(f"Number of users for prediction: {len(df_test_final)}")
    
    # One-hot encode categorical columns to mirror training encoding
    obj_cols = [col for col in df_test_final.columns if df_test_final[col].dtype == 'object']
    df_test_encoded = pd.get_dummies(df_test_final, columns=obj_cols, drop_first=True)

    # Reindex to training feature columns (add missing as 0, drop extras)
    X_test = df_test_encoded.reindex(columns=feature_cols, fill_value=0).fillna(0)
    
    print(f"Using {len(feature_cols)} features for prediction")
    print(f"Test data shape for prediction: {X_test.shape}")
    
    return X_test, df_test_final


def make_predictions(model, X_test):
    """
    Make predictions on test data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame or np.ndarray
        Test features
    
    Returns:
    --------
    predictions : np.ndarray
        Class predictions (0 or 1)
    probabilities : np.ndarray
        Prediction probabilities for class 1
    """
    print("Making predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction distribution:")
    print(pd.Series(predictions).value_counts())
    
    return predictions, probabilities


def create_submission(user_ids, predictions, output_path=None):
    """
    Create submission dataframe.
    
    Parameters:
    -----------
    user_ids : array-like
        User IDs for predictions
    predictions : array-like
        Model predictions
    output_path : str, optional
        Path to save submission CSV
    
    Returns:
    --------
    submission : pd.DataFrame
        Submission dataframe with 'id' and 'target' columns
    """
    submission = pd.DataFrame({
        'id': user_ids,
        'target': predictions
    })
    
    print(f"\nSubmission dataframe shape: {submission.shape}")
    print(f"Sample submissions:")
    print(submission.head(10))
    print(f"Submission target distribution:")
    print(submission['target'].value_counts())
    
    if output_path:
        submission.to_csv(output_path, index=False)
        print(f"Saved submission to {output_path}")
    
    return submission


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, 
                  learning_rate=0.1, random_state=42, scale_pos_weight=None, 
                  n_jobs=-1, **kwargs):
    """
    Train an XGBoost classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum depth of trees
    learning_rate : float
        Learning rate (eta)
    random_state : int
        Random seed
    scale_pos_weight : float, optional
        Balancing of positive and negative weights. If None, automatically calculated
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    **kwargs : additional arguments to pass to XGBClassifier
    
    Returns:
    --------
    model : XGBClassifier
        Trained model
    """
    # Calculate scale_pos_weight if not provided (for class imbalance)
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        eval_metric='logloss',
        **kwargs
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model
