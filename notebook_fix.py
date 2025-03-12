"""
Helper function to fix XGBoost feature order mismatch.
Add this to your notebook to resolve the feature order issue.
"""

def fix_feature_order(df, xgb_model):
    """
    Reorders dataframe columns to match the feature order expected by an XGBoost model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe whose columns need to be reordered
    xgb_model : XGBoost model
        The XGBoost model that will be used for prediction
        
    Returns:
    --------
    pandas.DataFrame
        A new dataframe with columns ordered to match the model's expected order
    """
    try:
        # Get feature names from model
        expected_features = xgb_model.get_booster().feature_names
        print(f"Model expects {len(expected_features)} features")
        
        # Check if all expected features are in the dataframe
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Dataframe is missing these expected features: {missing_features}")
            
        # Check if dataframe has extra features not in the model
        extra_features = [f for f in df.columns if f not in expected_features]
        if extra_features:
            print(f"Warning: Dataframe contains extra features not in the model: {extra_features}")
        
        # Return reordered dataframe
        return df[expected_features]
    except Exception as e:
        print(f"Error getting feature names: {e}")
        # Fallback - try to extract feature names from error message
        if hasattr(e, 'args') and len(e.args) > 0 and 'feature_names mismatch' in str(e.args[0]):
            error_msg = str(e.args[0])
            expected_order = error_msg.split('[')[1].split(']')[0]
            expected_features = [f.strip(" '") for f in expected_order.split(',')]
            print(f"Extracted {len(expected_features)} features from error message")
            return df[expected_features]
        raise e

# Example usage in notebook:
#
# # Add this code before making predictions with xgb_amt
# from notebook_fix import fix_feature_order
# amount_pred_df = fix_feature_order(amount_pred_df, xgb_amt)
# xgb_amount = xgb_amt.predict(amount_pred_df) 