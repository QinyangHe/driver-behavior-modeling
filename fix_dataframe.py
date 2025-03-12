"""
Direct fix for the feature order mismatch in the PrescriptiveUBI notebook.
Run this in a notebook cell after creating amount_pred_df and before calling xgb_amt.predict.
"""

def fix_amount_pred_df(amount_pred_df):
    """
    Reorders the amount_pred_df dataframe columns to match the expected order from the error message.
    """
    # This is the specific order that was shown in the error message as the model's expected order
    expected_order = ['Duration', 'Insured.age', 'Car.age', 'Credit.score', 'Annual.miles.drive', 
                     'Years.noclaims', 'Annual.pct.driven', 'Total.miles.driven', 'Pct.drive.mon', 
                     'Pct.drive.tue', 'Pct.drive.wed', 'Pct.drive.thr', 'Pct.drive.fri', 
                     'Pct.drive.sat', 'Pct.drive.sun', 'Pct.drive.2hrs', 'Pct.drive.3hrs', 
                     'Pct.drive.4hrs', 'Pct.drive.wkday', 'Pct.drive.wkend', 'Pct.drive.rush am', 
                     'Pct.drive.rush pm', 'Avgdays.week', 'Accel.06miles', 'Accel.08miles', 
                     'Accel.09miles', 'Accel.11miles', 'Accel.12miles', 'Accel.14miles', 
                     'Brake.06miles', 'Brake.08miles', 'Brake.09miles', 'Brake.11miles', 
                     'Brake.12miles', 'Brake.14miles', 'Left.turn.intensity08', 
                     'Left.turn.intensity09', 'Left.turn.intensity10', 'Left.turn.intensity11', 
                     'Left.turn.intensity12', 'Right.turn.intensity08', 'Right.turn.intensity09', 
                     'Right.turn.intensity10', 'Right.turn.intensity11', 'Right.turn.intensity12', 
                     'NB_Claim', 'Insured.sex_Male', 'Marital_Single', 'Car.use_Commute', 
                     'Car.use_Farmer', 'Car.use_Private', 'Region_Urban']
    
    # Verify all expected columns are in the dataframe
    missing_cols = [col for col in expected_order if col not in amount_pred_df.columns]
    if missing_cols:
        print(f"Warning: These expected columns are missing from the dataframe: {missing_cols}")
        print("Please ensure all required features are present.")
        return amount_pred_df  # Return original if can't fix
    
    print("Reordering dataframe columns to match the model's expectations...")
    return amount_pred_df[expected_order]

# Example usage in notebook:
# 
# # After creating amount_pred_df but before using xgb_amt.predict():
# from fix_dataframe import fix_amount_pred_df
# amount_pred_df = fix_amount_pred_df(amount_pred_df)
# # Now use the model
# xgb_amount = xgb_amt.predict(amount_pred_df) 