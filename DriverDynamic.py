import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
from typing import Optional, Union
from pathlib import Path
from torchmetrics import Accuracy, MeanSquaredError, R2Score, Metric
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

affected_features = ['Accel.06miles', 'Accel.08miles', 'Accel.09miles',
       'Accel.11miles', 'Accel.12miles', 'Accel.14miles', 'Brake.06miles',
       'Brake.08miles', 'Brake.09miles', 'Brake.11miles', 'Brake.12miles',
       'Brake.14miles', 'Left.turn.intensity08', 'Left.turn.intensity09',
       'Left.turn.intensity10', 'Left.turn.intensity11',
       'Left.turn.intensity12', 'Right.turn.intensity08',
       'Right.turn.intensity09', 'Right.turn.intensity10',
       'Right.turn.intensity11', 'Right.turn.intensity12']

# def apply_reward(original_df: pd.DataFrame, reward_percentage: float) -> pd.DataFrame:
#     modified_df = original_df.copy()
    
#     # Convert percentage to decimal (e.g., 10% -> 0.1)
#     reduction_factor = reward_percentage / 100

#     # For each feature in affected_features
#     for feature in affected_features:
#         # Create a mask for values greater than 0
#         mask = modified_df[feature] > 0
        
#         # Apply the reduction only to values > 0
#         modified_df.loc[mask, feature] = modified_df.loc[mask, feature] * (1 - reduction_factor)
    
#     return modified_df

def apply_reward(cur_df: pd.DataFrame, reward_percentage: float, original_df, sensitivity_1, sensitivity_2) -> pd.DataFrame:
    affected_features = ['Accel.06miles', 'Accel.08miles', 'Accel.09miles',
       'Accel.11miles', 'Accel.12miles', 'Accel.14miles', 'Brake.06miles',
       'Brake.08miles', 'Brake.09miles', 'Brake.11miles', 'Brake.12miles',
       'Brake.14miles', 'Left.turn.intensity08', 'Left.turn.intensity09',
       'Left.turn.intensity10', 'Left.turn.intensity11',
       'Left.turn.intensity12', 'Right.turn.intensity08',
       'Right.turn.intensity09', 'Right.turn.intensity10',
       'Right.turn.intensity11', 'Right.turn.intensity12']
    affected_df = cur_df.copy()
    for i in range(len(reward_percentage)):
        reduction_factor = reward_percentage[i]
        if original_df.loc[i,"AMT_Claim"] != 0:
            #print(-reduction_factor * sensitivity_1 * affected_df.loc[i,"AMT_Claim"])
            # print(reduction_factor)
            # print(sensitivity_1[i])
            # print(-reduction_factor * sensitivity_1[i])
            affected_df.loc[i,"AMT_Claim"] = -reduction_factor * sensitivity_1[i] * affected_df.loc[i,"AMT_Claim"] + sensitivity_2[i] * (affected_df.loc[i,"AMT_Claim"] - original_df.loc[i,"AMT_Claim"]) + original_df.loc[i,"AMT_Claim"]
        
        affected_df.loc[i,affected_features] = -reduction_factor * sensitivity_1[i] * affected_df.loc[i,affected_features] + sensitivity_2[i] * (affected_df.loc[i, affected_features] - original_df.loc[i,affected_features]) + original_df.loc[i,affected_features]
    
    return affected_df

def process_df(original_df: pd.DataFrame) -> pd.DataFrame:
    driver_df = original_df.copy()
    driver_df = driver_df.drop(columns = ['Territory', 'AMT_Claim'])
    categorical_features = ['Insured.sex','Marital','Car.use','Region']

    # Create the encoder object
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore',sparse_output=False)

    # Fit and transform the categorical columns
    encoded_data = encoder.fit_transform(driver_df[categorical_features])

    # Get feature names after encoding
    feature_names = encoder.get_feature_names_out(categorical_features)

    # Convert the encoded data to a DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=driver_df.index)

    # Drop original categorical columns and concatenate encoded columns
    numerical_features = [col for col in driver_df.columns if col not in categorical_features]
    driver_df = pd.concat([driver_df[numerical_features], encoded_df], axis=1)
    return driver_df

def add_premium(orginal_df: pd.DataFrame, premium_range: list[int]) -> pd.DataFrame:

    # Create a copy of the original DataFrame to avoid modifying it
    result_df = orginal_df.copy()
    
    # Get the min and max values from premium_range
    min_premium, max_premium = premium_range[0], premium_range[1]
    
    # Generate random premium values for each row in the DataFrame
    num_rows = len(orginal_df)
    premium_values = [random.randint(min_premium, max_premium) for _ in range(num_rows)]
    
    # Add the premium column to the DataFrame
    result_df['premium'] = premium_values
    
    return result_df