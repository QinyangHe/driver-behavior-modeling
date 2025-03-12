import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

class SubSimNetwork(pl.LightningModule):
    def __init__(
        self,
        n_hidden_layers: int,
        nodes_first_hidden: int,
        nodes_rest_hidden: int,
        learning_rate: float,
        batch_size: int,
        input_size: int,
        pos_weight: float = 9.0,
        dropout_rate: float = 0.1   # Must specify input size for the data
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        # Initialize accuracy metric for binary classification
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        # Track losses over epoch
        self.training_step_losses = []
        self.validation_step_losses = []
        # Build layers dynamically
        layers = []
        # Input to first hidden layer
        layers.append(nn.Linear(input_size, nodes_first_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        # Rest of hidden layers
        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(
                nodes_first_hidden if i == 0 else nodes_rest_hidden,
                nodes_rest_hidden
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        # Output layer - single node for binary classification
        layers.append(nn.Linear(
            nodes_rest_hidden if n_hidden_layers > 1 else nodes_first_hidden,
            1
        ))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)  # Ensure correct shape
        y_hat = self(x)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight.to(y.device) if self.pos_weight is not None else None)
        
        # Calculate accuracy
        preds = (torch.sigmoid(y_hat) > 0.5).float()
        self.train_accuracy.update(preds, y)
        
        # Store loss for epoch end logging
        self.training_step_losses.append(loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        y_hat = self(x)
        
        # Calculate validation loss
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight.to(y.device) if self.pos_weight is not None else None)
        
        # Calculate validation accuracy
        preds = (torch.sigmoid(y_hat) > 0.5).float()
        self.val_accuracy.update(preds, y)
        
        # Store loss for epoch end logging
        self.validation_step_losses.append(val_loss)
        
        return val_loss

    def on_train_epoch_end(self):
        # Calculate epoch metrics
        epoch_loss = torch.stack(self.training_step_losses).mean()
        epoch_acc = self.train_accuracy.compute()
        
        # Log epoch metrics
        self.log('train_loss', epoch_loss, on_epoch=True, on_step=False)
        self.log('train_acc', epoch_acc, on_epoch=True, on_step=False)
        
        # Reset tracking variables
        self.training_step_losses = []
        self.train_accuracy.reset()
    
    def on_validation_epoch_end(self):
        # Calculate epoch metrics
        epoch_loss = torch.stack(self.validation_step_losses).mean()
        epoch_acc = self.val_accuracy.compute()
        
        # Log epoch metrics
        self.log('val_loss', epoch_loss, on_epoch=True, on_step=False)
        self.log('val_acc', epoch_acc, on_epoch=True, on_step=False)
        
        # Reset tracking variables
        self.validation_step_losses = []
        self.val_accuracy.reset()
    
    def predict_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        predictions = (torch.sigmoid(y_hat) > 0.5).float()
        return predictions
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SymmetricMeanAbsolutePercentageError(Metric):
    """
    Implements SMAPE (Symmetric Mean Absolute Percentage Error) as a torchmetrics Metric
    """
    def __init__(self):
        super().__init__()
        # Initialize states for sum of errors and count
        self.add_state("smape_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update metric states with new predictions and targets
        """
        preds = preds.float()
        target = target.float()
        
        # Calculate SMAPE
        numerator = torch.abs(target - preds)
        denominator = (torch.abs(target) + torch.abs(preds)) / 2
        smape = (numerator / denominator) * 100
        
        # Update states
        self.smape_sum += torch.sum(smape)
        self.total_samples += target.numel()

    def compute(self):
        """
        Compute final SMAPE value
        """
        return self.smape_sum / self.total_samples

class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) loss function
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate SMAPE
        numerator = torch.abs(target - pred)
        denominator = (torch.abs(target) + torch.abs(pred)) / 2
        return torch.mean((numerator / denominator)) * 100

class RegressionNetwork(pl.LightningModule):
    def __init__(
        self,
        input_size,
        n_hidden_layers = 6,
        nodes_first_hidden = 344,
        nodes_rest_hidden = 67,
        learning_rate = 0.000526,
        batch_size = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Initialize regression metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.train_smape = SymmetricMeanAbsolutePercentageError()
        self.val_smape = SymmetricMeanAbsolutePercentageError()
        
        # Initialize SMAPE loss
        self.smape_loss = SMAPELoss()
        
        # Track losses over epoch
        self.training_step_losses = []
        self.validation_step_losses = []
        
        # Build layers dynamically
        layers = []
        # Input to first hidden layer
        layers.append(nn.Linear(input_size, nodes_first_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        
        # Rest of hidden layers
        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(
                nodes_first_hidden if i == 0 else nodes_rest_hidden,
                nodes_rest_hidden
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        
        # Output layer - single node for regression
        layers.append(nn.Linear(
            nodes_rest_hidden if n_hidden_layers > 1 else nodes_first_hidden,
            1
        ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)  # Ensure correct shape
        y_hat = self(x)
        
        # Calculate SMAPE loss
        loss = self.smape_loss(y_hat, y)
        
        # Calculate metrics
        self.train_mse.update(y_hat, y)
        self.train_r2.update(y_hat, y)
        self.train_smape.update(y_hat, y)
        
        # Store loss for epoch end logging
        self.training_step_losses.append(loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        y_hat = self(x)
        
        # Calculate validation loss
        val_loss = self.smape_loss(y_hat, y)
        
        # Calculate metrics
        self.val_mse.update(y_hat, y)
        self.val_r2.update(y_hat, y)
        self.val_smape.update(y_hat, y)
        
        # Store loss for epoch end logging
        self.validation_step_losses.append(val_loss)
        
        return val_loss

    def on_train_epoch_end(self):
        # Calculate epoch metrics
        epoch_loss = torch.stack(self.training_step_losses).mean()
        epoch_mse = self.train_mse.compute()
        epoch_r2 = self.train_r2.compute()
        epoch_smape = self.train_smape.compute()
        
        # Log epoch metrics
        self.log('train_loss', epoch_loss, on_epoch=True, on_step=False)
        self.log('train_mse', epoch_mse, on_epoch=True, on_step=False)
        self.log('train_r2', epoch_r2, on_epoch=True, on_step=False)
        self.log('train_smape', epoch_smape, on_epoch=True, on_step=False)
        
        # Reset tracking variables
        self.training_step_losses = []
        self.train_mse.reset()
        self.train_r2.reset()
        self.train_smape.reset()
    
    def on_validation_epoch_end(self):
        # Calculate epoch metrics
        epoch_loss = torch.stack(self.validation_step_losses).mean()
        epoch_mse = self.val_mse.compute()
        epoch_r2 = self.val_r2.compute()
        epoch_smape = self.val_smape.compute()
        
        # Log epoch metrics
        self.log('val_loss', epoch_loss, on_epoch=True, on_step=False)
        self.log('val_mse', epoch_mse, on_epoch=True, on_step=False)
        self.log('val_r2', epoch_r2, on_epoch=True, on_step=False)
        self.log('val_smape', epoch_smape, on_epoch=True, on_step=False)
        
        # Reset tracking variables
        self.validation_step_losses = []
        self.val_mse.reset()
        self.val_r2.reset()
        self.val_smape.reset()
    
    def predict_step(self, batch, batch_idx):
        x = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

# Create the three architectures specified
def create_subsim1(input_size: int):
    return SubSimNetwork(
        n_hidden_layers=3,
        nodes_first_hidden=353,
        nodes_rest_hidden=68,
        learning_rate=0.000667,
        batch_size=85,
        input_size=input_size,
        pos_weight=15
    )

def create_subsim2(input_size: int):
    return SubSimNetwork(
        n_hidden_layers=3,
        nodes_first_hidden=473,
        nodes_rest_hidden=67,
        learning_rate=0.001019,
        batch_size=18,
        input_size=input_size,
        pos_weight=15
    )

def create_subsim3(input_size: int):
    return SubSimNetwork(
        n_hidden_layers=2,
        nodes_first_hidden=60,
        nodes_rest_hidden=60,
        learning_rate=0.001922,
        batch_size=16,
        input_size=input_size,
        pos_weight=15
    )
class MetricPrinter(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"\nEpoch {epoch}:")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"{key}: {value:.4f}")

class BinaryClassificationDataset(Dataset):
    """Dataset class for binary classification"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Labels (0 or 1)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class BinaryClassificationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for binary classification"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
        random_state: int = 42,
        oversample_minority: bool = True  # Added parameter to control oversampling
    ):
        super().__init__()
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.random_state = random_state
        self.oversample_minority = oversample_minority

    def setup(self, stage: Optional[str] = None):
        """Prepare data splits"""
        
        # First split off test set
        features_temp, self.features_test, labels_temp, self.labels_test = train_test_split(
            self.features,
            self.labels,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=self.labels
        )
        
        # Then split remaining data into train and validation
        self.features_train, self.features_val, self.labels_train, self.labels_val = train_test_split(
            features_temp,
            labels_temp,
            test_size=self.val_split,
            random_state=self.random_state,
            stratify=labels_temp
        )
        
        # Create datasets
        self.train_dataset = BinaryClassificationDataset(self.features_train, self.labels_train)
        self.val_dataset = BinaryClassificationDataset(self.features_val, self.labels_val)
        self.test_dataset = BinaryClassificationDataset(self.features_test, self.labels_test)
        self.whole_dataset = BinaryClassificationDataset(self.features, self.labels)
        # Create sampler for training data if oversampling is enabled
        if self.oversample_minority:
            self.train_sampler = self.get_weighted_sampler(self.labels_train)
        else:
            self.train_sampler = None


    def get_weighted_sampler(self, labels):
        """Create a weighted sampler for the minority class"""
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Calculate weight for each class (inverse of frequency)
        weights_per_class = total_samples / (len(class_counts) * class_counts)
        
        # Assign weights to each sample
        weights = [weights_per_class[label] for label in labels]
        weights = torch.DoubleTensor(weights)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(labels),
            replacement=True
        )
        return sampler

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,  # Use weighted sampler instead of shuffle
            shuffle=self.train_sampler is None,  # Only shuffle if not using sampler
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def whole_dataloader(self):
        return DataLoader(
            self.whole_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
# Example usage
def prepare_data_module(
    df: pd.DataFrame,
    target_column: str,
    batch_size: int,
    oversample_minority: bool = True  # Added parameter
) -> Tuple[BinaryClassificationDataModule, int]:
    """
    Prepare DataModule from a pandas DataFrame with optional oversampling
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Name of target column
        batch_size (int): Batch size for training
        oversample_minority (bool): Whether to oversample minority class
        
    Returns:
        Tuple[BinaryClassificationDataModule, int]: DataModule and input size
    """
    # Separate features and labels
    features = df.drop(columns=[target_column]).values
    labels = df[target_column].values
    
    # Print class distribution before scaling
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for class_label, count in zip(unique, counts):
        print(f"Class {class_label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Get input size
    input_size = features.shape[1]
    
    # Create data module
    data_module = BinaryClassificationDataModule(
        features=features,
        labels=labels,
        batch_size=batch_size,
        oversample_minority=oversample_minority
    )
    
    return data_module, input_size, scaler

class RegressionDataset(Dataset):
    """Dataset class for regression"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class RegressionDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for regression"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
        random_state: int = 42
    ):
        super().__init__()
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.random_state = random_state

    def setup(self, stage: Optional[str] = None):
        """Prepare data splits"""
        
        # First split off test set
        features_temp, self.features_test, targets_temp, self.targets_test = train_test_split(
            self.features,
            self.targets,
            test_size=self.test_split,
            random_state=self.random_state
        )
        
        # Then split remaining data into train and validation
        self.features_train, self.features_val, self.targets_train, self.targets_val = train_test_split(
            features_temp,
            targets_temp,
            test_size=self.val_split,
            random_state=self.random_state
        )
        
        # Create datasets
        self.train_dataset = RegressionDataset(self.features_train, self.targets_train)
        self.val_dataset = RegressionDataset(self.features_val, self.targets_val)
        self.test_dataset = RegressionDataset(self.features_test, self.targets_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def prepare_regression_data_module(
    df: pd.DataFrame,
    target_column: str,
    batch_size: int,
    scale_target: bool = False
) -> Tuple[RegressionDataModule, int, Optional[StandardScaler]]:
    """
    Prepare DataModule from a pandas DataFrame for regression
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Name of target column
        batch_size (int): Batch size for training
        scale_target (bool): Whether to scale the target variable
        
    Returns:
        Tuple[RegressionDataModule, int, Optional[StandardScaler]]: 
            - DataModule
            - Input size
            - Target scaler (if scale_target=True)
    """
    # Separate features and targets
    features = df.drop(columns=[target_column]).values
    targets = df[target_column].values.reshape(-1, 1)  # Ensure 2D array
    
    # Print target statistics
    print("Target variable statistics:")
    print(f"Mean: {np.mean(targets):.4f}")
    print(f"Std: {np.std(targets):.4f}")
    print(f"Min: {np.min(targets):.4f}")
    print(f"Max: {np.max(targets):.4f}")
    
    # Scale features
    feature_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features)
    
    # Scale target if requested
    target_scaler = None
    if scale_target:
        target_scaler = StandardScaler()
        targets = target_scaler.fit_transform(targets)
    
    # Get input size
    input_size = features.shape[1]
    
    # Create data module
    data_module = RegressionDataModule(
        features=features,
        targets=targets,
        batch_size=batch_size
    )
    
    return data_module, input_size, target_scaler

def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Predicted vs Actual Values'
) -> None:
    """
    Plot regression results including predicted vs actual values and residuals
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Predicted vs Actual plot
    ax1.scatter(y_true, y_pred, alpha=0.5)
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual Values')
    ax1.legend()
    
    # Residuals plot
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate regression metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Tuple containing (MSE, RMSE, MAE, R²)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mae, r2

def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 50
) -> None:
    """
    Plot distribution of actual and predicted values
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        bins: Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_true, bins=bins, alpha=0.5, label='Actual', density=True)
    plt.hist(y_pred, bins=bins, alpha=0.5, label='Predicted', density=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Values')
    plt.legend()
    plt.show()

def evaluate_regression_model(
    model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    split: str = 'test',
    target_scaler: Optional[object] = None
) -> None:
    """
    Evaluate regression model and display various plots and metrics
    
    Args:
        model: Trained PyTorch Lightning model
        data_module: Data module containing the data
        split: Which split to evaluate ('train', 'val', or 'test')
        target_scaler: Scaler used for target variable (if any)
    """
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Get appropriate dataloader
    if split == 'train':
        dataloader = data_module.train_dataloader()
    elif split == 'val':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()
    
    # Initialize lists to store predictions and true values
    y_pred = []
    y_true = []
    
    # Disable gradient computation for prediction
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            # Get predictions
            outputs = model(x)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(y.numpy())
    
    # Convert lists to numpy arrays
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    # Inverse transform if scaler was used
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse, rmse, mae, r2 = calculate_regression_metrics(y_true, y_pred)
    
    # Print metrics
    print(f"\nRegression Metrics ({split} set):")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot results
    plot_regression_results(y_true, y_pred, f'Regression Results ({split} set)')
    plot_prediction_distribution(y_true, y_pred)

def get_trainer(
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    min_delta: float = 1e-4,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> pl.Trainer:
    """
    Configure a PyTorch Lightning trainer with early stopping
    
    Args:
        max_epochs (int): Maximum number of training epochs
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping
        min_delta (float): Minimum change in monitored quantity to qualify as an improvement
        monitor (str): Quantity to monitor ('val_loss', 'val_acc', etc.)
        mode (str): 'min' for loss, 'max' for metrics like accuracy
        
    Returns:
        pl.Trainer: Configured PyTorch Lightning trainer
    """
    # Configure early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        min_delta=min_delta,
        mode=mode,
        verbose=True
    )
    metric_printer = MetricPrinter()
    # Create trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping, metric_printer],
        # Enable GPU if available
        accelerator='auto',
        # Show progress bar
        enable_progress_bar=False,
        # Log metrics every epoch
        log_every_n_steps=1
    )
    
    return trainer

def train_model_with_early_stopping(
    model: pl.LightningModule,
    data_module: BinaryClassificationDataModule,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> pl.LightningModule:
    """
    Train a model with early stopping
    
    Args:
        model (pl.LightningModule): Model to train
        data_module (BinaryClassificationDataModule): Data module for training
        max_epochs (int): Maximum number of training epochs
        early_stopping_patience (int): Number of epochs to wait before early stopping
        monitor (str): Metric to monitor for early stopping
        mode (str): 'min' for loss, 'max' for metrics like accuracy
        
    Returns:
        pl.LightningModule: Trained model
    """
    # Get trainer with early stopping
    trainer = get_trainer(
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        monitor=monitor,
        mode=mode
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    return model

def plot_confusion_matrix(model: pl.LightningModule, 
                        dataloader: DataLoader,
                        class_names: List[str] = ['0', '1']) -> None:
    """
    Generate and plot confusion matrix for model predictions
    
    Args:
        model (pl.LightningModule): Trained PyTorch Lightning model
        dataloader (DataLoader): DataLoader containing the data to evaluate
        class_names (List[str]): Names of the classes for the plot
    """
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Initialize lists to store predictions and true labels
    y_pred = []
    y_true = []
    
    # Disable gradient computation for prediction
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            # Get predictions
            outputs = model(x)
            predictions = (outputs > 0.5).float().cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(y.numpy())
    
    # Convert lists to numpy arrays
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axis
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    plt.show()

# Example usage of confusion matrix plotting
def evaluate_model(model: pl.LightningModule, 
                  data_module: BinaryClassificationDataModule,
                  split: str = 'test') -> None:
    """
    Evaluate model and display confusion matrix
    
    Args:
        model (pl.LightningModule): Trained PyTorch Lightning model
        data_module (BinaryClassificationDataModule): Data module containing the data
        split (str): Which split to evaluate ('train', 'val', or 'test')
    """
    # Setup data module if not already done
    # if not data_module.is_setup:
    #     data_module.setup()
    
    # Get appropriate dataloader
    if split == 'train':
        dataloader = data_module.train_dataloader()
    elif split == 'val':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()
    
    # Plot confusion matrix
    plot_confusion_matrix(model, dataloader)

def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_type: str = 'subsim1',
    input_size: int = None,
    map_location: Optional[str] = None
) -> pl.LightningModule:
    """
    Load a SubSimNetwork model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_type: Type of model to load ('subsim1', 'subsim2', or 'subsim3')
        input_size: Input size for the model (required if not saved in checkpoint)
        map_location: Device to map model to ('cpu', 'cuda', etc.)
        
    Returns:
        Loaded model
    """
    try:
        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        # Create model architecture based on type
        if model_type == 'subsim1':
            model_class = create_subsim1
        elif model_type == 'subsim2':
            model_class = create_subsim2
        elif model_type == 'subsim3':
            model_class = create_subsim3
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Try to load checkpoint
        try:
            # First try loading with input_size from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            if 'hyper_parameters' in checkpoint and 'input_size' in checkpoint['hyper_parameters']:
                loaded_input_size = checkpoint['hyper_parameters']['input_size']
                model = model_class(input_size=loaded_input_size)
            else:
                # If input_size not in checkpoint, use provided input_size
                if input_size is None:
                    raise ValueError("input_size must be provided when not present in checkpoint")
                model = model_class(input_size=input_size)
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])
            
            print(f"Successfully loaded model from {checkpoint_path}")
            return model
            
        except Exception as e:
            raise Exception(f"Error loading checkpoint: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Error in load_model_from_checkpoint: {str(e)}")
    
def dr_test():
    return "new test1"