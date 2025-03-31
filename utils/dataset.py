import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class Normalizer:
    """用于标准化/归一化数据的类"""
    
    def __init__(self, method='standard'):
        """
        初始化归一化器
        
        Args:
            method: 归一化方法，可选 'standard'(标准化), 'minmax'(最小最大归一化)
        """
        self.method = method
        self.params = {}
        self.is_fitted = False
    
    def fit(self, data):
        """
        计算归一化参数
        
        Args:
            data: 输入数据，numpy数组
        """
        if self.method == 'standard':
            # 计算均值和标准差
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # 防止除零错误，标准差为0的特征不处理
            std = np.where(std == 0, 1.0, std)
            
            self.params = {
                'mean': mean,
                'std': std
            }
        
        elif self.method == 'minmax':
            # 计算最小值和最大值
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            # 防止除零错误，最大值等于最小值的特征设置范围为1
            range_vals = np.where(max_vals == min_vals, 1.0, max_vals - min_vals)
            
            self.params = {
                'min': min_vals,
                'max': max_vals,
                'range': range_vals
            }
        
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, data):
        """
        应用归一化
        
        Args:
            data: 输入数据，numpy数组
        
        Returns:
            归一化后的数据
        """
        if not self.is_fitted:
            raise ValueError("归一化器未拟合，请先调用fit方法")
        
        data = np.array(data)  # 确保是numpy数组
        
        if self.method == 'standard':
            return (data - self.params['mean']) / self.params['std']
        
        elif self.method == 'minmax':
            return (data - self.params['min']) / self.params['range']
    
    def inverse_transform(self, data):
        """
        反向转换，将归一化数据转换回原始尺度
        
        Args:
            data: 归一化后的数据，numpy数组
        
        Returns:
            原始尺度的数据
        """
        if not self.is_fitted:
            raise ValueError("归一化器未拟合，请先调用fit方法")
        
        data = np.array(data)  # 确保是numpy数组
        
        if self.method == 'standard':
            return data * self.params['std'] + self.params['mean']
        
        elif self.method == 'minmax':
            return data * self.params['range'] + self.params['min']
    
    def fit_transform(self, data):
        """
        拟合并应用归一化
        
        Args:
            data: 输入数据，numpy数组
        
        Returns:
            归一化后的数据
        """
        self.fit(data)
        return self.transform(data)
    
    def save(self, path):
        """
        保存归一化参数到文件
        
        Args:
            path: 保存路径
        """
        save_data = {
            'method': self.method,
            'params': {k: v.tolist() for k, v in self.params.items()},
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f)
    
    def load(self, path):
        """
        从文件加载归一化参数
        
        Args:
            path: 加载路径
        """
        with open(path, 'r') as f:
            load_data = json.load(f)
        
        self.method = load_data['method']
        self.params = {k: np.array(v) for k, v in load_data['params'].items()}
        self.is_fitted = load_data['is_fitted']
        
        return self


class PredictionDataset(Dataset):
    """Dataset class for prediction tasks"""
    
    def __init__(self, data, labels=None, transform=None):
        """
        Initialize dataset
        
        Args:
            data: Input data (numpy array or list)
            labels: Target labels (numpy array or list)
            transform: Optional transform to apply to the data
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        else:
            return x

def create_dataloaders(data, labels=None, batch_size=32, shuffle=True, val_split=0.2, 
                       test_split=0.1, num_workers=4, transforms=None):
    """
    Create train/val/test dataloaders
    
    Args:
        data: Input data
        labels: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of dataloader workers
        transforms: Dictionary of transforms for train/val/test sets
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if transforms is None:
        transforms = {'train': None, 'val': None, 'test': None}
    
    # Convert to numpy arrays if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if labels is not None and not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Split into train/val/test
    data_size = len(data)
    indices = np.arange(data_size)
    
    if shuffle:
        np.random.shuffle(indices)
    
    test_size = int(data_size * test_split)
    val_size = int(data_size * val_split)
    train_size = data_size - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create datasets
    train_data = data[train_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]
    
    if labels is not None:
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        test_labels = labels[test_indices]
        
        train_dataset = PredictionDataset(train_data, train_labels, transforms['train'])
        val_dataset = PredictionDataset(val_data, val_labels, transforms['val'])
        test_dataset = PredictionDataset(test_data, test_labels, transforms['test'])
    else:
        train_dataset = PredictionDataset(train_data, transform=transforms['train'])
        val_dataset = PredictionDataset(val_data, transform=transforms['val'])
        test_dataset = PredictionDataset(test_data, transform=transforms['test'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader 