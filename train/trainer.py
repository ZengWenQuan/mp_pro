import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from utils.general import plot_loss_curve


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, experiment_dir):
        """
        Trainer class for model training
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            experiment_dir: Directory to save results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        
        # 初始化TensorBoard
        self.tb_log_dir = self.experiment_dir / 'logs'
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = self._get_loss_fn(config.get('loss', 'mse'))
        
        # Optimizer
        self.optimizer = self._get_optimizer(
            config.get('optimizer', 'adam'),
            config.get('learning_rate', 0.001),
            config.get('weight_decay', 0)
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler(
            config.get('scheduler', None),
            config.get('scheduler_params', {})
        )
        
        # Training history
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 如果配置中指定了预训练权重，加载它们
        pretrained_path = config.get('pretrained', None)
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # 如果指定了断点路径，从断点恢复训练
        resume_path = config.get('resume_from', None)
        if resume_path:
            self._resume_from_checkpoint(resume_path)
    
    def _get_loss_fn(self, loss_name):
        """Get loss function"""
        if loss_name.lower() == 'mse':
            return nn.MSELoss()
        elif loss_name.lower() == 'mae':
            return nn.L1Loss()
        elif loss_name.lower() == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name.lower() == 'ce':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def _get_optimizer(self, opt_name, lr, weight_decay):
        """Get optimizer"""
        if opt_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif opt_name.lower() == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
    
    def _get_scheduler(self, scheduler_name, scheduler_params):
        """Get learning rate scheduler"""
        if scheduler_name is None:
            return None
        
        if scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('t_max', 10)
            )
        elif scheduler_name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 10)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _load_pretrained(self, pretrained_path):
        """加载预训练权重"""
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            return
            
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 可选：加载优化器状态
        if 'optimizer_state_dict' in checkpoint and not self.config.get('reset_optimizer', False):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 可选：加载学习率调度器状态
        if self.scheduler and 'scheduler_state_dict' in checkpoint and not self.config.get('reset_scheduler', False):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 可选：加载最佳验证损失
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            
        print(f"Successfully loaded pretrained weights")
    
    def _save_checkpoint(self, epoch, is_best=False, filename=None):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': self.config.get('model_config', {})
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if filename is None:
            filename = f"checkpoint_epoch{epoch+1}.pt"
        
        # 保存checkpoint文件
        checkpoint_path = Path(self.experiment_dir) / 'weights' / filename
        torch.save(state, checkpoint_path)
        
        # 如果是最佳模型，还保存为best.pt
        if is_best:
            best_path = Path(self.experiment_dir) / 'weights' / 'best.pt'
            torch.save(state, best_path)
    
    def _save_best_only(self, epoch):
        """只保存最佳模型，不保存checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': self.config.get('model_config', {})
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 只保存best.pt
        best_path = Path(self.experiment_dir) / 'weights' / 'best.pt'
        torch.save(state, best_path)
    
    def _resume_from_checkpoint(self, checkpoint_path):
        """从断点恢复训练"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return
            
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载学习率调度器状态
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载训练状态
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Resumed training from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]', leave=True, position=0)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # 收集预测和目标值，用于统计
            all_outputs.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            
            # 每隔一定步数记录训练损失到TensorBoard
            if batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        # 合并所有批次的输出，计算预测值统计信息
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        # 记录训练集上的预测值范围
        min_pred = np.min(all_outputs, axis=0)
        max_pred = np.max(all_outputs, axis=0)
        mean_pred = np.mean(all_outputs, axis=0)
        std_pred = np.std(all_outputs, axis=0)
        
        # 记录到TensorBoard
        for i in range(all_outputs.shape[1]):
            self.writer.add_scalar(f'Training/Predictions/Min_Feature{i+1}', min_pred[i], epoch)
            self.writer.add_scalar(f'Training/Predictions/Max_Feature{i+1}', max_pred[i], epoch)
            self.writer.add_scalar(f'Training/Predictions/Mean_Feature{i+1}', mean_pred[i], epoch)
            self.writer.add_scalar(f'Training/Predictions/Std_Feature{i+1}', std_pred[i], epoch)
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        # 收集预测和真实值用于TensorBoard可视化
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            # 创建进度条
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Valid]', leave=True, position=1)
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                
                # 收集数据
                all_outputs.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # 将所有预测和真实值合并
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 记录验证集上的预测值范围
        min_pred = np.min(all_outputs, axis=0)
        max_pred = np.max(all_outputs, axis=0)
        mean_pred = np.mean(all_outputs, axis=0)
        std_pred = np.std(all_outputs, axis=0)
        
        # 记录到TensorBoard
        for i in range(all_outputs.shape[1]):
            self.writer.add_scalar(f'Validation/Predictions/Min_Feature{i+1}', min_pred[i], epoch)
            self.writer.add_scalar(f'Validation/Predictions/Max_Feature{i+1}', max_pred[i], epoch)
            self.writer.add_scalar(f'Validation/Predictions/Mean_Feature{i+1}', mean_pred[i], epoch)
            self.writer.add_scalar(f'Validation/Predictions/Std_Feature{i+1}', std_pred[i], epoch)
            
            # 如果是多特征预测，为每个特征记录MSE
            mse = np.mean((all_outputs[:, i] - all_targets[:, i]) ** 2)
            self.writer.add_scalar(f'Validation/MSE_Feature{i+1}', mse, epoch)
        
        return avg_loss
    
    def train(self):
        """训练模型"""
        best_val_loss = float('inf')
        best_model_path = None
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 检查是否是最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                # 保存最佳模型
                self._save_checkpoint(epoch, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 每10轮保存一次常规检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 输出当前epoch的详细信息
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
            
            # 绘制损失曲线
            plot_loss_curve(
                self.train_losses, 
                self.val_losses, 
                save_path=str(self.experiment_dir / 'plots' / 'loss_curve.png')
            )
            
            # 早停检查
            if patience_counter >= self.config.get('patience', 10):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # 训练完成后返回最佳验证损失
        return best_val_loss 