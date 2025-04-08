import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
import os
import subprocess
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from utils.general import plot_loss_curve
from models.loss import L1Loss


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
        
        # 创建模型信息目录
        self.model_info_dir = self.experiment_dir / 'model_info'
        os.makedirs(self.model_info_dir, exist_ok=True)
        
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
        
        # 保存模型结构信息
        self._save_model_info()
        
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
            return L1Loss()
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
    
    def _save_best_and_latest(self, epoch, is_best=False):
        """只保存最佳模型和最新模型"""
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
        
        # 保存最新模型为latest.pt
        latest_path = Path(self.experiment_dir) / 'weights' / 'latest.pt'
        torch.save(state, latest_path)
        
        # 如果是最佳模型，还保存为best.pt
        if is_best:
            best_path = Path(self.experiment_dir) / 'weights' / 'best.pt'
            print(f"Saving best model to {best_path}")
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
    
    def evaluate_best_model(self):
        """运行评估脚本来评估最优模型"""
        print("\n=== 开始评估最优模型 ===")
        
        # 构建最优模型路径
        best_model_path = Path(self.experiment_dir) / 'weights' / 'best.pt'
        if not os.path.exists(best_model_path):
            print(f"最优模型文件不存在: {best_model_path}")
            return
        
        # 构建评估结果保存路径
        output_dir = self.experiment_dir / 'evaluation'
        
        # 构建评估命令
        cmd = [
            "python", "run_evaluate.py",
            "--model-path", str(best_model_path),
            "--output-dir", str(output_dir)
        ]
        
        # 如果配置中有config_path，使用它；否则尝试根据实验目录推断配置文件
        if "config_path" in self.config:
            config_path = self.config["config_path"]
            cmd.extend(["--config", config_path])
            print(f"使用配置文件: {config_path}")
        else:
            # 没有配置路径时，尝试从实验目录名推断模型类型，并使用对应配置
            exp_dir_name = self.experiment_dir.name
            model_type = None
            
            # 尝试从实验目录名推断模型类型
            for model_type_name in ['mlp', 'lstm', 'conv1d', 'transformer']:
                if model_type_name in exp_dir_name.lower():
                    model_type = model_type_name
                    break
            
            if model_type:
                inferred_config = f"configs/{model_type}.yaml"
                if os.path.exists(inferred_config):
                    cmd.extend(["--config", inferred_config])
                    print(f"推断使用配置文件: {inferred_config}")
                else:
                    print(f"警告: 推断的配置文件 {inferred_config} 不存在")
            else:
                print("警告: 无法从实验目录名推断模型类型，评估可能失败")
        
        # 打印命令
        print("执行评估命令:")
        print(" ".join(cmd))
        
        # 运行评估脚本
        subprocess.call(cmd)
        
        print(f"评估完成，结果保存在: {output_dir}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]', leave=True, position=0)
        
        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # 计算梯度范数
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # 梯度裁剪（如果配置中指定）
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # 收集预测和目标值，用于统计
            all_outputs.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            
            # 每隔一定步数记录训练损失和梯度范数到TensorBoard
            if batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Training/GradientNorm', total_norm, global_step)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'lr': f'{current_lr:.2e}',
                'grad_norm': f'{total_norm:.2f}'
            })
        
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
        
        return avg_loss, all_outputs, all_targets
    
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
        
        return avg_loss, all_outputs, all_targets
    
    def train(self):
        """训练模型"""
        # 初始化训练状态
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_path = None
        
        # 创建进度条
        pbar = tqdm(range(self.start_epoch, self.config['epochs']), 
                    desc='Training', 
                    leave=True, 
                    position=0)
        
        for epoch in pbar:
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            
            # 训练一个epoch
            train_loss, train_outputs, train_targets = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_outputs, val_targets = self.validate(epoch)
            
            # 检查是否是最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 保存最佳模型和最新模型
            self._save_best_and_latest(epoch, is_best=is_best)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
                # 记录更新后的学习率
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    self.writer.add_scalar('Training/LearningRate', new_lr, epoch + 0.5)
                    print(f"\nLearning rate changed from {current_lr:.2e} to {new_lr:.2e}")
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            # 输出当前epoch的详细信息
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 绘制损失曲线
            plot_loss_curve(
                self.train_losses, 
                self.val_losses, 
                save_path=str(self.experiment_dir / 'plots' / 'loss_curve.png')
            )

            # 更新进度条
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val_loss': f'{best_val_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # 早停检查
            early_stopping = self.config.get('early_stopping', 10)
            if early_stopping > 0 and patience_counter >= early_stopping:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # 训练结束，加载最佳模型
        if best_model_path:
            self._load_checkpoint(best_model_path)
            print(f"\nLoaded best model from epoch {best_epoch + 1}")
        
        # 训练结束后，自动评估最优模型
        self.evaluate_best_model()
        
        return best_val_loss

    def _save_model_info(self):
        """保存模型结构信息到文本文件"""
        info_path = self.model_info_dir / 'model_structure.txt'
        
        with open(info_path, 'w', encoding='utf-8') as f:
            # 写入模型名称和时间
            f.write(f"模型结构信息\n")
            f.write(f"{'='*50}\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入模型类型
            model_type = self.model.__class__.__name__
            f.write(f"模型类型: {model_type}\n\n")
            
            # 写入模型配置
            if 'model_config' in self.config:
                f.write("模型配置:\n")
                for key, value in self.config['model_config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 写入模型参数统计
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            f.write(f"参数统计:\n")
            f.write(f"  总参数数量: {total_params:,}\n")
            f.write(f"  可训练参数: {trainable_params:,}\n")
            f.write(f"  不可训练参数: {total_params - trainable_params:,}\n\n")
            
            # 写入模型结构
            f.write("模型结构:\n")
            f.write(f"{self.model}\n\n")
            
            # 写入每层参数数量
            f.write("每层参数数量:\n")
            for name, param in self.model.named_parameters():
                f.write(f"  {name}: {param.numel():,} 参数\n")
            
            # 写入优化器信息
            f.write("\n优化器信息:\n")
            f.write(f"  类型: {self.optimizer.__class__.__name__}\n")
            f.write(f"  学习率: {self.config.get('learning_rate', 0.001)}\n")
            f.write(f"  权重衰减: {self.config.get('weight_decay', 0)}\n")
            
            # 写入损失函数信息
            f.write("\n损失函数:\n")
            f.write(f"  类型: {self.criterion.__class__.__name__}\n")
            
            # 写入学习率调度器信息
            if self.scheduler:
                f.write("\n学习率调度器:\n")
                f.write(f"  类型: {self.scheduler.__class__.__name__}\n")
                if 'scheduler_params' in self.config:
                    for key, value in self.config.get('scheduler_params', {}).items():
                        f.write(f"  {key}: {value}\n")
            
            # 写入训练配置
            f.write("\n训练配置:\n")
            f.write(f"  批次大小: {self.config.get('batch_size', 'N/A')}\n")
            f.write(f"  Epochs: {self.config.get('epochs', 'N/A')}\n")
            early_stopping = self.config.get('early_stopping', 'N/A')
            if early_stopping == -1:
                early_stopping = "禁用"
            f.write(f"  早停机制: {early_stopping}\n")
            
            # 写入设备信息
            f.write("\n训练设备:\n")
            f.write(f"  {self.device}\n")
        
        print(f"模型结构信息已保存到: {info_path}") 