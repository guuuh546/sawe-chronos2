import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np


def compute_mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """计算均方误差"""
    return ((preds - targets) ** 2).mean().item()


def compute_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """计算平均绝对误差"""
    return torch.abs(preds - targets).mean().item()


class Trainer:
    """
    训练器 - 支持频谱稀疏正则化和门控权重收集
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        lambda_l1: float = 0.001,
        use_spectral_reg: bool = True,
        collect_gating_weights: bool = False,
        loss_fn: Optional[nn.Module] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.lambda_l1 = lambda_l1
        self.use_spectral_reg = use_spectral_reg
        self.collect_gating_weights = collect_gating_weights
        self.mse_loss = nn.MSELoss()
        # 可选自定义主损失（如频带加权 MSE）；默认使用标准 MSE
        self.loss_fn = loss_fn
        # 收集门控权重和频谱特征
        self.collected_gammas = []
        self.collected_spectral_feats = []
    
    def _collect_gating_info(self):
        """收集模型中的门控信息"""
        from src.freq_lora import FrequencyAwareLoRA
        
        for name, module in self.model.named_modules():
            # 检查是否是 FrequencyAwareLoRA，或者是否有 lora 属性（LoRAInjectedLinear）
            if isinstance(module, FrequencyAwareLoRA):
                if module.current_gamma is not None:
                    self.collected_gammas.append(module.current_gamma.detach().cpu().numpy())
                if module.current_spectral_feats is not None:
                    self.collected_spectral_feats.append(module.current_spectral_feats.detach().cpu().numpy())
            elif hasattr(module, 'lora') and isinstance(module.lora, FrequencyAwareLoRA):
                # 处理 LoRAInjectedLinear 包裹的情况
                lora_module = module.lora
                if lora_module.current_gamma is not None:
                    self.collected_gammas.append(lora_module.current_gamma.detach().cpu().numpy())
                if lora_module.current_spectral_feats is not None:
                    self.collected_spectral_feats.append(lora_module.current_spectral_feats.detach().cpu().numpy())
    
    def get_collected_data(self) -> Dict[str, np.ndarray]:
        """获取收集到的数据"""
        if len(self.collected_gammas) == 0:
            return {"gammas": np.array([]), "spectral_feats": np.array([])}
        
        gammas = np.vstack(self.collected_gammas)
        spectral_feats = np.vstack(self.collected_spectral_feats) if self.collected_spectral_feats else np.array([])
        
        return {
            "gammas": gammas,
            "spectral_feats": spectral_feats
        }
    
    def clear_collected_data(self):
        """清空收集到的数据"""
        self.collected_gammas = []
        self.collected_spectral_feats = []
    
    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            preds = self.model(x)
            
            # 收集门控权重
            if self.collect_gating_weights:
                self._collect_gating_info()
            
            # 计算主损失（可传入自定义损失，如频带加权 MSE）
            main_loss = (self.loss_fn(preds, y) if self.loss_fn is not None
                         else self.mse_loss(preds, y))
            
            # 计算频谱稀疏正则化项
            reg_loss = torch.tensor(0.0, device=self.device)
            if self.use_spectral_reg:
                from src.model_injection import compute_total_spectral_reg
                reg_loss = compute_total_spectral_reg(self.model)
            
            # 总损失
            total_batch_loss = main_loss + reg_loss
            
            # 反向传播
            total_batch_loss.backward()
            self.optimizer.step()
            
            # 记录指标
            total_loss += total_batch_loss.item()
            total_mse += compute_mse(preds, y)
            total_mae += compute_mae(preds, y)
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "mse": total_mse / num_batches,
            "mae": total_mae / num_batches
        }
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                preds = self.model(x)
                
                total_mse += compute_mse(preds, y)
                total_mae += compute_mae(preds, y)
                num_batches += 1
        
        return {
            "mse": total_mse / num_batches,
            "mae": total_mae / num_batches
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        val_interval: int = 1,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """完整训练流程"""
        best_val_mse = float('inf')
        patience_counter = 0
        train_history = []
        val_history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_one_epoch(train_loader)
            train_history.append(train_metrics)
            
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            
            # 验证
            if (epoch + 1) % val_interval == 0:
                val_metrics = self.evaluate(val_loader)
                val_history.append(val_metrics)
                
                print(f"  Val   - MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}")
                
                # 早停逻辑
                if val_metrics['mse'] < best_val_mse:
                    best_val_mse = val_metrics['mse']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
        
        return {
            "train_history": train_history,
            "val_history": val_history,
            "best_val_mse": best_val_mse
        }
