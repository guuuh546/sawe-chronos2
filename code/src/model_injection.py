import torch
import torch.nn as nn
from typing import Optional, List
from src.freq_lora import FrequencyAwareLoRA


class LoRAInjectedLinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        num_freq_bins: int = 64,
        lambda_l1: float = 0.0001,
        use_gating: bool = True,
        use_spectral_reg: bool = True,
        use_entropy: bool = False,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        self.lora = FrequencyAwareLoRA(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank, alpha=alpha,
            num_freq_bins=num_freq_bins,
            lambda_l1=lambda_l1,
            use_gating=use_gating,
            use_spectral_reg=use_spectral_reg,
            use_entropy=use_entropy,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_linear(x)
        lora_out = self.lora(x)
        return original_out + lora_out
    
    def compute_spectral_regularization(self) -> torch.Tensor:
        return self.lora.compute_spectral_regularization()


def inject_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    num_freq_bins: int = 64,
    lambda_l1: float = 0.0001,
    use_gating: bool = True,
    use_spectral_reg: bool = True,
    use_entropy: bool = False,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "o_proj"]
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name.lower() for target in target_modules):
                print(f"Injecting LoRA to: {name}")
                
                parent_name = ".".join(name.split(".")[:-1])
                module_name = name.split(".")[-1]
                
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                
                injected_linear = LoRAInjectedLinear(
                    module, rank=rank, alpha=alpha,
                    num_freq_bins=num_freq_bins,
                    lambda_l1=lambda_l1,
                    use_gating=use_gating,
                    use_spectral_reg=use_spectral_reg,
                    use_entropy=use_entropy,
                )
                
                setattr(parent, module_name, injected_linear)
    
    return model


def compute_total_spectral_reg(model: nn.Module) -> torch.Tensor:
    """
    计算整个模型的总频谱稀疏正则化项
    """
    total_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for module in model.modules():
        if hasattr(module, 'compute_spectral_regularization'):
            total_reg += module.compute_spectral_regularization()
    
    return total_reg
