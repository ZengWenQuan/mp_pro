import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for regression tasks
    
    This implementation adapts the focal loss concept from classification to regression
    by focusing more on hard-to-predict samples and less on easy-to-predict ones.
    
    Args:
        alpha (float): Focusing parameter that reduces the relative loss for well-predicted samples
        beta (float): Weight parameter to balance the loss
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, alpha=2.0, beta=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input (Tensor): Predicted values
            target (Tensor): Target values
            
        Returns:
            Tensor: The focal loss
        """
        # Calculate absolute error
        abs_error = torch.abs(input - target)
        
        # Calculate the focal weight
        # For small errors (easy samples), the weight will be small
        # For large errors (hard samples), the weight will be large
        focal_weight = (1.0 - torch.exp(-abs_error)) ** self.alpha
        
        # Apply beta parameter to balance the loss
        loss = focal_weight * abs_error * self.beta
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 