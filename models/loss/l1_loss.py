import torch
import torch.nn as nn

class L1Loss(nn.Module):
    """
    L1 Loss (Mean Absolute Error)
    
    Calculates the mean absolute error between each element in the input and target.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
    
    def forward(self, input, target):
        """
        Args:
            input (Tensor): The predicted values
            target (Tensor): The target values
            
        Returns:
            Tensor: The L1 loss
        """
        return self.loss_fn(input, target) 