import torch
import torch.nn.functional as F

def compress_average_pooling(input_tensor, pool_size):
    """
    Applies a simple average pooling to the input tensor to reduce its spatial dimensions of the tokens.
    Following the paper : https://arxiv.org/abs/2505.19812

    Args:
        input_tensor (torch.Tensor): The input tensor to be pooled. Shape (batch_size, seq_len, feature_dim).
        pool_size (int): The size of the pooling window.

    Returns:
        torch.Tensor: The pooled tensor with reduced spatial dimensions.
    """
    return F.avg_pool1d(
        input_tensor.transpose(1, 2),  # Change to (batch_size, feature_dim, seq_len)
        kernel_size=pool_size,
        stride=pool_size
    ).transpose(1, 2)  # Change back to (batch_size, seq_len, feature_dim)
    
class Learnable_pooling(torch.nn.Module):
    """
    A learnable pooling layer that reduces the spatial dimensions of the input tensor.
    Following the paper : https://arxiv.org/abs/2505.19812
    """
    def __init__(self, pooling_size, sorted=False):
        """
        The intuition behind sorted is that it will not be position-dependant, but only value-dependant.
        Idea being to capture how to aggregate values based on their relative importance rather than their position.
        It also allows it to learn the max pooling if it is ultimately the best way to aggregate values.
        """
        super(Learnable_pooling, self).__init__()
        self.pooling = torch.nn.Linear(pooling_size, 1, bias=False)
        # Initialize the weights to be the average
        torch.nn.init.constant_(self.pooling.weight, 1.0 / pooling_size)

        self.pooling_size = pooling_size
        self.sorted = sorted

    def forward(self, x):
        """
        Apply the pooling operation to the input tensor with a stride of pooling_size.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, feature_dim).
        Returns:
            torch.Tensor: The pooled tensor with reduced spatial dimensions.
        batch_size, seq_len, feature_dim = x.size()
        """
        if self.sorted:
            x, _ = torch.sort(x, dim=1)
        batch_size, seq_len, feature_dim = x.size()
        # Pad the input tensor if seq_len is not divisible by pooling_size
        if seq_len % self.pooling_size != 0:
            pad_size = self.pooling_size - (seq_len % self.pooling_size)
            x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
            seq_len += pad_size

        # Reshape to (batch_size, num_pools, pooling_size, feature_dim)
        x = x.view(batch_size, seq_len // self.pooling_size, self.pooling_size, feature_dim)

        # Apply the learnable pooling
        x = self.pooling(x).squeeze(2)  # Shape: (batch_size, num_pools, feature_dim)

        return x