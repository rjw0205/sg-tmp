import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Computes the Dice Loss, a measure of overlap between two samples.
    
    This implementation supports both hard labels (shape: N, H, W) and soft labels (shape: N, C, H, W).
    Refer to the original paper for more details about focal dice loss: https://arxiv.org/abs/1810.07842

    Args:
        smooth (float, optional): A smoothing constant to prevent division by zero. Defaults to 1.0.
        p (int, optional): Exponent value applied to predictions and targets when computing union. Defaults to 2.
        gamma (float, optional): Focal coefficient to adjust the weight of difficult samples. Defaults to 1.0.
    
    Shape:
        - predictions: (N, C, H, W) where C is the number of classes (soft labels).
        - targets: (N, H, W) for hard labels, or (N, C, H, W) for soft labels.
        
    Returns:
        Loss value depending on the specified reduction mode.
    """
    def __init__(self, smooth=1.0, p=2, gamma=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.gamma = gamma

    def _convert_to_one_hot(self, targets, num_classes):
        """
        Converts the target (hard labels) into one-hot encoding format with the same shape as predictions.

        Args:
            targets (torch.Tensor): Target tensor with shape (N, H, W).
            num_classes (int): Number of classes in the predictions.

        Returns:
            torch.Tensor: One-hot encoded target with shape (C, N, H, W).
        """
        return F.one_hot(targets, num_classes=num_classes).permute(3, 0, 1, 2).float()

    def _get_dice_score(self, predictions, targets):
        """
        Computes the Dice score between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted tensor of shape (C, N, H, W).
            targets (torch.Tensor): Target tensor of shape (C, N, H, W).

        Returns:
            torch.Tensor: Dice score for each channel.
        """
        # Flatten predictions and targets
        predictions = predictions.contiguous().view(predictions.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        # Compute intersection and union
        intersection = torch.sum(predictions * targets, dim=1)
        union = torch.sum(predictions.pow(self.p) + targets.pow(self.p), dim=1)

        # Calculate dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice_score
    
    def _apply_focal_coefficient(self, dice_loss):
        """
        Applies focal coefficient to dice loss if gamma is not equal to 1.0.

        Args:
            dice_loss (torch.Tensor): Dice loss before applying focal coefficient.

        Returns:
            torch.Tensor: Dice loss after applying focal coefficient.
        """
        if torch.isclose(torch.tensor(self.gamma), torch.tensor(1.0)):
            return dice_loss

        # Apply focal coefficient
        return torch.pow(dice_loss, 1.0 / self.gamma)

    def forward(self, predictions, targets):
        """
        Forward pass to compute the Dice loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Model output of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth labels. Can be of shape (N, H, W) for hard labels or (N, C, H, W) for soft labels.

        Returns:
            torch.Tensor: Calculated Dice loss.
        """
        # Convert NCHW to CNHW format
        predictions = predictions.permute(1, 0, 2, 3)

        # Convert hard labels (N, H, W) to one-hot encoding (C, N, H, W)
        if targets.dim() == 3:
            num_classes = predictions.shape[0]
            targets = self._convert_to_one_hot(targets, num_classes)
        elif targets.dim() == 4:
            targets = targets.permute(1, 0, 2, 3)
        else:
            raise ValueError("Target shape not supported. Expected 3D or 4D tensor.")

        # Compute the Dice score
        dice_score = self._get_dice_score(predictions, targets)

        # Compute Dice loss (1 - Dice score)
        dice_loss = 1.0 - dice_score

        # Apply focal coefficient to dice loss
        focal_dice_loss = self._apply_focal_coefficient(dice_loss)

        # Return the mean loss across the channels
        return focal_dice_loss.mean()


if __name__ == "__main__":
    # Define predictions and targets
    predictions = torch.randn(4, 3, 128, 128)
    hard_targets = torch.randint(0, 3, (4, 128, 128))
    soft_targets = torch.stack([hard_targets for _ in range(3)], dim=1) / 3.0

    # Initialize Dice loss
    dice_loss = DiceLoss(smooth=1.0, p=2, gamma=1.0)

    # Compute Dice loss
    loss = dice_loss(predictions, hard_targets)
    print("Dice Loss (Hard Labels):", loss.item())

    loss = dice_loss(predictions, soft_targets)
    print("Dice Loss (Soft Labels):", loss.item())
