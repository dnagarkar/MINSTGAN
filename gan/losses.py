import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn.functional import mse_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    ones_real = torch.ones_like(logits_real)
    zeros_fake = torch.zeros_like(logits_fake)


    D_loss_real = bce_loss(logits_real, ones_real)
    D_loss_fake = bce_loss(logits_fake, zeros_fake)

    loss = D_loss_real + D_loss_fake

    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    ones = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, ones)


    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    ones_real = torch.ones_like(scores_real)
    zeros_fake = torch.zeros_like(scores_fake)


    D_loss_real = torch.mean(mse_loss(scores_real, ones_real))
    D_loss_fake = torch.mean(mse_loss(scores_fake, zeros_fake))

    loss = D_loss_real + D_loss_fake

    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    ones = torch.ones_like(scores_fake)

    loss = torch.mean(mse_loss(scores_fake, ones))

    return loss
