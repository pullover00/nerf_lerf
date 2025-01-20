import torch
from torch import nn, Tensor
from jaxtyping import Float


class CLIPRenderer(nn.Module):
    """Calculate CLIP embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: Float[Tensor, "bs num_samples num_classes"],  # Input embeddings along rays (batch size, number of samples, embedding dimensions).
        weights: Float[Tensor, "bs num_samples 1"], # Weights for each sampled point along the ray (batch size, number of samples, 1 weight per sample).
    ) -> Float[Tensor, "bs num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        # Compute the weighted sum of embeddings along the sampled points for each ray.
        output = output / torch.linalg.norm(output, dim=-1, keepdim=True)
        return output


class MeanRenderer(nn.Module):
    """Calculate average of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: Float[Tensor, "bs num_samples num_classes"],
        weights: Float[Tensor, "bs num_samples 1"],
    ) -> Float[Tensor, "bs num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        return output
