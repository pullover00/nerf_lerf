from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision

try:
    import clip
except ImportError:
    assert False, "clip is not installed, install it with `pip install clip`"

from lerf.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig

# Define a configuration dataclass for the CLIP network
@dataclass
class CLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: CLIPNetwork)
    clip_model_type: str = "ViT-B/16"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class CLIPNetwork(BaseImageEncoder):
    def __init__(self, config: CLIPNetworkConfig):
        super().__init__()
        self.config = config
        # Define a preprocessing pipeline for input images
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)), # Resize images to 224x224
                torchvision.transforms.Normalize( # Normalize image pixel values
                    mean=[0.48145466, 0.4578275, 0.40821073], # Mean RGB values
                    std=[0.26862954, 0.26130258, 0.27577711], # Standard deviation RGB values
                ),
            ]
        )

        # Load CLIP model
        model, _ = clip.load(self.config.clip_model_type)
        model.eval()
        self.tokenizer = clip.tokenize # Get the tokenizer from CLIP
        self.model = model.to("cuda") # Move the model to GPU
        self.clip_n_dims = self.config.clip_n_dims # Store the embedding dimensionality

         # Initialize positive and negative phrases
        self.positives = ["hand sanitizer"] # Default positive phrases
        self.negatives = self.config.negatives # Negative phrases from configuration

        #  Compute embeddings for positive and negative phrases
        with torch.no_grad():
            # Tokenize and encode positive phrases
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)

            # Tokenize and encode negative phrases
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)

        # Normalize embeddings to unit length
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        # Ensure embedding dimensionalities are consistent
        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        """Returns the name of the model with its type."""
        return "clip_openai_{}".format(self.config.clip_model_type)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the embeddings."""
        return self.config.clip_n_dims

    def set_positives(self, text_list):
        """Updates the list of positive phrases and recomputes embeddings."""
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        """
        Computes relevancy scores for an input embedding against positive and negative embeddings.

        Args:
            embed: The input embedding tensor.
            positive_id: Index of the positive embedding to evaluate.

        Returns:
            A tensor of relevancy scores.
        """
        # Concatenate positive and negative embeddings
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)  
        p = phrases_embeds.to(embed.dtype)  # phrases x 512   # Match data types
        output = torch.mm(embed, p.T)  # rays x phrases # Compute similarity scores
        positive_vals = output[..., positive_id : positive_id + 1]  # Scores for the positive embedding rays x 1
        negative_vals = output[..., len(self.positives) :]  # Scores for the negative embeddings: rays x N_phrase

        # Repeat positive scores to match the number of negative scores
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        # Compute softmax scores for positive and negative relevancy
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # Group scores: rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # Apply softmax: rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # Identify the least relevant negative phrase: rays x 2

        # Gather and return relevancy scores
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input):
        """
        Encodes an input image into a CLIP embedding.

        Args:
            input: The input image tensor.

        Returns:
            A tensor representing the image embedding.
        """
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
