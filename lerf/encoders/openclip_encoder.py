from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from lerf.encoders.image_encoder import (BaseImageEncoder,
                                         BaseImageEncoderConfig)

from nerfstudio.viewer.viewer_elements import *
# Import for own methods
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
#from nerfstudio.utils.eval_utils import eval_setup

#from nerfstudio.scripts.exporter import ExportPointCloud
from pathlib import Path

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, Pipeline
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from lerf.data.lerf_datamanager import (
    LERFDataManager,
    LERFDataManagerConfig
)

@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")

class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config

        # Define image preprocessing transformations
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        # Load CLIP model
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        # Viewer UI elements
        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)
        
        # Initialize positive and negative embeddings
        self.positives = self.positive_input.value.split(";")
        self.negatives = self.config.negatives

        # Compute text embeddings for positive and negative phrases
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)

        # Normalize embeddings
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        # Ensure consistency of embedding dimensions
        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        """Returns a name identifier for the CLIP model."""
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        """Returns the number of embedding dimensions."""
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        """Callback function for GUI input."""
        # Split input words with comma and input to set_positives
        self.set_positives(element.value.split(";"))

#        
#    def export_object(self, element):
#        """Callback function to export the point cloud for the positive relevancy map using generate_point_cloud."""
#        from nerfstudio.exporter.exporter_utils import generate_point_cloud
#
#        # Define parameters (these can be adjusted as needed for your scenario)
#        num_points = 1000000  # Number of points to generate
#        remove_outliers = True
#        reorient_normals = False
#        normal_method = "open3d"
#        rgb_output_name = "rgb"
#        depth_output_name = "depth"
#        normal_output_name = "normals"
#        
#        # Define the path to the configuration file
#        config_path = Path('scene04/outputs/output/lerf/2025-01-09_134140/config.yml')
#        
#        # Define the output directory for the exported point cloud
#        output_dir = Path('exports/pcd/')
#        
#        # Ensure the output directory exists
#        if not output_dir.exists():
#            output_dir.mkdir(parents=True)
#
#        # Set up the pipeline and other necessary components
#        #_, pipeline, _, _ = eval_setup(config_path)
#        
#        # Whether the normals should be estimated based on the point cloud
#        estimate_normals = normal_method == "open3d"
#
#        # Generate the point cloud
#        pcd = generate_point_cloud(
#            pipeline=pipeline,
#            num_points=num_points,
#            remove_outliers=remove_outliers,
#            reorient_normals=reorient_normals,
#            estimate_normals=estimate_normals,
#            rgb_output_name=rgb_output_name,
#            depth_output_name=depth_output_name,
#            normal_output_name=normal_output_name if normal_method == "model_output" else None,
#            crop_obb=None,  # Oriented Bounding Box (if applicable)
#            std_ratio=10.0,  # Threshold for outlier removal
#        )
#
#        # Save the point cloud to a file
#        #output_file = output_dir / 'point_cloud.ply'
        #o3d.io.write_point_cloud(str(output_file), pcd)

        #print(f"Point cloud successfully exported to {output_file}")

        #except Exception as e:
        #print(f"Error exporting point cloud: {e}")

    def set_positives(self, text_list):
        """Updates the list of positive phrases and re-computes their embeddings.
        
        Args:
            text_list (list of str): A list of positive phrases to be embedded.

        Returns:
            Abstract matrix as an output (matrix of puppy is more similar to dog than to table)
        """
        # Store the input positive phrases
        self.positives = text_list
        
        with torch.no_grad():   # Disable gradient computation for efficiency
             # Tokenize and encode each phrase in the positive list and move to cuda
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")

            # Pass tokenized phrases through the model to get text embeddings
            self.pos_embeds = self.model.encode_text(tok_phrases)

        # Normalize embeddings to unit vectors
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        """Computes the relevancy score of a given embedding against positive and negative embeddings.
        
        Args:
            embed (torch.Tensor): The input embedding tensor representing the features of interest.
            positive_id (int): The index of the positive phrase to compare against.
        
        Returns:
            torch.Tensor: Softmax-normalized relevancy scores.
            tensor([[0.4470, 0.5530],   -->  [probability that the positive phrase,  probability that the negative phrase is more relevant.]
                    [0.4419, 0.5581], 
                    ...])
        """
        # Combine positive and negative embeddings for comparison
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # Convert to match the dtype of input embeddings # phrases x 512

        # Compute cosine similarity between input embeddings and phrase embeddings
        output = torch.mm(embed, p.T)  # Matrix multiplication: [rays x phrases]

        # Extract similarity values for the selected positive phrase
        positive_vals = output[..., positive_id : positive_id + 1]  # Shape: [rays x 1]

        # Extract similarity values for all negative phrases
        negative_vals = output[..., len(self.positives) :]  # Shape: [rays x N_negatives]

        # Repeat positive scores to match the shape of negative scores
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        # Stack positive and negative similarity scores for softmax computation
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2

        # Apply softmax to obtain normalized relevancy scores
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2

        # Identify the least relevant negative phrase for each ray
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2

        relevancy = (torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
        )

        # Select the corresponding relevancy scores
        return relevancy

    def encode_image(self, input):
        """Encodes an image into CLIP's embedding space"""
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
       


