from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from jaxtyping import Float
import sys

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
#from torchtyping import TensorType

try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


class LERFField(Field):
    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        clip_n_dims: int,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ):
        super().__init__()
        # Grid configuration
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(grid_layers)
        # Spatial distortion to normalize/ warp 3D positions into a compact space.
        self.spatial_distortion = spatial_distortion

        # Create hash grdi encodings for each grid layer
        self.clip_encs = torch.nn.ModuleList(
            [
                LERFField._get_encoding(
                    grid_resolutions[i][0], 
                    grid_resolutions[i][1], 
                    grid_layers[i], 
                    indim=3, 
                    hash_size=grid_sizes[i] 
                )
                for i in range(len(grid_layers))
            ]
        )
        
        # Calculate total output dimensions from all hash grid encodings.
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])

        # Define NN for processing CLIP embeddings
        self.clip_net = tcnn.Network(
            n_input_dims=tot_out_dims + 1,  # Total hash grid features for scale input
            n_output_dims=clip_n_dims,      # Dimensionality of CLIP embedding output
            network_config={             
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )

        # Define NN
        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

    @staticmethod
    # Create a hashgrid encoder
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        # Compute growth factor for resolution scaling across levels.
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))

        #  Create a TinyCUDA Neural Network (tinycudann) HashGrid encoding.
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    # This method computes outputs for a batch of ray samples, producing CLIP and DINO embeddings.
    def get_outputs(self, ray_samples: RaySamples, clip_scales) -> Dict[LERFFieldHeadNames, Float[Tensor, "bs dim"]]:
        # random scales, one scale
        outputs = {}

        # Retrieve positions of ray samples and apply spatial distortion.
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)

        # Normalize positions to fit within the range [0, 1].
        positions = (positions + 2.0) / 4.0

        # Apply hash grid encodings for each level, and concatenate results.
        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)
        hash = x.view(*ray_samples.frustums.shape, -1) # Added by myself

        # Store hash grid features in the outputs dictionary
        outputs[LERFFieldHeadNames.HASHGRID] = x.view(*ray_samples.frustums.shape, -1)

        # Compute CLIP embeddings with hash grid features and scales
        clip_pass = self.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1)).view(*ray_samples.frustums.shape, -1)
        
        # Normalize CLIP embeddings 
        outputs[LERFFieldHeadNames.CLIP] = clip_pass / clip_pass.norm(dim=-1, keepdim=True)
        #print(outputs[LERFFieldHeadNames.CLIP])

        # Compute Dino embeddings
        dino_pass = self.dino_net(x).view(*ray_samples.frustums.shape, -1)
        outputs[LERFFieldHeadNames.DINO] = dino_pass

        return outputs#, hash # Return CLIP-DINO Outputs # return hash added by myself
    
    # Method added by myself
#    def get_mlp(self, hash: TensorType, instance_scales: TensorType) -> TensorType:
#        """
#        Get the GARField affinity field outputs. Note that this is scale-conditioned.
#        This function *does* assume that the hash values are normalized.
#        The MLP output is normalized to unit length.
#        """
#        assert self.quantile_transformer is not None
#
#        # Check that # of rays is the same as # of scales
#        assert hash.shape[0] == instance_scales.shape[0]
#
#        epsilon = 1e-5
#        if self.use_single_scale:
#            instance_pass = self.instance_net(hash)
#            return instance_pass / (instance_pass.norm(dim=-1, keepdim=True) + epsilon)
#
#        scales = instance_scales.contiguous().view(-1, 1)
#
#        # Normalize scales before passing to MLP
#        scales = self.quantile_transformer(scales)
#        instance_pass = self.instance_net(torch.cat([hash, scales], dim=-1))
#
#        norms = instance_pass.norm(dim=-1, keepdim=True)
#        return instance_pass / (norms + epsilon)

    # Method computes CLIP outputs for a given hashgrid encoding and specific scale.
    def get_output_from_hashgrid(self, ray_samples: RaySamples, hashgrid_field, scale):

        # Reshape hashgrid features to prepare for neural network input.
        hashgrid_field = hashgrid_field.view(-1, self.clip_net.n_input_dims - 1)

        # Combine hashgrid features with scale information
        clip_pass = self.clip_net(torch.cat([hashgrid_field, scale.view(-1, 1)], dim=-1)).view(
            *ray_samples.frustums.shape, -1
        )

         # Normalize the resulting CLIP embedding.
        output = clip_pass / clip_pass.norm(dim=-1, keepdim=True)

        return output # Return the normalized CLIP embedding.
