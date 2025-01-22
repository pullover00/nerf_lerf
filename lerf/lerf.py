from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.viewer_elements import *
from torch.nn import Parameter

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.lerf_field import LERFField
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from lerf.lerf_renderers import CLIPRenderer, MeanRenderer


@dataclass
#Adds hyperparameters specific to lerf
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    # Stuff for spatial data. Hashgrid is like a voxel grid (3 lines below changed by myself)
    hashgrid_layers: Tuple[float, float] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[float, float]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[float, float] = (19, 19)

# Extending Nerfacto Model
class LERFModel(NerfactoModel):
    config: LERFModelConfig

     # Initialize and populate additional modules required for LERF
    def populate_modules(self):
        super().populate_modules()

        # Initialize renderers for CLIP-based relevancy and mean embeddings
        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        # Image encoder for embedding
        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.lerf_field = LERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )

        from lerf.interaction import LERFInteraction
        self.click_scene: LERFInteraction = LERFInteraction(
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            #scale_handle=self.scale_slider,
            model_handle=[self]
            )
        
     # Compute the maximum relevancy across scales for each query phrase
    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # Initialize variables to store maximum relevancy and corresponding scales
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]

        # Iterate over scales and compute relevancy
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                # Compute relevancy output from the hashgrid
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            # Update maximum relevancy for each word in the query
            # j = Number of phrases!
            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    # Get relevancies for query j
                    probs = self.image_encoder.get_relevancy(clip_output, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    # Compute outputs of given ray bundle
    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # Sample rays and compute outputs
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        # Compute  Nerfacto outputs
        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        # Select LERF samples based on weights
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        # Apply best ID selection to ray samples
        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        # Apply gathering to LERF ray samples
        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)
        # Can I extract these lerf samples????

        # Adjust scales for training mode
        if self.training:
            with torch.no_grad():
                clip_scales = ray_bundle.metadata["clip_scales"]
                clip_scales = clip_scales[..., None]
                dist = (lerf_samples.frustums.get_positions() - ray_bundle.origins[:, None, :]).norm(
                    dim=-1, keepdim=True
                )
            clip_scales = clip_scales * ray_bundle.metadata["height"] * (dist / ray_bundle.metadata["fy"])
        else:
            clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        # Check for scale overrides in the metadata
        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )

        # Append weights 
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # Compute depth outputs for LERF field
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # Compute LERF field outputs including DINO and CLIP embeddings
        lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)
        #print(type(lerf_field_outputs), lerf_field_outputs)
        outputs["clip"] = self.renderer_clip(
            embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
        )
        outputs["dino"] = self.renderer_mean(
            embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
        )

        # If not in training, compute relevancy across scales
        if not self.training:
            with torch.no_grad():
                max_across, best_scales = self.get_max_across(
                    lerf_samples,
                    lerf_weights,
                    lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                    clip_scales.shape,
                    preset_scales=override_scales,
                )
                outputs["raw_relevancy"] = max_across  # N x B x 1
                outputs["best_scales"] = best_scales.to(self.device)  # N

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """

        # Set up configuration for batch processing
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)

        # Iterate through ray batches to calculate the best scales for each query
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk # Slice the ray bundle into chunks
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)  # Get the sliced ray bundle
            outputs = self.forward(ray_bundle=ray_bundle)   # Perform a forward pass on the sliced ray bundle
            
        # Store the best scale for each query phrase across batches
            if i == 0:
                best_scales = outputs["best_scales"]
                best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
            else:
                # Compare relevancy across batches to track the best scale
                for phrase_i in range(outputs["best_scales"].shape[0]):
                    m = outputs["raw_relevancy"][phrase_i, ...].max() # get max relevancy across all scales
                    if m > best_relevancies[phrase_i]:
                        best_scales[phrase_i] = outputs["best_scales"][phrase_i] # Update best scales
                        best_relevancies[phrase_i] = m # Update best relevancy

        # After determining the best scales, re-render the outputs for each batch
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle.metadata["override_scales"] = best_scales
            outputs = self.forward(ray_bundle=ray_bundle)

            # Concatenate the outputs across batches
            for output_name, output in outputs.items():  # type: ignore
                if output_name == "best_scales":
                    continue # Skip 'best_scales' as it's already processed
                if output_name == "raw_relevancy":
                     # Append relevancy output to the list of outputs
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)

        # Combine the outputs from all batches            
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                continue

            # Concatenate the tensors and reshape for the final output
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        # Post-process the relevancy outputs for visualization    
        for i in range(len(self.image_encoder.positives)):
            p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1) # Normalize relevancy output to [0, 1]
            outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo")) # Apply a colormap to the normalized relevancy
            #mask = (outputs["relevancy_0"] < 0.5).squeeze() # Create a mask for areas with low relevancy (less than 0.5)
            #outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :] # Replace low relevancy areas with the original RGB image
            mask = (outputs["relevancy_0"] < 0.8).squeeze() # Create a mask for areas with high relevancy (less than 0.5)
            outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :] # Replace low relevancy areas with the original RGB image

        return outputs
    
        # # # Function to get relevan outputs out of NERF # # # 

        # Iterate through positive image_encoders:
        #for i in range(len(self.image_encoder.positives)):
        #    p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1) # Normalize relevancy output to [0, 1]
        #    outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo")) # Apply a colormap to the normalized relevancy
        #    # THRESHOLDING IS HERE!
        #    mask_high_relevancy = (outputs["relevancy_0"] > 0.8).squeeze() # Create a mask for areas with high relevancy (less than 0.5)
        #    outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :] # Replace low relevancy areas with the original RGB image
        #return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        normals = self.config.predict_normals
        field_outputs = self.field(ray_samples, normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        # Render RGB, depth, and accumulation from the field outputs
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        normals = self.config.predict_normals

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normals": normals
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups
    

