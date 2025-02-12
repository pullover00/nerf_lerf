from typing import List, Optional, Tuple, Union
import viser
import trimesh
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared

from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

from lerf.lerf import LERFModel


class LERFInteraction(nn.Module):
    """UI for clicking on a scene (visualized as spheres).
    This needs to be a nn.Module to allow the viewer to register callbacks.
    """
    _click_handle: viser.GlbHandle # Handle for the click visualization (sphere mesh)
    _box_handle: viser.GlbHandle  # Handle for the scale visualization (box mesh)
    selected_location: np.ndarray # Stores the 3D position of the selected point
    #scale_handle: ViewerSlider  # For getting the scale to query GARField
    model_handle: List[LERFModel]  # Store as list to avoid circular children

    def __init__(
        self,
        device: torch.device,
        model_handle: List[LERFModel],
    ):
        """
        Initializes the Export UI component.

        Args:
            device: The device (CPU/GPU) for computations.
            model_handle: A list containing references to the GARField model.
        """
        super().__init__()

        # UI Components
        self.add_click_button: ViewerButton = ViewerButton(
            name="Click", cb_hook=self._add_click_cb
        )
        self.del_click_button: ViewerButton = ViewerButton(
            name="Reset Click", cb_hook=self._del_click_cb
        )

        # New button for showing the highest relevancy position
        self.show_max_relevancy_button: ViewerButton = ViewerButton(
        name="Show Max Relevancy", cb_hook=self._show_max_relevancy_cb
        )

        self.viewer_control: ViewerControl = ViewerControl()

        # Store references to and model
        self.model_handle = model_handle

        # Initialize handles and variables
        self._click_handle = None
        self._box_handle = None
        self.selected_location = None
        self.device = device

        # Initialize exporting infos
        #self.z_export_options_camera_path_filename = ViewerText("Camera Path Filename", "", visible=False)
        #self.z_export_options_camera_path_render = ViewerButton("Render Current Pipeline", cb_hook=self.render_from_path, visible=False)

    def _show_max_relevancy_cb(self, button: ViewerButton):
        """
        Callback for showing the highest relevancy position with a sphere.
        """
        model = self.model_handle[0]  # Get the LERF model instance
    
        # Get outputs from the latest model run
        outputs = model.get_outputs_for_camera_ray_bundle(model.camera_ray_bundle)
    
        if "highest_relevancy_position" not in outputs:
            print("Error: No highest relevancy position found in outputs.")
            return
    
        # Extract the highest relevancy position
        highest_relevancy_position = outputs["highest_relevancy_position"].cpu().numpy()
    
        # Remove any existing sphere
        self._del_click_cb(None)
    
        # Create a sphere at the highest relevancy position
        sphere_mesh: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)
        sphere_mesh.vertices += highest_relevancy_position
        sphere_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 5.0)  # Red sphere
    
        # Add to the viewer
        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/max_relevancy_pos", mesh=sphere_mesh
        )
        
        self._click_handle = sphere_mesh_handle
        self.selected_location = highest_relevancy_position


    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_camera_path_filename.set_hidden(not checkbox.value)
        self.z_export_options_camera_path_render.set_hidden(not checkbox.value)

    def _add_click_cb(self, button: ViewerButton):
        """
        Button press registers a click event, which will add a sphere.
        Refer more to nerfstudio docs for more details. 
        """
        self.add_click_button.set_disabled(True)
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.add_click_button.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """
        On click, calculate the 3D position of the click and visualize it.
        Also keep track of the selected location.
        """
        # Convert click origin and direction to tensors
        origin = torch.tensor(click.origin).view(1, 3)
        direction = torch.tensor(click.direction).view(1, 3)

        # Create a RayBundle for the click
        bundle = RayBundle(
            origin,
            direction,
            torch.tensor(0.001).view(1, 1),
            nears=torch.tensor(0.05).view(1, 1),
            fars=torch.tensor(100).view(1, 1),
            camera_indices=torch.tensor(0).view(1, 1),
        ).to(self.device)

        # Get the distance/depth to the intersection --> calculate 3D position of the click
        # Get the GARField model instance
        model = self.model_handle[0]

        # Sample rays to find intersection points
        ray_samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
        field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
        if model.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        # Compute depth and calculate the 3D click position
        with torch.no_grad():
            depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        distance = depth[0, 0].detach().cpu().numpy()
        click_position = np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO

        # Update click visualization
        self._del_click_cb(None) # Remove previous visualizations
        sphere_mesh: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)
        sphere_mesh.vertices += click_position
        sphere_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 5.0)  # type: ignore
        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/hit_pos", mesh=sphere_mesh
        )
        self._click_handle = sphere_mesh_handle
        self.selected_location = np.array(origin + direction * distance)
        # Later! Substitute the selected_location with something else. 
        #self._update_scale_vis(self.scale_handle)

    def _del_click_cb(self, button: ViewerButton):
        """
        Remove the click location and click visualizations.
        """
        if self._click_handle is not None:
            self._click_handle.remove()
        self._click_handle = None
        if self._box_handle is not None:
            self._box_handle.remove()
        self._box_handle = None
        self.selected_location = None

    def get_outputs(self, outputs: dict):
        """
        Computes the affinity between the selected 3D point and the points visible in the current rendered view.

        Args:
            outputs: Dictionary of model outputs for the current view.

        Returns:
            Dictionary with the computed instance interaction metrics.
        """
        if self.selected_location is None:
            return None
        location = self.selected_location
    
        #instance_scale = self.scale_handle.value
        
        # mimic the fields call
        grouping_field = self.model_handle[0].grouping_field
        positions = torch.tensor(location).view(1, 3).to(self.device)
        positions = grouping_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        xs = [e(positions.view(-1, 3)) for e in grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        instance_pass = grouping_field.get_mlp(x, torch.tensor([instance_scale]).to(self.device).view(1, 1))

        return {
            "instance_interact": torch.norm(outputs['instance'] - instance_pass.float(), p=2, dim=-1)
        }
    
    def render_from_path(self, button: ViewerButton):
        from nerfstudio.cameras.camera_paths import get_path_from_json
        import json
        from nerfstudio.scripts.render import _render_trajectory_video

        assert self.z_export_options_camera_path_filename.value != ""
        camera_path_filename = Path(self.z_export_options_camera_path_filename.value)
        
        with open(camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        camera_path = get_path_from_json(camera_path)
        self.model.eval()
        with torch.no_grad():
            _render_trajectory_video(
                self,
                camera_path,
                output_filename=Path('render.mp4'),
                rendered_output_names=['rgb'],
                rendered_resolution_scaling_factor=1.0 ,
                seconds=seconds,
                output_format="video",
            )
        self.model.train()