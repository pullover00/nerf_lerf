"""Helper functions for interacting/visualization with GARField model."""
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

#from nerfstudio.scripts.exporter import ExportPointCloud

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
            #scale_handle: ViewerSlider,
            model_handle: List[LERFModel]
        ):
        """
        Initializes the Export UI component.

        Args:
            device: The device (CPU/GPU) for computations.
            model_handle: A list containing references to the GARField model.
        """
        super().__init__()
        self.add_click_button: ViewerButton = ViewerButton(
            name="Export Pointcloud", cb_hook=self._export_point_cloud
        )
        #self.del_click_button: ViewerButton = ViewerButton(
        #    name="Reset Click", cb_hook=self._del_click_cb
        #)
        self.viewer_control: ViewerControl = ViewerControl()

        # Store references to slider and model
        #self.scale_handle = scale_handle
        self.model_handle = model_handle
        #self.scale_handle.cb_hook = self._update_scale_vis

        # Initialize handles and variables
        self._click_handle = None
        self._box_handle = None
        self.selected_location = None
        self.device = device

        # Initialize exporting infos
        self.z_export_options_camera_path_filename = ViewerText("Camera Path Filename", "", visible=False)
        #self.z_export_options_camera_path_render = ViewerButton("Render Current Pipeline", cb_hook=self.render_from_path, visible=False)


    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_camera_path_filename.set_hidden(not checkbox.value)
        self.z_export_options_camera_path_render.set_hidden(not checkbox.value)

    def _export_point_cloud(self, element):
        """
        Callback function to export the point cloud of the whole scene.
        """
        try:
            # Define parameters
            num_points = 1000000  # Number of points to generate
            remove_outliers = True
            reorient_normals = False
            normal_method = "open3d"
            rgb_output_name = "rgb"
            depth_output_name = "depth"
            normal_output_name = "normals"
            bounding_box =  (0.0, 0.0, 0.0)
            #config_path = Path('pointcloud/point_cloud.pcd')
            config_path = Path('outputs/figurines/lerf/2025-01-16_103610/config.yml')
            #print(config_path)
            output_dir = Path('exports/pcd/')   # Define the output directory

            # Create an instance of ExportPointCloud with desired parameters
            exporter = ExportPointCloud(
                load_config = config_path,
               # load_config=self.load_config,  # Ensure this points to your YAML config
                output_dir = output_dir,
                num_points = num_points,
                remove_outliers = remove_outliers,
                reorient_normals = reorient_normals,
                normal_method = None, #normal_method,
                normal_output_name = normal_output_name,
                depth_output_name = depth_output_name,
                rgb_output_name = rgb_output_name,
                obb_center = bounding_box,
                obb_rotation = bounding_box,
                obb_scale = bounding_box,
            )

            # Call the main export method
            exporter.main()

            print(f"Point cloud successfully exported to {output_dir / 'point_cloud.ply'}")
        except Exception as e:
            print(f"Error exporting point cloud: {e}")

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
        sphere_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/hit_pos", mesh=sphere_mesh
        )
        self._click_handle = sphere_mesh_handle
        self.selected_location = np.array(origin + direction * distance)
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

    def _update_scale_vis(self, slider: ViewerSlider):
        """
        Update the scale visualization.
        """
        if self._box_handle is not None:
            self._box_handle.remove()
            self._box_handle = None
        if self.selected_location is not None:
            # Create a wireframe box representing the scale
            box_mesh = trimesh.creation.icosphere(radius=VISER_NERFSTUDIO_SCALE_RATIO*max(0.001, slider.value)/2, subdivision=0)
            self._box_handle = self.viewer_control.viser_server.add_mesh_simple(
                name=f"/hit_pos_box", 
                vertices=box_mesh.vertices,
                faces=box_mesh.faces,
                position=(self.selected_location * VISER_NERFSTUDIO_SCALE_RATIO).flatten(),
                wireframe=True
            )

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