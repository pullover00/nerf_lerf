import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig

from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from lerf.data.lerf_datamanager import (
    LERFDataManager,
    LERFDataManagerConfig,
)
from lerf.lerf import LERFModel, LERFModelConfig
from lerf.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder

# UI libraries
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.viewer.viewer_elements import *
from pathlib import Path


@dataclass
class LERFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: LERFPipeline)
    """target class to instantiate"""
    datamanager: LERFDataManagerConfig = LERFDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = LERFModelConfig()
    """specifies the model config"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""

class LERFPipeline(VanillaPipeline):  
    def __init__(
        self,
        config: LERFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.image_encoder: BaseImageEncoder = config.network.setup()

        self.datamanager: LERFDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            image_encoder=self.image_encoder,
         )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            image_encoder=self.image_encoder,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(LERFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        # UI Elements
        self.viewer_control = ViewerControl()
        self.export_pcd: ViewerButton = ViewerButton(
            name="Export Point Cloud", cb_hook=self._export_point_cloud
        )
        self.export_pcd: ViewerButton = ViewerButton(
            name="Find highest relevancy", cb_hook=self.find_highest_relevancy_3d_points
        )

        # Buttons and controls for interaction
        #self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_nerf) # in the code
        self.click_location = None
        self.click_handle = None

    def _update_interaction_method(self, dropdown: ViewerDropdown):
        """
        Update the UI based on the interaction method
        """
        hide_in_interactive = (not (dropdown.value == "Interactive")) # # Hide elements in interactive mode

        self._click_nerf.set_hidden(hide_in_interactive) # adjusted
        self.crop_to_click.set_hidden(hide_in_interactive)
        self.crop_to_group_level.set_hidden(hide_in_interactive)
        self.move_current_crop.set_hidden(hide_in_interactive)
    
    def _click_nerf(self, button: ViewerButton):
        # Maybe depth is missing
        """
        Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details.
        """
        # Process the click event to capture 3D point location
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)

            # Re-enable the click Nerf button and crop options
            self._click_nerf.set_disabled(False)
            self.crop_to_click.set_disabled(False)

            # Unregister the click callback once the click is processed
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        # Disable the click Gaussian button to prevent multiple registrations
        self.click_gaussian.set_disabled(True)

        # Register the click callback to handle the click event
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """
        On click, calculate the 3D position of the click and visualize it.
        """

        # Get the camera transformation matrix
        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]

        # Import the transform library for manipulation
        import viser.transforms as vtf

        # Compute the inverse rotation to convert from world to camera coordinates
        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()

        # Apply the camera transformation to the click direction
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()

        # Project the click direction onto 2D camera coordinates
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])

        # Evaluate the model and get the depth at the clicked pixel
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()

        # Get the depth value at the clicked pixel, and calculate the 3D location
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        # Calculate the click location in 3D space
        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        # Create a visual marker at the clicked location
        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # Set marker color to green
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )

    ## @torch.no_grad()
    def find_highest_relevancy_3d_points(self, ViewerClick):
    #    #camera_ray_bundle: RayBundle, outputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
    #    Calculates the 3D position of the highest relavancy and visualizes it.
    #    Args:
    #        camera_ray_bundle: The input ray bundle containing ray origins, directions, and bounds.
    #        outputs: The output dictionary from the model containing relevancy and other results.
    #    Returns:
    #        List of 3D coordinates for the highest relevancy points for each query.
    #    """
        
        # get the bundle with highest relevancy
        
        from lerf.lerf import L
        highest_relevancy_points = []
    #    for i in range(len(self.image_encoder.positives)):  # Iterate over each relevancy map (e.g., relevancy_0, relevancy_1)
    #        # Get the relevancy map for the current query
    #        relevancy_map = outputs[f"relevancy_{i}"]  # Shape: (H, W)
    #        # Find the pixel location with the highest relevancy
    #        max_relevancy_value, max_idx = torch.max(relevancy_map.view(-1), dim=0)
    #        max_y, max_x = torch.div(max_idx, relevancy_map.shape[1], rounding_mode="trunc"), max_idx % relevancy_map.shape[1]
    #        # Extract the corresponding ray origin and direction
    #        ray_origin = camera_ray_bundle.origins[max_y, max_x]  # Origin of the ray
    #        ray_direction = camera_ray_bundle.directions[max_y, max_x]  # Direction of the ray
    #        # Find the 3D point along the ray
    #        # Assume `t` is chosen to maximize the relevancy; if `t` isn't provided, use `nears` as an estimate
    #        t = camera_ray_bundle.nears[max_y, max_x]  # Starting distance along the ray
    #        point_3d = ray_origin + t * ray_direction  # Compute the 3D point
    #        # Store the result
    #        highest_relevancy_points.append(point_3d)
    #        print(f"Relevancy_{i}: Max Value = {max_relevancy_value}, 3D Point = {point_3d}")
        return highest_relevancy_points
#
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
            config_path = Path('outputs/figurines/lerf/2025-01-16_103610/config.yml')

            # Debugging for datamanager
            if not hasattr(self, 'datamanager') or not hasattr(self.datamanager, 'config'):
                raise AttributeError("Datamanager or its configuration is not properly initialized.")

            if not hasattr(self.datamanager.config, 'dataparser') or not hasattr(self.datamanager.config.dataparser, 'data'):
                raise AttributeError("Dataparser or its data attribute is not properly initialized in datamanager config.")
            
            # Output directory
            output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
            filename = Path(output_dir) / "point_cloud.ply"

            # Create an instance of ExportPointCloud with desired parameters
            from nerfstudio.scripts.exporter import ExportGaussianSplat
            exporter = ExportPointCloud(
                load_config=config_path,
                output_dir=output_dir,
                num_points=num_points,
                remove_outliers=remove_outliers,
                reorient_normals=reorient_normals,
                normal_method=None,  # Set to None if not used
                normal_output_name=normal_output_name,
                depth_output_name=depth_output_name,
                rgb_output_name=rgb_output_name,
                obb_center=self.selected_location if self.selected_location is not None else (0.0, 0.0, 0.0),
                obb_rotation=(0.0, 0.0, 0.0),
                obb_scale=(1.0, 1.0, 1.0),
            )

            # Call the main export method
            exporter.main()

            print(f"Point cloud successfully exported to {filename}")
        except Exception as e:
            print(f"Error exporting point cloud: {e}")

