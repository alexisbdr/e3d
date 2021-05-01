import numpy as np
from pytorch3d.renderer import (BlendParams, HardFlatShader, MeshRasterizer,
                                MeshRenderer, PointLights,
                                RasterizationSettings, PerspectiveCameras,
                                SoftSilhouetteShader)


def silhouette_renderer(img_size: tuple, device: str):

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
        perspective_correct=False
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    return silhouette_renderer


def flat_renderer(img_size: tuple, device: str):

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0,
        faces_per_pixel=1,
        max_faces_per_bin=5000,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device,
        location=[[-3, 4, -3]],
        diffuse_color=((0.5, 0.5, 0.5),),
        specular_color=((0.5, 0.5, 0.5),),
    )

    flat_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardFlatShader(device=device, lights=lights),
    )

    return flat_renderer
