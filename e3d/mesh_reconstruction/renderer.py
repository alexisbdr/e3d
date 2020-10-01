from pytorch3d.renderer import (
    SfMPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, 
    BlendParams, SoftSilhouetteShader, HardPhongShader, PointLights
)

"""
A set of functional interfaces for creating pytorch3d renderers
TODO: need to move values to params
"""


def silhouette_renderer(img_size: tuple):

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size= RenderParams.img_size[0], 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    return silhouette_renderer


def flat_renderer(img_size: tuple):
    
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_size[0], 
        blur_radius=1e-5, 
        faces_per_pixel=1, 
    )
    
    # We can add a point light in front of the object. 
    lights = PointLights(
        device=device, 
        location=[[3.0, 3.0, 0.0]], 
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((1.0, 1.0, 1.0),),
    )
    
    flat_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(device=device, lights=lights)
    )
    
    return flat_renderer
    
    