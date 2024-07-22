import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames



parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--white_background',type=bool, default=True, help='if true make white ground for black background')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
seed_everything(args.seed)




config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')


# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files

# if os.path.isdir(args.input_path):
#     input_files = [
#         os.path.join(args.input_path, file) 
#         for file in os.listdir(args.input_path) 
#         if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
#     ]
# else:
#     input_files = [args.input_path]
# print(f'Total number of input images: {len(input_files)}')

assert os.path.isdir(args.input_path), 'Input path must be a directory.'
idxs = [2,6,10,14]
input_files = [f'colors_{i}.png' for i in idxs]
input_files = [
    os.path.join(args.input_path, file)
    for file in input_files
]

input_images = [np.array(Image.open(file).convert('RGB')) for file in input_files]


input_files_single = [f'colors_single_{i}.png' for i in idxs]
input_files_single = [
    os.path.join(args.input_path, file)
    for file in input_files_single
]
input_images_single = [np.array(Image.open(file).convert('RGB')) for file in input_files_single]


seg_files = [f'instance_segmaps_{i}.png' for i in idxs]
backgrounds = [os.path.join(args.input_path, file) for file in seg_files]
backgrounds = [np.array(Image.open(file)) for file in backgrounds]
if args.white_background:
    input_images = [(seg == 1)[..., None]  * img + (seg !=1) [..., None] *255 for img, seg in zip(input_images, backgrounds)]
    


outputs = []

input_images = np.stack(input_images).astype(np.float32) / 255.0
input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous().float()
input_images_single = np.stack(input_images_single).astype(np.float32) / 255.0
input_images_single = torch.from_numpy(input_images_single).permute(0, 3, 1, 2).contiguous().float()

backgrounds = np.stack(backgrounds)
backgrounds = torch.from_numpy(backgrounds)


outputs.append({'name': 'test', 'images': input_images, 'images_single': input_images_single, 'mask': backgrounds})

import h5py
cameras_file = os.path.join(args.input_path, 'data.h5')
cameras_h5 = h5py.File(cameras_file, 'r')
Ks = torch.tensor(cameras_h5['K'][idxs])
cam_poses = cameras_h5['cam_poses'][idxs]
extrinsics = torch.tensor(cam_poses).float().flatten(-2)[:, :12]
intrinsics = Ks.float().flatten(-2)
intrinsics = torch.stack([intrinsics[:, 0], intrinsics[:, 4], intrinsics[:, 2], intrinsics[:, 5]], dim = -1) / 511

cameras = torch.cat([extrinsics, intrinsics], dim=-1)
input_cameras = cameras.unsqueeze(0).repeat(1, 1, 1).to(device)


# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()









### run LRM

# input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    images_single = sample['images_single'].unsqueeze(0).to(device)
    images_single = v2.functional.resize(images_single, 320, interpolation=3, antialias=True).clamp(0,1)

    mask = (sample['mask'] == 2).unsqueeze(0).to(device)
    mask = v2.functional.resize(mask, 320, interpolation=0).float()

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes_mask(images, input_cameras, mask)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # get video
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=args.distance, 
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")

