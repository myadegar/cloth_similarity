from PIL import Image
from typing import List
import warnings
import numpy as np
from scipy.ndimage import binary_erosion
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from cloth_segmentation.data.base_dataset import Normalize_image
from cloth_segmentation.utils.saving_utils import load_checkpoint_mgpu
from cloth_segmentation.networks import U2NET

do_palette = False
def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

def load_segment_model(checkpoint_path, device):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net

palette = get_palette(4)

kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # convert again to binary
    return mask

def alpha_matting_cutout(img: Image, mask: Image,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> Image:

    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")

    img = np.asarray(img)
    mask = np.asarray(mask)

    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout

def naive_cutout(img: Image, mask: Image) -> Image:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout

def get_concat_v(img1: Image, img2: Image) -> Image:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst

def get_concat_v_multi(imgs) -> Image:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot

def cloth_segment(input_image, net, device,
                  post_process_mask: bool = False,
                  alpha_matting: bool = False,
                  alpha_matting_foreground_threshold: int = 240,
                  alpha_matting_background_threshold: int = 10,
                  alpha_matting_erode_size: int = 10,
                  concat_parts: bool = False,
                  max_img_size: int = 1024
                  ):

    img_w, img_h = input_image.size
    max_dim = max(input_image.size)
    if max_dim > max_img_size: # resize
        ratio = max_img_size / (max_dim + 1)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        image = input_image.resize((new_w, new_h))
        do_resize = True
    else:
        image = input_image.copy()
        do_resize = False

    image_tensor = transform_rgb(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    ##
    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    ##
    all_masks = [output_arr == i for i in range(1, 4)]
    segment_images, mask_images = [], []
    for mask in all_masks:
        mask = mask.astype('uint8') * 255
        if post_process_mask:
            mask = post_process(mask)
        mask = Image.fromarray(mask)
        if do_resize: # back to origin size
            mask = mask.resize((img_w, img_h))
        if alpha_matting:
            cutout = alpha_matting_cutout(input_image, mask,
                alpha_matting_foreground_threshold,
                alpha_matting_background_threshold,
                alpha_matting_erode_size
                                         )
        else:
            cutout = naive_cutout(input_image, mask)
        mask_images.append(mask)
        segment_images.append(cutout)

    if concat_parts:
        segment_images = [segment_image.crop(segment_image.getbbox()) for segment_image in segment_images]
        segment_image = get_concat_v_multi(segment_images)
        segment_images = [segment_image]

    return segment_images, mask_images






