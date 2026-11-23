import os
import cv2
import imageio.v2 as imageio
import torch
import PIL.Image
import torchvision.transforms as tvf
import torchvision.transforms.functional as F
import numpy as np
from dust3r.utils.geometry import geotrf
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


ratios_resolutions = {
    224: {1.0: [224, 224]},
    512: {4 / 3: [512, 384], 32 / 21: [512, 336], 16 / 9: [512, 288], 2 / 1: [512, 256], 16 / 5: [512, 160]}
}



def get_mask_bbox(mask):
    """
    Finds the bounding box (height, width) of a binary mask using NumPy.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1=object, 0=background.

    Returns:
        int, int: Bounding box height and width.
    """
    # Step 1: Find nonzero (foreground) pixel locations
    mask_indices = np.argwhere(mask > 0)  # Returns (y, x) coordinates

    if mask_indices.size == 0:
        raise ValueError("The mask is empty! No foreground pixels found.")

    # Step 2: Compute min/max coordinates
    ymin, xmin = mask_indices.min(axis=0)
    ymax, xmax = mask_indices.max(axis=0)

    # Step 3: Compute bounding box dimensions
    bbox_height = ymax - ymin + 1  # Include both ends
    bbox_width = xmax - xmin + 1  # Include both ends

    return bbox_height, bbox_width


def get_HW_resolution(H, W, maxdim, patchsize=16):
    assert maxdim in ratios_resolutions, "Error, maxdim can only be 224 or 512 for now. Other maxdims not implemented yet."
    ratios_resolutions_maxdim = ratios_resolutions[maxdim]
    mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
    islandscape = (W >= H)
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions_maxdim[selkey]
    # check patchsize and make sure output resolution is a multiple of patchsize
    if isinstance(patchsize, tuple):
        assert len(patchsize) == 2 and isinstance(patchsize[0], int) and isinstance(
            patchsize[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
        patchsize = patchsize[0]
    assert max(res) == maxdim
    assert min(res) in mindims
    return res[::-1] if islandscape else res  # return HW



def crop_resize_transform(mask, maxdim, patch_size, padding=10):
    """
    Creates a transformation function to crop and resize an image based on the segmentation mask.

    Args:
        mask (np.array): Binary mask
        maxdim (int): Maximum allowed resolution (e.g., 224 or 512).
        patch_size (int): The final resolution should be a multiple of this.
        padding (int): Extra pixels to include around the bounding box (if within image boundaries).

    Returns:
        function: A resize function (for later use on images).
        np.array: Forward transformation matrix (original → resized).
        np.array: Inverse transformation matrix (resized → original).
    """
    H, W = mask.shape  # Get original dimensions

    # Step 1: Get Bounding Box Around Mask
    mask_indices = np.argwhere(mask > 0)  # Get nonzero pixels
    if mask_indices.size == 0:
        raise ValueError("Segmentation mask is empty, cannot crop.")

    ymin, xmin = mask_indices[:, 0].min(), mask_indices[:, 1].min()
    ymax, xmax = mask_indices[:, 0].max(), mask_indices[:, 1].max()

    # Step 2: Apply Padding (ensure it stays within image bounds)
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(W, xmax + padding)
    ymax = min(H, ymax + padding)

    crop_W = xmax - xmin
    crop_H = ymax - ymin

    # Step 3: Compute Target Resolution Based on Model Requirements
    target_H, target_W = get_HW_resolution(crop_H, crop_W, maxdim=maxdim, patchsize=patch_size)
    target_ratio = target_W / target_H  # Target aspect ratio

    # Step 4: Adjust Crop Size to Match Target Aspect Ratio
    crop_ratio = crop_W / crop_H

    if crop_ratio < target_ratio:  # Crop is too tall, increase width
        new_crop_W = int(crop_H * target_ratio)
        new_crop_W = min(new_crop_W, W)  # Ensure within image bounds
        xmin = max(0, xmin - (new_crop_W - crop_W) // 2)  # Center adjust
        xmax = min(W, xmin + new_crop_W)
    elif crop_ratio > target_ratio:  # Crop is too wide, increase height
        new_crop_H = int(crop_W / target_ratio)
        new_crop_H = min(new_crop_H, H)  # Ensure within image bounds
        ymin = max(0, ymin - (new_crop_H - crop_H) // 2)  # Center adjust
        ymax = min(H, ymin + new_crop_H)

    # Update final crop dimensions
    crop_W = xmax - xmin
    crop_H = ymax - ymin

    # Step 5: Define Transformation Matrices for Cropping
    to_orig_crop = np.array([[1, 0, xmin],  # Translate x
                             [0, 1, ymin],  # Translate y
                             [0, 0, 1]])

    to_rescaled_crop = np.array([[1, 0, -xmin],  # Inverse translation x
                                 [0, 1, -ymin],  # Inverse translation y
                                 [0, 0, 1]])

    # Step 6: Define Resize Operation
    def resize_func(img):
        """Applies cropping and resizing to an image."""
        img = F.crop(img, top=ymin, left=xmin, height=crop_H, width=crop_W)  # Crop
        img = F.resize(img, size=(target_H, target_W))  # Resize
        return img

    # Step 7: Compute Transformation Matrices for Resizing
    to_orig_resize = np.array([[crop_W / target_W, 0, 0],
                               [0, crop_H / target_H, 0],
                               [0, 0, 1]])

    to_rescaled_resize = np.array([[target_W / crop_W, 0, 0],
                                   [0, target_H / crop_H, 0],
                                   [0, 0, 1]])

    # Step 8: Compute Final Transformations
    to_resize = to_rescaled_resize @ to_rescaled_crop  # Maps original → resized
    to_orig = to_orig_crop @ to_orig_resize  # Maps resized → original

    return resize_func, to_resize, to_orig, crop_W, crop_H




def read_file_from_dir(path):
    frames = []
    files = sorted([x for x in os.listdir(path) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".npy")])
    for file in files:
        # res = np.load(os.path.join(path, file))
        if file.endswith(".npy"):
            res = np.load(os.path.join(path, file))
        elif file.endswith(".png") or file.endswith(".jpg"):
            res = imageio.imread(os.path.join(path, file))
        frames.append(res)
    return frames, files


def post_process_matches(matches, to_orig):
    # in-place operation
    # from cv2 to colmap
    # matches = matches.astype(np.float64)
    matches[:, 0] += 0.5
    matches[:, 1] += 0.5
    # rescale coordinates
    matches = geotrf(to_orig, matches, norm=True)
    # from colmap back to cv2
    matches[:, 0] -= 0.5
    matches[:, 1] -= 0.5
    return matches



def tensor_to_img(img_tensor):
    """
    Converts a normalized PyTorch tensor (C, H, W) to a format compatible with OpenCV.

    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W) with values in range [-1, 1].

    Returns:
        np.ndarray: OpenCV-compatible image (H, W, 3) with values in [0, 255], RGB format.
    """
    # Step 1: Move tensor to CPU and convert to NumPy
    img_np = img_tensor.detach().cpu().numpy()

    # Step 2: Convert from (C, H, W) to (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))  # Rearrange dimensions to (H, W, C)

    # Step 3: Denormalize (undo the normalization applied by ImgNorm)
    img_np = (img_np * 0.5) + 0.5  # Convert back to range [0, 1]

    # Step 4: Convert to [0, 255] and uint8
    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def filter_segmentation_ids(mask: np.ndarray, ratio_threshold=0.9, pixel_threshold=200):
    """
    Filters segmentation IDs based on their appearance ratio across T frames,
    considering only frames where the segmentation id appears with at least
    `pixel_threshold` pixels.

    Args:
        mask (np.ndarray): A 3D array of shape (T, H, W) where each element is a segmentation id.
        ratio_threshold (float): A value between 0 and 1. Only segmentation ids that appear in at least
                                 this fraction of frames (with sufficient pixels) will be returned.
        pixel_threshold (int): Minimum number of pixels required in a frame for that segmentation id to count as present.

    Returns:
        list: A list of segmentation ids that meet the appearance ratio criteria.
    """
    T = mask.shape[0]
    segmentation_ids = np.unique(mask)

    valid_ids = []
    for seg_id in segmentation_ids:
        if seg_id == 0:
            continue
        # Determine for each frame whether the pixel count of seg_id is above the pixel threshold
        count_pixels_per_frame = np.sum(mask == seg_id, axis=(1, 2))
        present_in_frame = count_pixels_per_frame >= pixel_threshold

        # Calculate the ratio of frames where seg_id is present with enough pixels
        appearance_ratio = np.sum(present_in_frame) / T

        if appearance_ratio >= ratio_threshold:
            valid_ids.append(seg_id)

    return valid_ids


def find_max_pixel_frame(mask: np.ndarray, seg_ids: list, disable_max_pixel: bool = False):
    """
    For each segmentation id in seg_ids, finds the frame index in which the id has the largest pixel area.
    If disable_max_pixel is True, selects a random frame where the segmentation id is present instead.

    Args:
        mask (np.ndarray): A 3D array of shape (T, H, W) where each element is a segmentation id.
        seg_ids (list): A list of segmentation ids (e.g., returned from filter_segmentation_ids).
        disable_max_pixel (bool): If True, select a random frame where the segmentation id exists rather than
                                  the frame with the maximum pixel area.

    Returns:
        dict: A dictionary where each key is a segmentation id and the value is the frame index.
    """
    results = {}
    T = mask.shape[0]

    for seg_id in seg_ids:
        # Count pixels for seg_id in each frame, resulting in a (T,) array.
        pixel_counts = np.sum(mask == seg_id, axis=(1, 2))

        if disable_max_pixel:
            # Find indices of frames where the segmentation id appears.
            valid_frames = np.nonzero(pixel_counts)[0]
            if valid_frames.size > 0:
                frame_index = np.random.choice(valid_frames)
            else:
                frame_index = 0  # Or handle as needed if the seg_id never appears.
        else:
            # Select the frame with the maximum pixel count.
            frame_index = np.argmax(pixel_counts)

        results[seg_id] = frame_index

    return results




def optimal_segmentation_matches(match_matrix: np.ndarray, min_matches: float):
    """
    Finds the optimal one-to-one matching pairs between segmentation ids from two images,
    where each seg id in M (rows) can be matched with one seg id in N (columns) at most once.
    Only pairs with a non-zero matching score and a score that is at least min_matches are considered.

    This function uses SciPy's linear_sum_assignment. It converts the problem into a minimization
    problem by using negative scores and excludes zero or low-score matches by setting them to a large cost.

    Args:
        match_matrix (np.ndarray): A 2D array of shape (M, N) where each element represents
                                  the matching score between segmentation ids.
        min_matches (float): Minimum allowed matching score for a pair to be included.

    Returns:
        list of tuples: Each tuple (i, j) indicates a match between segmentation id at row i in M
                        and segmentation id at column j in N that meets the min_matches threshold.
    """
    M, N = match_matrix.shape
    # Convert to a minimization problem: for allowed matches, cost = -score; for disallowed or low-score matches, set cost high.
    high_cost = 1e9
    # Use min_matches threshold: if match_matrix value is less than min_matches, treat it as disallowed.
    cost_for_assignment = np.where(match_matrix >= min_matches, -match_matrix, high_cost)

    # Pad the cost matrix to be square.
    L = max(M, N)
    padded_cost = np.full((L, L), high_cost)
    padded_cost[:M, :N] = cost_for_assignment

    # Use linear_sum_assignment to get the assignment indices.
    row_ind, col_ind = linear_sum_assignment(padded_cost)

    matches = []
    for i, j in zip(row_ind, col_ind):
        # Only keep assignments that are within the original dimensions.
        if i < M and j < N:
            # Only include the match if the original match score is both non-zero and meets the min_matches threshold.
            if match_matrix[i, j] >= min_matches:
                matches.append((i, j))

    return matches



def get_ordered_colors(N):
    """
    Generate a list of N distinct colors in an ordered sequence using the 'jet' colormap.

    Parameters:
        N (int): Number of colors to generate.

    Returns:
        colors (list of tuple): A list of N RGBA tuples, each in the form (R, G, B, A) with values in [0, 1].
    """
    cmap = plt.cm.get_cmap('jet', N)
    colors = [cmap(i)[:3] for i in range(N)]
    colors = [tuple(int(255 * c) for c in color) for color in colors]
    return colors


def visualize_mask(segmentation, image, colors):
    """
    Visualizes a segmentation overlay on the original BGR image.

    Args:
        segmentation (numpy.ndarray): Segmentation map of shape (H, W) with segmentation ids.
        image (numpy.ndarray): Original BGR image of shape (H, W, 3).
        colors tuple

    Returns:
        visual_image (numpy.ndarray): Image with segmentation overlay.
    """
    H, W = segmentation.shape
    visual_image = image.copy()

    # Create a blank color overlay
    color_overlay = np.zeros((H, W, 3), dtype=np.uint8)

    # Get unique segmentation ids in the map
    seg_ids = np.unique(segmentation)

    assert len(seg_ids) <= len(colors), "Number of colors should be at least as many as unique segmentation ids."

    # Iterate over segmentation ids, ignoring background (assumed to be 0)
    for idx, seg_id in enumerate(seg_ids):
        if seg_id == 0:
            continue

        # Create a binary mask for the current seg_id
        mask = segmentation == seg_id

        # Retrieve the corresponding color; if not available, use a default color (green)
        color = colors[idx]

        # Assign the color to the overlay where mask is True
        color_overlay[mask] = color

    # Define an alpha for blending
    alpha = 0.8

    # Blend the overlay on the original image where the segmentation is non-background
    mask_overlay = segmentation > 0
    visual_image[mask_overlay] = (
        alpha * color_overlay[mask_overlay] + (1 - alpha) * visual_image[mask_overlay]
    ).astype(np.uint8)

    return visual_image


def resize_image(img, target_size=512):
    """
    resize an image so that its largest dimension equals target_size while maintaining the aspect ratio.

    Parameters:
        image_path (str): Path to the input image.
        target_size (int): The maximum dimension for the output image. Default is 512.

    Returns:
        img_resized (numpy.ndarray): The resized image.
    """
    height, width = img.shape[:2]
    scale = target_size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = cv2.resize(img, (new_width, new_height))

    return img_resized


def process_image(image, maxdim, patch_size, load_mask, mask):
    """Prepare the full image and map MASt3R coordinates back to source pixels."""
    from PIL import Image

    height, width = image.shape[:2]
    target_height, target_width = get_HW_resolution(height, width, maxdim, patch_size)
    rgb = Image.fromarray(image).convert("RGB")
    resized = rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)
    valid = mask > 0 if load_mask else np.ones((height, width), dtype=bool)
    valid_rescaled = Image.fromarray(valid).resize(
        (target_width, target_height), Image.Resampling.NEAREST
    )
    return {
        "rgb": rgb,
        "rgb_rescaled": F.to_tensor(resized).mul(2).sub(1),
        "valid": valid,
        "valid_rescaled": torch.from_numpy(np.array(valid_rescaled)),
        "to_orig": np.diag([width / target_width, height / target_height, 1.0]),
    }
