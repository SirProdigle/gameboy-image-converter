from collections import Counter

import gradio as gr
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.spatial import cKDTree
from numpy import std
from PIL import Image, features
import random
import os
import zipfile
import threading
import time
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb  # For color space conversion
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import cdist  # For calculating distances between color sets
import traceback
from skimage.metrics import structural_similarity as ssim
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import logging
import json
import uuid
from contextlib import contextmanager



# Constants for dithering and quantization methods
DITHER_METHODS = {
    "None": Image.Dither.NONE,
    "Floyd-Steinberg": Image.Dither.FLOYDSTEINBERG
}

QUANTIZATION_METHODS = {
    "Median cut": Image.Quantize.MEDIANCUT,
    "Maximum coverage": Image.Quantize.MAXCOVERAGE,
    "Fast octree": Image.Quantize.FASTOCTREE,
    "libimagequant": Image.Quantize.LIBIMAGEQUANT
}

# Structured logging setup
logger = logging.getLogger("gradio_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

active_tasks = 0

@contextmanager
def task_log(task_type="image_convert"):
    global active_tasks
    task_id = str(uuid.uuid4())
    start_time = time.time()
    active_tasks += 1

    logger.info(json.dumps({
        "event": "task_start",
        "task_id": task_id,
        "task_type": task_type,
        "timestamp": start_time,
        "active_tasks": active_tasks,
    }))

    try:
        yield task_id
    finally:
        end_time = time.time()
        duration = end_time - start_time
        active_tasks -= 1

        logger.info(json.dumps({
            "event": "task_finish",
            "task_id": task_id,
            "task_type": task_type,
            "timestamp": end_time,
            "duration": duration,
            "active_tasks": active_tasks,
        }))


def tile_variance(tile):
    """Compute the variance of a tile."""
    arr = np.array(tile)
    return np.std(arr, axis=(0, 1)).mean()  # Compute the mean std deviation across color channels


def tile_similarity(tile1, tile2):
    """Calculate the Hamming similarity between two tiles."""
    # Convert tiles to numpy arrays if they aren't already
    arr1 = np.array(tile1)
    arr2 = np.array(tile2)
    # Flatten arrays to compare them pixel-by-pixel
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    # Calculate Hamming distance
    hamming_distance = np.sum(flat1 != flat2)
    # Normalize the Hamming distance to get a similarity measure
    similarity = 1 - (hamming_distance / flat1.size)
    return similarity


def dominant_color(tile, color_palette):
    # Convert the tile to a NumPy array
    arr = np.array(tile)

    # Check the shape of the array to determine if it's grayscale or color
    if len(arr.shape) == 2:
        # Grayscale image, so reshape it to (-1, 1) instead of (-1, 3)
        arr = arr.reshape(-1, 1)
    elif len(arr.shape) == 3:
        # Color image, so ensure it's reshaped correctly for RGB
        arr = arr.reshape(-1, 3)
    else:
        # Unexpected image format
        raise ValueError("Unexpected image format!")

    # For grayscale images, the dominant 'color' will just be the most common value
    if arr.shape[1] == 1:
        unique, counts = np.unique(arr, return_counts=True)
        dominant = unique[np.argmax(counts)]
        return (dominant,) * 3  # Return as a tuple to keep consistent format with RGB
    else:
        # Find the most frequent color in the case of an RGB image
        unique, counts = np.unique(arr, axis=0, return_counts=True)
        dominant_index = np.argmax(counts)
        return tuple(unique[dominant_index])  # Convert to tuple to match expected format


def apply_gothic_filter(image, threshold, dot_size, spacing, contrast_boost=1.5, edge_enhance=True, noise_factor=0.1,
                        apply_blur=True, irregular_shape=True, irregular_size=True):
    original_mode = image.mode
    if original_mode == 'P':
        image = image.convert('RGB')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_boost)

    # Edge enhancement
    if edge_enhance:
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    img_array = np.array(image)

    # Determine background color
    unique_colors, color_counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
    tree = cKDTree(unique_colors)

    # Choose background color: darkest color among the top 3 most common colors
    top_colors = unique_colors[np.argsort(color_counts)[-3:]]
    background_color = tuple(top_colors[np.argmin(np.sum(top_colors, axis=1))])

    result = Image.new('RGB', image.size, color=background_color)
    draw = ImageDraw.Draw(result)

    # Create a distressed texture if irregular size is enabled
    if irregular_size:
        texture = Image.new('L', image.size)
        texture_draw = ImageDraw.Draw(texture)
        for _ in range(int(image.width * image.height * 0.1)):  # Adjust density as needed
            x = random.randint(0, image.width - 1)
            y = random.randint(0, image.height - 1)
            texture_draw.point((x, y), fill=random.randint(0, 255))

    for y in range(0, image.height, spacing):
        for x in range(0, image.width, spacing):
            original_color = img_array[y, x]
            luminance = 0.299 * original_color[0] + 0.587 * original_color[1] + 0.114 * original_color[2]

            if luminance > threshold:
                _, index = tree.query(original_color)
                nearest_color = tuple(unique_colors[index])

                # Determine dot size
                if irregular_size:
                    texture_value = texture.getpixel((x, y))
                    adjusted_dot_size = max(1, int(dot_size * (texture_value / 255)))
                else:
                    adjusted_dot_size = dot_size

                # Add slight randomness to dot position
                x_offset = int(random.uniform(-spacing / 2, spacing / 2) * noise_factor)
                y_offset = int(random.uniform(-spacing / 2, spacing / 2) * noise_factor)

                if irregular_shape:
                    # Draw an irregular shape
                    points = []
                    for i in range(8):  # 8-sided irregular shape
                        angle = i * (2 * np.pi / 8) + random.uniform(0, np.pi / 4)
                        r = adjusted_dot_size * (1 + random.uniform(-0.2, 0.2))  # Vary the radius
                        px = x + x_offset + int(r * np.cos(angle))
                        py = y + y_offset + int(r * np.sin(angle))
                        points.append((px, py))
                    draw.polygon(points, fill=nearest_color)
                else:
                    # Draw a regular circle
                    draw.ellipse([(x + x_offset - adjusted_dot_size, y + y_offset - adjusted_dot_size),
                                  (x + x_offset + adjusted_dot_size, y + y_offset + adjusted_dot_size)],
                                 fill=nearest_color)

    # Apply a slight blur to soften the effect
    if apply_blur:
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

    if original_mode == 'P':
        # Get the number of colors in the original palette image
        original_colors = image.getcolors()
        if original_colors is None:
            # If there are more than 256 colors, default to 256
            original_num_colors = 256
        else:
            original_num_colors = len(original_colors)

        # Quantize the result to match the original number of colors
        result = result.quantize(colors=original_num_colors, method=Image.MEDIANCUT)

    return result


def most_common_surrounding_color(image, x, y, tile_size, default_color):
    """Calculate the most common color immediately bordering a specific tile."""
    border_colors = []

    # Define the ranges for the bordering pixels
    top_range = (max(0, y - 1), x, min(image.width, x + tile_size))
    bottom_range = (min(image.height, y + tile_size), x, min(image.width, x + tile_size))
    left_range = (y, max(0, x - 1), min(image.height, y + tile_size))
    right_range = (y, min(image.width, x + tile_size), min(image.height, y + tile_size))

    # Sample colors from each bordering side
    for y_pos, x_start, x_end in [top_range, bottom_range]:
        for adj_x in range(x_start, x_end):
            try:
                color = image.getpixel((adj_x, y_pos))
            except IndexError:
                color = (0, 0, 0)

            if color != (0, 0, 0):  # Exclude black if necessary
                border_colors.append(color)
            else:
                border_colors.append(default_color)

    for x_pos, y_start, y_end in [left_range, right_range]:
        for adj_y in range(y_start, y_end):
            try:
                color = image.getpixel((x_pos, adj_y))
            except IndexError:
                color = (0, 0, 0)
            if color != (0, 0, 0):  # Exclude black if necessary
                border_colors.append(color)

    # Find the most common border color
    if border_colors:
        most_common = max(set(border_colors), key=border_colors.count)
        return most_common
    else:
        # Return the default color if no valid bordering colors were found
        return default_color


def get_most_common_color(tile):
    colors = tile.getcolors()
    if tile.mode == 'P':
        # If the tile is palettized, get the most common color from the palette
        palette = tile.getpalette()
        most_common_color = palette[colors[0][1] * 3:colors[0][1] * 3 + 3]
    else:
        # Otherwise, get the most common color directly
        most_common_color = max(colors, key=lambda x: x[0])[1]
    return most_common_color


def get_adjacent_common_color(main_image, x, y, default_color):
    # Collect colors from adjacent pixels
    adjacent_colors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            # Skip the center pixel itself
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < main_image.width and 0 <= ny < main_image.height:
                adjacent_color = main_image.getpixel((nx, ny))
                if adjacent_color != (0, 0, 0):
                    adjacent_colors.append(adjacent_color)

    # Find the most common adjacent color, excluding the default (to avoid counting 'empty' or 'yet to fill' areas)
    if adjacent_colors:
        most_common = max(set(adjacent_colors), key=adjacent_colors.count)
        # get that color from the palette if the image is palettized
        if main_image.mode == 'P':
            palette = main_image.getpalette()
            most_common = palette[most_common * 3:most_common * 3 + 3]
        return most_common
    else:
        return default_color


def adjust_tile_colors(tile, surrounding_colors):
    """Adjust colors of the tile based on surrounding colors."""
    if not surrounding_colors:
        return tile  # No surrounding colors, no adjustment needed

    # Count occurrences of each color in surrounding tiles
    color_counter = Counter(surrounding_colors)

    # Find the most common color
    most_common_color = color_counter.most_common(1)[0][0]

    # Replace all colors in the tile with the most common color
    adjusted_tile = np.full_like(tile, most_common_color)

    return adjusted_tile


def most_common_border_color(image, x, y, tile_size, default_color):
    """Calculate the most common color in the bordering pixels of a specific tile."""
    border_colors = []

    # Define pixel coordinates for the bordering line
    border_positions = [(x + i, y) for i in range(-1, tile_size + 1)] + [(x + i, y + tile_size - 1) for i in
                                                                         range(-1, tile_size + 1)] \
                       + [(x, y + i) for i in range(tile_size)] + [(x + tile_size - 1, y + i) for i in range(tile_size)]

    # Sample colors from each border pixel, ensuring they are within image bounds
    for bx, by in border_positions:
        if 0 <= bx < image.width and 0 <= by < image.height:
            color = image.getpixel((bx, by))
            if color != (0, 0, 0):  # Skip black or adjust as needed
                border_colors.append(color)

    # Find the most common border color
    if border_colors:
        return max(set(border_colors), key=border_colors.count)
    else:
        return default_color


def tile_similarity_indexed(tile1, tile2):
    # Compare two tiles based on their palette indices
    data1 = tile1.load()
    data2 = tile2.load()
    similar_pixels = 0
    total_pixels = tile1.size[0] * tile1.size[1]

    for y in range(tile1.size[1]):
        for x in range(tile1.size[0]):
            value1 = data1[x, y]
            value2 = data2[x, y]
            if data1[x, y] == data2[x, y]:  # Compare index values instead of colors
                similar_pixels += 1

    return similar_pixels / total_pixels


from PIL import Image


def convert_to_rgb_with_four_colors(image: Image, target_colors: list):
    # Ensure image is in 'P' mode; if not, convert it.
    if image.mode != 'P':
        raise ValueError("Image must be in 'P' mode")

    # Create a new RGB image with the same dimensions as the original
    new_rgb_image = Image.new('RGB', image.size)

    # Get the palette of the original image
    original_palette = image.getpalette()

    # Map each palette index to one of the four colors
    # Assuming 'top_colors' contains RGB tuples of the desired colors
    color_mapping = {index: target_colors[index % len(target_colors)] for index in range(256)}

    # Convert the palette to an RGB image
    for y in range(image.size[1]):  # Iterate over height
        for x in range(image.size[0]):  # Iterate over width
            # Get the palette index of the current pixel
            index = image.getpixel((x, y))
            # Set the new image's pixel to the corresponding color from 'top_colors'
            new_rgb_image.putpixel((x, y), color_mapping[index])

    return new_rgb_image
def calculate_ssim(tile1, tile2):
    """
    Calculate the Structural Similarity Index (SSIM) between two tiles.
    """
    # Convert tiles to grayscale for SSIM calculation
    tile1_gray = np.array(tile1.convert('L'))
    tile2_gray = np.array(tile2.convert('L'))
    
    # Calculate SSIM. Ensure data_range matches the max of the data type.
    score, _ = ssim(tile1_gray, tile2_gray, full=True, data_range=255)
    return score
def normalize_tile(tile):
    """
    Convert a tile to a normalized form based on the frequency of each color,
    disregarding the specific colors themselves. The tile should already be in 'P' mode.
    """
    # In P mode, the data is already flat: a sequence of indices, not rows of pixels
    flat_tile = list(tile.getdata())
    # Count frequency of each color (index) in the tile
    color_counts = Counter(flat_tile)
    # Sort colors by frequency (and then by index value to ensure consistency)
    sorted_colors = sorted(color_counts.keys(), key=lambda color: (-color_counts[color], color))
    # Map each color to its rank (most common = 0, next = 1, etc.)
    color_map = {color: rank for rank, color in enumerate(sorted_colors)}
    # Create a new tile with normalized color values
    normalized_pixels = [color_map[color] for color in flat_tile]
    # Convert back to an Image
    new_tile = Image.new('P', tile.size)
    new_tile.putdata(normalized_pixels)
    return new_tile

def tile_similarity_indexed(tile1, tile2):
    """
    Compare two tiles based on their normalized forms.
    """
    # Convert both tiles to their normalized forms
    norm_tile1 = normalize_tile(tile1)
    norm_tile2 = normalize_tile(tile2)
    
    # Now compare the normalized tiles directly
    data1 = norm_tile1.getdata()
    data2 = norm_tile2.getdata()
    
    # Measure similarity as the percentage of matching pixels
    matches = sum(1 for p1, p2 in zip(data1, data2) if p1 == p2)
    return matches / len(data1)

def map_pattern_to_palette(source_tile, target_tile):
    """
    Maps the color pattern from the source tile to the target tile's palette.
    
    :param source_tile: The source tile (Image object) whose pattern will be used.
    :param target_tile: The target tile (Image object) whose palette will be applied.
    :return: A new tile with the source pattern and target palette.
    """
    # Get the normalized form of the source tile to understand its pattern
    norm_source_tile = normalize_tile(source_tile)
    norm_source_data = list(norm_source_tile.getdata())

    # Get the data from the target tile (these are palette indexes)
    target_data = list(target_tile.getdata())

    # Create a mapping from the normalized source pattern to the target's indexes
    pattern_to_color = {}
    for norm_val, target_val in zip(norm_source_data, target_data):
        if norm_val not in pattern_to_color: # if this pattern not yet mapped, map to current color in target
            pattern_to_color[norm_val] = target_val
            
    # Apply this mapping to create a new tile based on the source pattern but using target's color indexes
    new_tile_data = [pattern_to_color[norm_val] for norm_val in norm_source_data]
    
    # Create a new image for the mapped tile
    new_tile = Image.new('P', target_tile.size)
    new_tile.putpalette(target_tile.getpalette())
    new_tile.putdata(new_tile_data)
    return new_tile

def reduce_tiles_index_palette_aware(image: Image, enhanced_palettes=None, tile_palette_mapping=None, tile_size=8, max_unique_tiles=192, similarity_threshold=0.7, use_tile_variance=False, custom_palette_colors=None):
    """
    Palette-aware tile reduction that preserves palette assignments while enforcing tile limits.
    Only merges tiles that are both visually similar AND use the same palette.
    """
    width, height = image.size
    width -= width % tile_size
    height -= height % tile_size
    image = image.crop((0, 0, width, height))

    # Initialize variables
    tiles = [(x, y, image.crop((x, y, x + tile_size, y + tile_size)))
             for y in range(0, height, tile_size)
             for x in range(0, width, tile_size)]

    # Sort tiles by variance if required
    if use_tile_variance:
        tiles.sort(key=lambda x: tile_variance(x[2]))

    unique_tiles = []  # Store (x, y, tile, palette_id)
    tile_mapping = {}
    new_image = Image.new('RGB', (width, height))

    notice = ''
    tiles_per_row = width // tile_size
    
    for i, (x, y, tile) in enumerate(tiles):
        # Determine palette ID for this tile position
        palette_id = tile_palette_mapping[i] if tile_palette_mapping and i < len(tile_palette_mapping) else 0
        
        best_similarity = -1
        best_match = None
        
        # Only compare with tiles that use the SAME palette
        for unique_x, unique_y, unique_tile, unique_palette_id in unique_tiles:
            if palette_id == unique_palette_id:  # Same palette requirement
                sim = tile_similarity_indexed(tile.convert('P'), unique_tile.convert('P'))
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = (unique_x, unique_y, unique_tile, unique_palette_id)

        if best_similarity > similarity_threshold:
            tile_mapping[(x, y)] = (best_match[0], best_match[1])
        elif len(unique_tiles) < max_unique_tiles:
            unique_tiles.append((x, y, tile, palette_id))
            tile_mapping[(x, y)] = (x, y)
        else:
            # Find best match among tiles with same palette, even if below threshold
            same_palette_tiles = [(ux, uy, ut, up) for ux, uy, ut, up in unique_tiles if up == palette_id]
            if same_palette_tiles:
                best_match = min(same_palette_tiles, key=lambda ut: tile_similarity_indexed(tile.convert('P'), ut[2].convert('P')))
                tile_mapping[(x, y)] = (best_match[0], best_match[1])
            else:
                # Force match with any tile (this shouldn't happen often)
                best_match = min(unique_tiles, key=lambda ut: tile_similarity_indexed(tile.convert('P'), ut[2].convert('P')))
                tile_mapping[(x, y)] = (best_match[0], best_match[1])

    # Paint the new image
    for (x, y), (ux, uy) in tile_mapping.items():
        original_tile = image.crop((ux, uy, ux + tile_size, uy + tile_size))
        new_image.paste(original_tile, (x, y))

    if len(unique_tiles) < max_unique_tiles:
        notice = f"✅ {len(unique_tiles)}/{max_unique_tiles} tiles used. You have {max_unique_tiles - len(unique_tiles)} tiles to spare!"
    else:
        notice = f"⚠️ Hit {max_unique_tiles} tile limit! Try: Lower similarity threshold • Use simpler images • Reduce colors • Increase image compression"

    return new_image, notice


def reduce_tiles_index(image: Image, tile_size=8, max_unique_tiles=192, similarity_threshold=0.7, use_tile_variance=False, custom_palette_colors=None):
    """
    Reduces the number of unique tiles in an image based on similarity.
    """
    width, height = image.size
    width -= width % tile_size
    height -= height % tile_size
    # convert to P mode perfectly with exactly same colour info
    image = image.crop((0, 0, width, height)).convert('P', colors=256, dither=Image.NONE)

    # Initialize variables
    tiles = [(x, y, image.crop((x, y, x + tile_size, y + tile_size)))
             for y in range(0, height, tile_size)
             for x in range(0, width, tile_size)]

    # Sort tiles by variance if required
    if use_tile_variance:
        tiles.sort(key=lambda x: tile_variance(x[2]))

    unique_tiles = []
    tile_mapping = {}
    new_image = Image.new('P', (width, height)) # Use 'P' mode for the new image
    image_palette = image.getpalette() # Get the palette of the original image
    new_image.putpalette(image_palette)  # Apply the same palette to the new image

    notice = ''
    for x, y, tile in tiles:
        best_similarity = -1
        best_match = None
        for unique_x, unique_y, unique_tile in unique_tiles:
            sim = tile_similarity_indexed(tile, unique_tile) # Define or use a suitable function
            if sim > best_similarity:
                best_similarity = sim
                best_match = (unique_x, unique_y, unique_tile)

        if best_similarity > similarity_threshold:
            tile_mapping[(x, y)] = (best_match[0], best_match[1])
        elif len(unique_tiles) < max_unique_tiles:
            unique_tiles.append((x, y, tile))
            tile_mapping[(x, y)] = (x, y)
        else:
            best_match = min(unique_tiles, key=lambda ut: tile_similarity_indexed(tile, ut[2]))
            tile_mapping[(x, y)] = (best_match[0], best_match[1])

    # Paint the new image
    for (x, y), (ux, uy) in tile_mapping.items():
        original_tile = image.crop((x, y, x + tile_size, y + tile_size)) # Get the original tile
        pattern_tile = image.crop((ux, uy, ux + tile_size, uy + tile_size)) # Get the tile to copy the pattern from
        new_tile = map_pattern_to_palette(pattern_tile, original_tile) # Create a new tile with the original palette but new pattern
        new_image.paste(new_tile, (x, y))

    if len(unique_tiles) < max_unique_tiles:
        notice = f"✅ {len(unique_tiles)}/{max_unique_tiles} tiles used. Try raising similarity threshold to save more tiles!"
    else:
        notice = f"⚠️ Hit {max_unique_tiles} tile limit! Try: Lower similarity threshold • Use simpler images • Reduce detail/noise"

    return new_image, notice


def reduce_tiles_index(image: Image, tile_size=8, max_unique_tiles=192, similarity_threshold=0.7, use_tile_variance=False, custom_palette_colors=None):
    width, height = image.size
    width -= width % tile_size
    height -= height % tile_size
    # convert to P mode perfectly with exactly same colour info
    image = image.crop((0, 0, width, height)).convert('P', colors=256, dither=Image.NONE)


    # Initialize variables
    tiles = [(x, y, image.crop((x, y, x + tile_size, y + tile_size)))
             for y in range(0, height, tile_size)
             for x in range(0, width, tile_size)]

    # Sort tiles by variance if required
    if use_tile_variance:
        tiles.sort(key=lambda x: tile_variance(x[2]))

    unique_tiles = []
    tile_mapping = {}
    new_image = Image.new('P', (width, height))  # Use 'P' mode for the new image
    image_palette = image.getpalette()  # Get the palette of the original image
    new_image.putpalette(image_palette)  # Apply the same palette to the new image

    notice = ''
    for x, y, tile in tiles:
        best_similarity = -1
        best_match = None
        for unique_x, unique_y, unique_tile in unique_tiles:
            sim = tile_similarity_indexed(tile, unique_tile)  # Define or use a suitable function
            if sim > best_similarity:
                best_similarity = sim
                best_match = (unique_x, unique_y, unique_tile)

        if best_similarity > similarity_threshold:
            tile_mapping[(x, y)] = (best_match[0], best_match[1])
        elif len(unique_tiles) < max_unique_tiles:
            unique_tiles.append((x, y, tile))
            tile_mapping[(x, y)] = (x, y)
        else:
            best_match = min(unique_tiles, key=lambda ut: tile_similarity_indexed(tile, ut[2]))
            tile_mapping[(x, y)] = (best_match[0], best_match[1])

    # Paint the new image
    for (x, y), (ux, uy) in tile_mapping.items():
        original_tile = image.crop((x, y, x + tile_size, y + tile_size))  # Get the original tile
        pattern_tile = image.crop((ux, uy, ux + tile_size, uy + tile_size))  # Get the tile to copy the pattern from
        new_tile = map_pattern_to_palette(pattern_tile,
                                          original_tile)  # Create a new tile with the original palette but new pattern
        new_image.paste(new_tile, (x, y))

    if len(unique_tiles) < max_unique_tiles:
        notice = f"✅ {len(unique_tiles)}/{max_unique_tiles} tiles used. Try raising similarity threshold to save more tiles!"
    else:
        notice = f"⚠️ Hit {max_unique_tiles} tile limit! Try: Lower similarity threshold • Use simpler images • Reduce detail/noise"

    return new_image, notice


def reduce_tiles(image, tile_size=8, max_unique_tiles=192, similarity_threshold=0.7):
    width, height = image.size
    width -= width % tile_size
    height -= height % tile_size
    image = image.crop((0, 0, width, height))
    notice = ''
    image = image.convert('P')
    # Assuming the image has only four colors
    color_counts = image.getcolors(width * height)  # Get all colors in the image
    top_colors = [color for count, color in color_counts]  # Extract the colors

    # Gather and sort tiles with variance
    tiles = [(x, y, image.crop((x, y, x + tile_size, y + tile_size)))
             for y in range(0, height, tile_size)
             for x in range(0, width, tile_size)]
    if use_tile_variance.value:
        tiles.sort(key=lambda x: tile_variance(x[2]))

    unique_tiles = []  # Store unique tiles
    tile_mapping = {}  # Map from old tiles to new tiles (for merged tiles)
    new_image = Image.new('RGB', (width, height))  # Prepare the new image

    # Add a block for each of the four colors in the image to unique_tiles
    for i, color in enumerate(top_colors):
        color_tile = Image.new('P', (tile_size, tile_size))
        color_tile.putpalette([color])
        # Use special coordinates for color blocks
        unique_tiles.append((i + 1 * -8, i + 1 * -8, color_tile))

    for x, y, tile in tiles:
        # Find the most similar tile in the unique set
        similarity, (idx, (ux, uy, utile)) = max(
            ((tile_similarity(tile, utile), (i, unique_tiles[i])) for i, (ux, uy, utile) in enumerate(unique_tiles)),
            key=lambda x: x[0]
        )
        if similarity > similarity_threshold:
            # Merge similar tiles by referencing the similar tile in the mapping
            tile_mapping[(x, y)] = (ux, uy)
            continue

        elif len(unique_tiles) < max_unique_tiles:
            # Add the tile to the unique set if we have room
            unique_tiles.append((x, y, tile))
            tile_mapping[(x, y)] = (x, y)

            if len(unique_tiles) == max_unique_tiles:
                remaining_tiles = len(tiles) - (x // tile_size + y // tile_size * (width // tile_size))
                notice = (f"**OUT OF SPARE TILES**\n"
                          f"Tiles left to process: {remaining_tiles}\n"
                          f"Consider reducing Tile Similarity Threshold")

        else:

            # Initialize a variable to store the best match index and its similarity score
            best_match_index = -1
            best_similarity = -1  # Start with -1 to ensure any real similarity will be higher

            # Iterate through all unique tiles to find the most similar one
            for i, (ux, uy, utile) in enumerate(unique_tiles):
                # Calculate the similarity between the current tile and this unique tile
                current_similarity = tile_similarity(tile, utile)

                # Update the best match if this tile is more similar than previous ones
                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    best_match_index = i

            # After finding the most similar tile, retrieve its information
            if best_match_index != -1:  # Check that we found a match
                ux, uy, utile = unique_tiles[best_match_index]
                tile_mapping[(x, y)] = (ux, uy)
            else:
                # This else block should ideally never be hit since we always have unique tiles,
                # but it's good practice to handle this case.
                # Fallback: Use the first unique tile or handle this error appropriately.
                ux, uy, utile = unique_tiles[0]  # Default to the first unique tile
                tile_mapping[(x, y)] = (ux, uy)

    # Paint the new image
    for (x, y), (ux, uy) in tile_mapping.items():
        tile = image.crop((ux, uy, ux + tile_size, uy + tile_size))
        new_image.paste(tile, (x, y))  # Directly pasting the tile without color adjustment

    if not notice:
        notice = f"✅ {len(unique_tiles)}/{max_unique_tiles} tiles used. Try raising similarity threshold to save more tiles!"
    return new_image, notice if notice else None


def downscale_image(image: Image, new_width: int, new_height: int, keep_aspect_ratio: bool) -> Image:
    if keep_aspect_ratio:
        old_width, old_height = image.size
        aspect_ratio = old_width / old_height
        if new_width / new_height > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)
    return image.resize((new_width, new_height), Image.NEAREST)


def limit_colors(image, limit=16, quantize=None, dither=None, palette_image=None):
    if palette_image:
        ppalette = palette_image.getcolors()
        color_palette = palette_image.quantize(colors=len(list(set(ppalette))))
    else:
        color_palette = image.quantize(colors=limit, kmeans=limit if limit else 0, method=quantize,
                                       dither=dither)
    image = image.quantize(palette=color_palette, dither=dither)
    return image


def create_palette_from_colors(color_list):
    # Create an empty image with size (1, len(color_list))
    palette_image = Image.new("RGB", (1, len(color_list)))

    # Iterate over the colors and set each pixel in the corresponding row
    for i, color in enumerate(color_list):
        palette_image.putpixel((0, i), color)

    # Convert the image to the palette mode
    palette_image = palette_image.convert("P", palette=Image.ADAPTIVE)

    return palette_image


def convert_to_grayscale(image):
    return image.convert("L").convert("RGB")


def convert_to_black_and_white(image: Image, threshold: int = 128, is_inversed: bool = False):
    apply_threshold = lambda x: 0 if x > threshold else 255 if is_inversed else 255 if x > threshold else 0
    return image.convert('L').point(apply_threshold, mode='1').convert("RGB")


# Gradio UI and processing function
original_width = gr.State(value=0)
original_height = gr.State(value=0)
palette_color_1_string = gr.State(value="#000000")
palette_color_2_string = gr.State(value="#000000")
palette_color_3_string = gr.State(value="#000000")
palette_color_4_string = gr.State(value="#000000")
quantize_for_GBC = gr.State(False)
use_tile_variance = gr.State(False)


def capture_original_dimensions(image):
    # Update global variables with the dimensions of the uploaded image
    width, height = image.size
    return width, height, image  # Return original dimensions and the unchanged image for further processing


def adjust_for_aspect_ratio(keep_aspect, current_width, current_height):
    if keep_aspect and original_width.value and original_height.value:
        # Using the global variables for original dimensions
        aspect_ratio = original_width.value / original_height.value
        # Calculate the new height based on the new width while maintaining the original aspect ratio
        new_height = int(current_width / aspect_ratio)
        return current_width, new_height
    else:
        return current_width, current_height


def create_gradio_interface():
    header = '<script async defer data-website-id="f5b8324e-09b2-4d56-8c1f-40a1f1457023" src="https://metrics.prodigle.dev/umami.js"></script><script type="module" data-entity="gameboy-image-converter" src="https://analytics.prodigle.dev/script.js"></script>'
    with gr.Blocks(head=header) as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Input Image")
                    folder_input = gr.File(label="Input Folder", file_count='directory')
                with gr.Row():
                    new_width = gr.Number(label="Width", value=160)
                    new_height = gr.Number(label="Height", value=144)
                    keep_aspect_ratio = gr.Checkbox(label="Keep Aspect Ratio", value=False)
                with gr.Row():
                    logo_resolution = gr.Button("Use Logo Resolution")
                    original_resolution = gr.Button("Use Original Resolution(Image)")
                with gr.Row():
                    enable_color_limit = gr.Checkbox(label="Limit number of Colors", value=True)
                    number_of_colors = gr.Slider(label="Target Number of colors (32 max for GB Studio)", minimum=2, maximum=64, step=1, value=4)
                    limit_4_colors_per_tile = gr.Checkbox(label="Limit to 4 colors per tile, 8 palettes (For GB Studio development only)",
                                                          value=False, visible=True)
                with gr.Group():
                    with gr.Row():
                        reduce_tile_checkbox = gr.Checkbox(label="Reduce to 192 unique 8x8 tiles (Not needed for LOGO scene mode)", value=False)
                        use_tile_variance_checkbox = gr.Checkbox(label="Sort by tile complexity (Complex tiles get saved first)", value=False)
                    reduce_tile_similarity_threshold = gr.Slider(label="Tile similarity threshold", minimum=0.3,
                                                                 maximum=0.99, value=0.8, step=0.01, visible=False)
                with gr.Row():
                    quantization_method = gr.Dropdown(choices=list(QUANTIZATION_METHODS.keys()),
                                                      label="Quantization Method", value="libimagequant")
                    dither_method = gr.Dropdown(choices=list(DITHER_METHODS.keys()), label="Dither Method",
                                                value="None")
                with gr.Group():
                    use_custom_palette = gr.Checkbox(label="Use Custom Color Palette", value=True)
                    palette_image = gr.Image(label="Color Palette Image", type="pil", visible=True,
                                             value=os.path.join(os.path.dirname(__file__), "gb_palette.png"))
                with gr.Group():
                    gr.Markdown("### Gothic Filter")
                    enable_gothic_filter = gr.Checkbox(label="Enable Gothic Filter", value=False)
                    brightness_threshold = gr.Slider(label="Brightness Threshold", minimum=0, maximum=255, value=0,
                                                     step=1)
                    dot_size = gr.Slider(label="Dot Size", minimum=0.25, maximum=6, value=1, step=0.25)
                    spacing = gr.Slider(label="Spacing", minimum=0, maximum=10, value=1, step=1)
                    contrast_boost = gr.Slider(label="Contrast Boost", minimum=1.0, maximum=2.0, value=1.5, step=0.1)
                    noise_factor = gr.Slider(label="Noise Factor", minimum=0, maximum=1, value=0.5, step=0.05)
                    edge_enhance = gr.Checkbox(label="Edge Enhancement", value=False)
                    apply_blur = gr.Checkbox(label="Apply Blur", value=False)
                    irregular_shape = gr.Checkbox(label="Irregular Dot Shape", value=False)
                    irregular_size = gr.Checkbox(label="Irregular Dot Size", value=False)
                is_grayscale = gr.Checkbox(label="Convert to Grayscale", value=False)
                with gr.Row():
                    is_black_and_white = gr.Checkbox(label="Convert to Black and White", value=False)
                    black_and_white_threshold = gr.Slider(label="Black and White Threshold", minimum=0, maximum=255,
                                                          value=128, visible=False)


                is_black_and_white.change(lambda x: gr.update('black_and_white_threshold', visible=x),
                                          inputs=[is_black_and_white], outputs=[black_and_white_threshold])

                # Logic to capture and display original image dimensions
                def capture_original_dimensions(image):
                    # Update the global variables with the dimensions of the uploaded image
                    if image is None:
                        return None
                    width, height = image.size
                    original_width.value = width
                    original_height.value = height
                    return image  # Return unchanged image for further processing

                def limit_4_colors_per_tile_change(x):
                    quantize_for_GBC.value = x
                    return quantize_for_GBC.value


                limit_4_colors_per_tile.change(limit_4_colors_per_tile_change, inputs=[limit_4_colors_per_tile])

                def on_use_tile_variance_click(x):
                    use_tile_variance.value = x
                    return x

                use_tile_variance_checkbox.change(on_use_tile_variance_click, inputs=[use_tile_variance_checkbox])

                image_input.change(
                    fn=capture_original_dimensions,
                    inputs=[image_input],
                    outputs=[image_input]
                )

                def on_logo_resolution_click():
                    # Return the values you want to update in the UI components
                    # No need to call .update() on individual components here, just return the new values
                    return False, 160, 144

                # In the logo_resolution button click setup,
                # Ensure you're mapping the outputs of the function to the correct UI elements
                logo_resolution.click(
                    fn=on_logo_resolution_click,
                    outputs=[keep_aspect_ratio, new_width, new_height]
                    # The outputs should correspond to the UI components you want to update
                )

                def on_original_resolution_click():
                    return False, original_width.value, original_height.value

                original_resolution.click(
                    fn=on_original_resolution_click,
                    outputs=[keep_aspect_ratio, new_width, new_height]
                )

                # Dynamic updates based on aspect ratio checkbox and width changes
                keep_aspect_ratio.change(
                    fn=adjust_for_aspect_ratio,
                    inputs=[keep_aspect_ratio, new_width, new_height],
                    outputs=[new_width, new_height]
                )
                new_width.change(
                    fn=adjust_for_aspect_ratio,
                    inputs=[keep_aspect_ratio, new_width, new_height],
                    outputs=[new_width, new_height]
                )

            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            image_output = gr.Image(type="pil", label="Output Image", height=300)
                        with gr.Column():
                            image_output_no_palette = gr.Image(type="pil", label="Output Image (Natural Palette)",
                                          height=300)
                    with gr.Row():
                        with gr.Column():
                            notice_text = gr.Text(value="No Warnings", lines=3, max_lines=3, autoscroll=False, interactive=False, label="Warnings", show_label=False)
                        with gr.Column():
                            kofi_html = gr.HTML(
                                "<a href='https://ko-fi.com/prodigle' target='_blank'><img height='36' style='border:0px; margin:auto; padding: 5px; width: 100%' src='https://cdn.ko-fi.com/cdn/kofi1.png?v=2' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>")
                    with gr.Row():
                        with gr.Column():
                            palette_text = gr.Textbox(label="Custom Palette Info", value="None", interactive=False,
                                                  show_copy_button=True, lines=4, max_lines=4, autoscroll=False)
                with gr.Row():
                    execute_button = gr.Button("Convert Image")
                    execute_button_folder = gr.Button("Convert Folder")
                image_output_zip = gr.File(label="Output Folder Zip", type="filepath")

        reduce_tile_checkbox.change(lambda x: gr.update('reduce_tile_checkbox', visible=x),
                                    inputs=[reduce_tile_checkbox], outputs=[reduce_tile_similarity_threshold])

        use_custom_palette.change(lambda x: gr.update('palette_image', visible=x),
                                  inputs=[use_custom_palette], outputs=[palette_image])

        def extract_tiles(image, tile_size=(8, 8)):
            """Extract 8x8 tiles from the image."""
            tiles = []
            for y in range(0, image.height, tile_size[1]):
                for x in range(0, image.width, tile_size[0]):
                    box = (x, y, x + tile_size[0], y + tile_size[1])
                    tiles.append(image.crop(box))
            return tiles

        def generate_palette(tile, num_colors=4):
            """Generate a 4-color palette for an 8x8 tile using K-means clustering.

            Args:
                tile: The input tile as a PIL image or a NumPy array.
                num_colors: The number of colors to include in the palette.

            Returns:
                A NumPy array representing the palette, with each color as a row.
            """
            # Ensure the input is a NumPy array and has the correct shape
            if not isinstance(tile, np.ndarray):
                tile = np.array(tile)
            if tile.shape[0] * tile.shape[1] < num_colors:
                raise ValueError("Tile is too small for the number of colors requested")

            # Reshape the tile data for K-means clustering
            data = tile.reshape((-1, 3))

            # Perform K-means clustering to find the dominant colors
            kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(data)
            palette = kmeans.cluster_centers_.round().astype(
                int)  # Round before converting to int for more accurate colors


            return palette

        def get_color_distribution(tile):
            """
            Analyzes the tile and returns a frequency distribution of its colors.
            """
            # Flatten the tile to a list of colors
            data = tile.reshape(-1, 3)
            # Count the frequency of each color
            colors, counts = np.unique(data, axis=0, return_counts=True)
            # Create a normalized distribution (frequency)
            total = counts.sum()
            distribution = {tuple(color): count / total for color, count in zip(colors, counts)}
            return distribution

        import numpy as np
        from scipy.stats import wasserstein_distance

        import numpy as np
        from scipy.stats import wasserstein_distance
        from sklearn.metrics import pairwise_distances

        import numpy as np
        from sklearn.metrics import pairwise_distances

        def find_best_matching_palette(tile_distribution, existing_palettes, adjacent_distributions=None,
                                       balance_factor=0.5, key_color_weight=1):
            """
            Enhanced palette matching considering color distribution, adjacent tiles, balance between local and global harmony, and key colors.
            Args:
                tile_distribution (dict): Color distribution of the current tile.
                existing_palettes (list): List of available palettes to choose from.
                adjacent_distributions (list): List of color distributions from adjacent tiles.
                balance_factor (float): Balances between matching the tile's own colors and blending with adjacent tiles.
                key_color_weight (float): Additional weight given to key colors to ensure their presence in the selected palette.
            Returns:
                int: Index of the best matching palette.
            """
            best_score = float('inf')
            best_palette_index = None
            # Define key colors that need special attention (e.g., white, black, skin tones)
            key_colors = [(255, 255, 255), (0, 0, 0)]  # White and black

            # Iterate through each candidate palette
            for palette_index, palette in enumerate(existing_palettes):
                palette_colors = np.array(palette)
                tile_score = 0

                # Calculate how well the palette matches the tile's own color distribution
                for color, frequency in tile_distribution.items():
                    distances = pairwise_distances([color], palette_colors, metric='euclidean')[0]
                    closest_distance = np.min(distances)
                    # Apply additional weight to key colors
                    weight = key_color_weight if color in key_colors else 1
                    tile_score += frequency * closest_distance * weight

                # Context-aware selection: consider adjacent tiles if available
                context_score = 0
                if adjacent_distributions:
                    for adj_dist in adjacent_distributions:
                        adj_score = 0
                        for adj_color, adj_freq in adj_dist.items():
                            distances = pairwise_distances([adj_color], palette_colors, metric='euclidean')[0]
                            closest_distance = np.min(distances)
                            # Apply additional weight to key colors in adjacent tiles
                            weight = key_color_weight if adj_color in key_colors else 1
                            adj_score += adj_freq * closest_distance * weight
                        context_score += adj_score  # Accumulate context score from all adjacent tiles

                    # Average the context score based on the number of adjacent tiles considered
                    context_score /= len(adjacent_distributions)

                # Combine tile score and context score using the balance factor
                combined_score = (1 - balance_factor) * tile_score + balance_factor * context_score

                # Update best palette if current one is better
                if combined_score < best_score:
                    best_score = combined_score
                    best_palette_index = palette_index

            return best_palette_index

        import numpy as np
        from PIL import Image

        from PIL import Image
        import numpy as np

        from skimage.color import deltaE_ciede2000
        from skimage.color import rgb2lab, lab2rgb
        from skimage import color  # Import the color module from scikit-image
        def apply_palette(tile, palette):
            """
            Applies a palette to a tile using vectorized operations for efficiency.
            """
            tile_array = np.array(tile)
            palette_array = np.array(palette)

            # Convert tile and palette to LAB color space
            tile_lab = rgb2lab(tile_array)
            palette_lab = rgb2lab(palette_array[np.newaxis, :, :])

            # Expand dimensions for broadcasting
            tile_lab_expanded = tile_lab[:, :, np.newaxis, :]

            # Calculate color distances in a vectorized manner
            distances = np.linalg.norm(tile_lab_expanded - palette_lab, axis=3)

            # Find the index of the closest palette color for each pixel
            closest_palette_indices = np.argmin(distances, axis=2)

            # Map the tile to the new palette using advanced indexing
            new_tile_array = palette_array[closest_palette_indices]

            # Convert back to PIL Image
            new_tile = Image.fromarray(np.uint8(new_tile_array), 'RGB')
            return new_tile

        from skimage.color import rgb2lab, lab2rgb
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
        import random
        from skimage.color import lab2rgb
        from sklearn.metrics.pairwise import euclidean_distances

        import itertools

        from skimage.color import lab2rgb, rgb2lab
        import numpy as np
        from PIL import Image

        import numpy as np
        from skimage.color import lab2rgb, rgb2lab

        import numpy as np
        from skimage.color import lab2rgb, rgb2lab
        from PIL import Image

        def create_refined_palettes(cluster_centers, tiles, num_palettes=8, colors_per_palette=4):
            # Flatten and convert LAB color arrays to RGB, then clip and convert to integers.
            flat_cluster_centers = np.vstack(cluster_centers)
            cluster_centers_rgb = (np.clip(lab2rgb(flat_cluster_centers), 0, 1) * 255).astype(np.uint8)

            # Initialize palettes and a list to keep track of color frequencies.
            refined_palettes_rgb = [[] for _ in range(num_palettes)]
            color_frequencies = np.zeros(len(cluster_centers_rgb), dtype=int)

            # Calculate color frequencies based on their presence in tiles.
            for tile in tiles:
                lab_tile = rgb2lab(np.array(tile.convert('RGB'), dtype=np.float64) / 255).reshape(-1, 3)
                for color in lab_tile:
                    distances = np.linalg.norm(cluster_centers_rgb - color, axis=1)
                    nearest_color_index = np.argmin(distances)
                    color_frequencies[nearest_color_index] += 1

            # Rank colors by frequency and distribute among palettes.
            sorted_indices = np.argsort(-color_frequencies)
            palette_counters = [0] * num_palettes
            for color_index in sorted_indices:
                least_filled_palette_index = palette_counters.index(min(palette_counters))
                if palette_counters[least_filled_palette_index] < colors_per_palette:
                    current_color = tuple(cluster_centers_rgb[color_index])
                    if all(current_color != tuple(color) for color in refined_palettes_rgb[least_filled_palette_index]):
                        refined_palettes_rgb[least_filled_palette_index].append(cluster_centers_rgb[color_index])
                        palette_counters[least_filled_palette_index] += 1

                if all(count == colors_per_palette for count in palette_counters):
                    break

            # Define function to calculate color distance outside of any loop
            def color_distance(c1, c2):
                return np.sqrt(np.sum((c1 - c2) ** 2))

            # Calculate global color usage outside of the initial distribution loop
            global_color_usage = {tuple(color): 0 for color in cluster_centers_rgb}
            for palette in refined_palettes_rgb:
                for color in palette:
                    global_color_usage[tuple(color)] += 1

            # Fill up any palettes that are short of colors, outside of the initial distribution loop
            for palette_index, palette in enumerate(refined_palettes_rgb):
                while len(palette) < colors_per_palette:
                    best_color = None
                    best_color_score = -np.inf  # Lower score is better; start with worst possible
                    existing_colors_tuples = [tuple(color) for color in palette]

                    for color in cluster_centers_rgb:
                        color_tuple = tuple(color)  # Ensure color is a tuple for comparison
                        if color_tuple not in existing_colors_tuples:  # Corrected comparison
                            # Calculate color's overall score based on global usage and diversity
                            usage_score = -global_color_usage[color_tuple]  # Prefer less used colors
                            diversity_score = min(
                                [color_distance(np.array(color), np.array(existing_color)) for existing_color in
                                 palette] or [np.inf])  # Prefer colors different from existing ones
                            total_score = usage_score + diversity_score

                            if total_score > best_color_score:
                                best_color_score = total_score
                                best_color = color

                    if best_color is not None:
                        refined_palettes_rgb[palette_index].append(best_color)
                        global_color_usage[tuple(best_color)] += 1

            return refined_palettes_rgb

        from skimage.color import lab2rgb, rgb2lab
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
        from PIL import Image

        def analyze_and_construct_palettes(tiles, max_palettes=8, max_colors=32, local_influence=0.5):
            # Extract unique colors from all tiles
            unique_colors_set = set()
            for tile in tiles:
                rgb_tile = np.array(tile.convert('RGB'), dtype=np.uint8)  # Convert each PIL Image tile to a NumPy array
                unique_colors = set(tuple(color) for row in rgb_tile for color in row)
                unique_colors_set.update(unique_colors)

            # Determine the actual number of clusters based on unique colors in the image
            num_unique_colors = len(unique_colors_set)
            num_clusters = min(max_colors, num_unique_colors)  # Adjust number of clusters based on unique colors

            # Convert PIL Images to LAB and perform clustering as before
            lab_tiles = [rgb2lab(np.array(tile.convert('RGB'), dtype=np.float64) / 255) for tile in tiles]
            all_tiles_lab = np.vstack([tile.reshape(-1, 3) for tile in lab_tiles])

            global_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
            global_kmeans.fit(all_tiles_lab)
            # The rest of your function continues as before...

            broad_palette_lab = global_kmeans.cluster_centers_

            # Compute global color weights
            global_labels = global_kmeans.labels_
            global_color_weights = np.bincount(global_labels, minlength=num_clusters) / float(len(global_labels))

            # Adjust global color weights based on local tile colors
            for tile_lab in lab_tiles:
                local_labels = global_kmeans.predict(tile_lab.reshape(-1, 3))
                local_color_weights = np.bincount(local_labels, minlength=num_clusters) / float(len(local_labels))
                global_color_weights = (
                                                   1 - local_influence) * global_color_weights + local_influence * local_color_weights

            # Fallback for zero weights
            if not np.any(global_color_weights):
                global_color_weights = np.ones_like(global_color_weights) / len(global_color_weights)

            # Apply weights to determine final color selection
            weighted_colors = np.repeat(broad_palette_lab, np.maximum(global_color_weights.astype(int), 1), axis=0)

            # Ensure there are weighted colors to fit
            if weighted_colors.size == 0:
                weighted_colors = broad_palette_lab

            # Perform final KMeans clustering to determine refined palettes
            # Cap colors at max_palettes * 4 to ensure GB Studio compatibility (8 palettes × 4 colors = 32 max)
            final_kmeans = MiniBatchKMeans(n_clusters=min(max_colors, max_palettes * 4), random_state=42)
            final_kmeans.fit(weighted_colors)

            # Construct refined palettes
            refined_palettes_lab = [final_kmeans.cluster_centers_[i:i + 4] for i in
                                    range(0, len(final_kmeans.cluster_centers_), 4)]
            refined_palettes_rgb = create_refined_palettes(refined_palettes_lab, tiles)

            return refined_palettes_rgb

        def process_tiles(tiles, max_palettes=8, tile_width=8, tile_height=8, image_width=None, num_colors=4, should_dither=False, enhanced_palettes=None):
            """Process the tiles to limit them to the best 4-color palettes, based on global analysis and frequency."""
            # Step 1: Generate enhanced palettes considering the whole image
            tile_palette_mapping = []
            if not enhanced_palettes:
                enhanced_palettes = analyze_and_construct_palettes(tiles, max_palettes, num_colors)

            # Calculate the number of tiles per row if image width is known
            tiles_per_row = image_width // tile_width if image_width else None

            # Initialize list to hold the best palette for each tile
            tile_palettes = []
            palette_for_tile_text = ""
            # Step 2: Assign each tile the most suitable enhanced palette, considering adjacent tiles
            for index, tile in enumerate(tiles):
                closest_palette_index = None
                np_tile = np.array(tile)
                tile_distribution = get_color_distribution(np_tile)

                # Gather distributions from adjacent tiles if possible
                adjacent_distributions = []
                if tiles_per_row:  # If the layout of tiles is known
                    # Determine the positions of adjacent tiles
                    positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (dy, dx) pairs for up, down, left, right
                    for dy, dx in positions:
                        adj_index = index + dy * tiles_per_row + dx
                        # Check if adjacent index is valid and within the same row/column as appropriate
                        if 0 <= adj_index < len(tiles) and (
                                dy == 0 or (index // tiles_per_row == adj_index // tiles_per_row)):
                            adjacent_tile = tiles[adj_index]
                            adj_distribution = get_color_distribution(np.array(adjacent_tile))
                            adjacent_distributions.append(adj_distribution)

                # Find the closest enhanced palette to the original tile palette, considering adjacent tiles
                closest_palette_index = find_best_matching_palette(tile_distribution, enhanced_palettes,
                                                                  adjacent_distributions)

                tile_palettes.append(closest_palette_index)
                x,y = index % tiles_per_row, index // tiles_per_row
                palette_for_tile_text += f"Tile at ({x},{y}) uses palette {closest_palette_index + 1}\n"
                tile_palette_mapping.append(closest_palette_index)

            # Step 3: Apply the selected palettes to each tile
            processed_tiles = [apply_palette(tiles[i], enhanced_palettes[tile_palettes[i]]) for i in range(len(tiles))]

            return processed_tiles, enhanced_palettes, palette_for_tile_text, tile_palette_mapping

        from PIL import Image

        def replace_tile_palettes(img, custom_palette):
            """
            Apply a custom palette to each 8x8 tile in the image.
            Each 8x8 tile will map its original color indexes directly to the custom palette indexes.

            :param img: PIL Image to modify.
            :param custom_palette: List of tuples, each representing an RGB color.
            :return: PIL Image with the modified palette.
            """
            # Ensure the custom palette has exactly 4 colors
            if len(custom_palette) != 4:
                raise ValueError("Custom palette must have exactly 4 colors")

            # Make into a PIL Image palette
            custom_palette_image = create_palette_from_colors(custom_palette)

            # Process each 8x8 tile
            for y in range(0, img.height, 8):
                for x in range(0, img.width, 8):
                    tile = img.crop((x, y, x + 8, y + 8))
                    tile = tile.convert('P')  # Convert to palette mode
                    tile_data = list(tile.getdata())  # Convert data to list if it's not already
                    new_tile = Image.new('P', (8, 8))
                    new_tile.putpalette(custom_palette_image.getpalette())
                    new_tile.putdata(tile_data)  # Use the data
                    new_tile = new_tile.convert('RGB')  # Convert back to RGB

                    # Paste the modified tile back into the image
                    img.paste(new_tile, (x, y))

            return img

        def apply_mapped_colors_to_tile(tile, tile_palette, custom_palette_palette):
            if tile.mode != "RGB":
                tile = tile.convert("RGB", palette=Image.ADAPTIVE, dither=False)
            # Create a new tile to avoid changing the original while iterating
            new_tile = Image.new('RGB', tile.size)
            pixels = tile.load()  # Load tile pixels for reading
            new_pixels = new_tile.load()  # Load new tile pixels for writing

            # Convert tile_palette to a list of tuples if it's not already, to avoid the ambiguous truth value error
            tile_palette_tuples = [tuple(color) for color in tile_palette]
            tile_palette_tuples = tuple(tile_palette_tuples)

            for y in range(tile.size[1]):  # For each row in the tile
                for x in range(tile.size[0]):  # For each column in the row
                    original_color = pixels[x, y]
                    original_color = tuple(original_color)
                    if original_color in tile_palette_tuples:  # If the color is in the tile's palette
                        # Find the index of the color in the original tile's palette
                        color_index = tile_palette_tuples.index(original_color)
                        # Find the new color from the custom palette using the same index
                        new_color = custom_palette_palette[color_index]
                        # Set the pixel in the new tile to the new color
                        new_pixels[x, y] = new_color
                    else:
                        # If the color is not in the palette, keep it as is (or handle as needed)
                        print(f"Color {original_color} not in palette")
                        new_pixels[x, y] = original_color

            return new_tile
        def process_image(image, width, height, aspect_ratio, color_limit, num_colors, quant_method, dither_method,
                          use_palette, custom_palette, grayscale, black_and_white, bw_threshold, reduce_tile_flag,
                          reduce_tile_threshold, limit_4_colors_per_tile, enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                          noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size):
            with task_log("process_image") as task_id:
                if num_colors <= 4:
                    limit_4_colors_per_tile = False
                text_for_palette = ""
                text_for_palette_tile_application = ""
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    image = downscale_image(image, int(width), int(height), aspect_ratio)
                    notice = None
                    image_for_reference_palette = None
                    base_image: Image = image.copy()
                    if color_limit:
                    quant_method_key = quant_method if quant_method in QUANTIZATION_METHODS else 'Median cut'
                    dither_method_key = dither_method if dither_method in DITHER_METHODS else 'None'

                    image_for_reference_palette: Image = image.copy()
                    image_for_reference_palette = limit_colors(image_for_reference_palette, limit=num_colors,
                                                               quantize=QUANTIZATION_METHODS[quant_method_key],
                                                               dither=DITHER_METHODS[dither_method_key])
                    image_for_reference_palette: Image = image_for_reference_palette.convert('RGB')

                palette_color_values = []
                enhanced_palettes = None
                tile_palette_mapping = None
                if limit_4_colors_per_tile and not reduce_tile_flag:
                    image_for_reference_palette: Image = image.copy()
                    tiles = extract_tiles(image_for_reference_palette)
                    processed_tiles, enhanced_palettes, text_for_palette_tile_application, tile_palette_mapping = process_tiles(tiles,8,8,8, image_for_reference_palette.width, num_colors, True if dither_method_key != "None" else False)
                    # Reconstruct the image from the processed tiles
                    new_image = Image.new('RGB', image_for_reference_palette.size)
                    tile_index = 0
                    for y in range(0, image_for_reference_palette.height, 8):
                        for x in range(0, image_for_reference_palette.width, 8):
                            new_image.paste(processed_tiles[tile_index], (x, y))
                            tile_index += 1
                    image_for_reference_palette = new_image
                    for palette in enhanced_palettes:
                        palette_color_values.append(
                            [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in palette])
                else:
                    palette_colors = image_for_reference_palette.getcolors(maxcolors=num_colors)
                    palette_colors = [color for count, color in palette_colors]
                    palette_color_values = [
                        "#{0:02x}{1:02x}{2:02x}".format(*color) for color in palette_colors
                    ]

                if use_palette and custom_palette is not None:
                    if quantize_for_GBC and quantize_for_GBC.value == True and not reduce_tile_flag:
                        if limit_4_colors_per_tile:
                            image = image_for_reference_palette.copy()
                            if not enhanced_palettes or not tile_palette_mapping:
                                processed_tiles, enhanced_palettes, text_for_palette_tile_application, tile_palette_mapping = process_tiles(extract_tiles(image),8,8,8, image_for_reference_palette.width, num_colors, True if dither_method_key != "None" else False, False)
                            custom_palette = custom_palette.quantize(colors=num_colors,
                                                                     method=QUANTIZATION_METHODS[quant_method_key],
                                                                     dither=DITHER_METHODS[dither_method_key])
                            if image.mode != "P":
                                image = image.convert("P", palette=Image.ADAPTIVE, dither=False if dither_method_key == "None" else True)
                            image_tiles = processed_tiles
                            # Ensure the custom_palette_palette is a list of RGB tuples
                            custom_palette_palette = custom_palette.getpalette()
                            custom_palette_palette = [(r, g, b) for r, g, b in
                                                      zip(custom_palette_palette[0::3], custom_palette_palette[1::3],
                                                          custom_palette_palette[2::3])]

                            mapped_image = Image.new('RGB', image.size)
                            for index, tile in enumerate(image_tiles):
                                tile_palette = enhanced_palettes[tile_palette_mapping[index]]
                                # No change needed here to mapped_colors because we're doing pixel-wise color replacement now
                                # Apply new colors to the tile based on the mapping
                                recolored_tile = apply_mapped_colors_to_tile(tile, tile_palette, custom_palette_palette)
                                # Paste the recolored tile back into the image
                                mapped_image.paste(recolored_tile,
                                            (index % (image.width // 8) * 8, index // (image.width // 8) * 8))
                            image = mapped_image
                        else:
                            image = image_for_reference_palette.copy()
                            image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                                             dither=DITHER_METHODS[dither_method_key], palette_image=custom_palette)
                            image

                    else:
                        image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                                             dither=DITHER_METHODS[dither_method_key], palette_image=custom_palette)
                else:
                    image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                                         dither=DITHER_METHODS[dither_method_key])
                if reduce_tile_flag and limit_4_colors_per_tile:
                    #image = image_for_reference_palette.copy()
                    # Apply our custom_palette to the image, without quantising, just the palette
                    image = image_for_reference_palette.copy()
                    image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                                         dither=DITHER_METHODS[dither_method_key])
                    custom_palette_info = custom_palette.getcolors()
                    # just keep the tuple colours
                    for i in range(len(custom_palette_info)):
                        custom_palette_info[i] = custom_palette_info[i][1]
                    enhanced_palettes = analyze_and_construct_palettes(extract_tiles(image), max_palettes=8, max_colors=min(num_colors, 32))
                    # REMOVED: Early tile reduction - moved to after all palette processing
                    # image, notice = reduce_tiles_index(image, similarity_threshold=reduce_tile_threshold, custom_palette_colors=custom_palette_info)
                    image = image.convert("RGB")
                    tiles = extract_tiles(image)
                    processed_tiles, enhanced_palettes, text_for_palette_tile_application, tile_palette_mapping = process_tiles(tiles,8,8,8, image_for_reference_palette.width, len(image.getcolors()), enhanced_palettes=enhanced_palettes)

                    if use_custom_palette:
                        image_tiles = processed_tiles
                        mapped_image = Image.new('RGB', image.size)
                        for index, tile in enumerate(image_tiles):
                            # add to reference image
                            image_for_reference_palette.paste(tile, (index % (image.width // 8) * 8, index // (image.width // 8) * 8))
                            tile_palette = enhanced_palettes[tile_palette_mapping[index]]
                            # No change needed here to mapped_colors because we're doing pixel-wise color replacement now
                            # Apply new colors to the tile based on the mapping
                            recolored_tile = apply_mapped_colors_to_tile(tile, tile_palette, custom_palette_info)
                            # Paste the recolored tile back into the image
                            mapped_image.paste(recolored_tile,
                                               (index % (image.width // 8) * 8, index // (image.width // 8) * 8))
                        image = mapped_image
                        palette_color_values = []
                        for palette in enhanced_palettes:
                            palette_color_values.append(
                                [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in palette])
                    
                    # MOVED: Apply palette-aware tile reduction AFTER all palette processing is complete
                    # This prevents palette changes from creating new tiles that exceed the limit
                    # Uses palette awareness to preserve color assignments while enforcing tile limits
                    image, notice = reduce_tiles_index_palette_aware(image, enhanced_palettes=enhanced_palettes, 
                                                                   tile_palette_mapping=tile_palette_mapping,
                                                                   similarity_threshold=reduce_tile_threshold, 
                                                                   custom_palette_colors=custom_palette_info)
                    image = image.convert("RGB")
                elif reduce_tile_flag:
                    image, notice = reduce_tiles(image, similarity_threshold=reduce_tile_threshold)
                    image_for_reference_palette, notice = reduce_tiles_index(image_for_reference_palette, similarity_threshold=reduce_tile_threshold)

                if enable_gothic_filter:
                    # Apply the gothic filter after all other processing
                    image = apply_gothic_filter(image, brightness_threshold, dot_size, spacing, contrast_boost,
                                                edge_enhance, noise_factor, apply_blur, irregular_shape, irregular_size)
                    image_for_reference_palette = apply_gothic_filter(image_for_reference_palette, brightness_threshold, dot_size, spacing, contrast_boost,
                                                                      edge_enhance, noise_factor, apply_blur, irregular_shape, irregular_size)

                # Return all necessary components including the processed image and color values
                # set pallete_color_values to exactly 4 values nomatter if there's less or more
                for i in range(len(palette_color_values)):
                    text_for_palette += f"Palette {i + 1}: {palette_color_values[i]}\n"
                text_for_palette += f"\n\n{text_for_palette_tile_application}"

                if image.mode != "RGB":
                    image = image.convert("RGB")
                if image_for_reference_palette.mode != "RGB":
                    image_for_reference_palette = image_for_reference_palette.convert("RGB")

                return image, text_for_palette, image_for_reference_palette, notice

            if grayscale:
                image = convert_to_grayscale(image)
            if black_and_white:
                image = convert_to_black_and_white(image, threshold=bw_threshold)
            if reduce_tile_flag:
                image, notice = reduce_tiles(image, similarity_threshold=reduce_tile_threshold)
            return (
                image,
                text_for_palette, None,
                None
            )

        def process_image_folder(input_files, width, height, aspect_ratio, color_limit, num_colors, quant_method, dither_method,
                                 use_palette, custom_palette, grayscale, black_and_white, bw_threshold, reduce_tile_flag,
                                    reduce_tile_threshold, limit_4_colors_per_tile, enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                                 noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size):
            with task_log("process_image_folder") as task_id:
                folder_name = "output_" + str(random.randint(0, 100000))
            while os.path.exists(folder_name):
                folder_name = "output_" + str(random.randint(0, 100000))
            os.makedirs(folder_name)
            try:
                fileListing = []
                text_for_palette = []
                for index, file in enumerate(input_files):
                    # if file is folder, skip
                    if os.path.isdir(file.name):
                        continue
                    imageData = Image.open(file.name)
                    result = process_image(imageData, width, height, aspect_ratio, color_limit, num_colors, quant_method, dither_method,
                                           use_palette, custom_palette, grayscale, black_and_white, bw_threshold, reduce_tile_flag,
                                           reduce_tile_threshold, limit_4_colors_per_tile, enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                                           noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size)
                    result[0].save(os.path.join(folder_name, os.path.basename(input_files[index].name)))
                    result[2].save(os.path.join(folder_name, os.path.basename(input_files[index].name).replace(".png", "_palette.png").replace(".jpg", "_palette.jpg")))
                    text_for_palette.append(f"File {index + 1}: {os.path.basename(input_files[index].name)}\n{result[1]}")
                # zip the folder
                with zipfile.ZipFile(os.path.join(folder_name, folder_name + ".zip"), 'w') as zipf:
                    for root, dirs, files in os.walk(folder_name):
                        for file in files:
                            if file != folder_name + ".zip":
                                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_name))
                    # add the palette text to the zip
                    zipf.writestr("palette_info.txt", "\n\n".join(text_for_palette))
                return os.path.join(os.getcwd(), folder_name, folder_name + ".zip"), "\n\n".join(text_for_palette), None, None

            except Exception as e:
                os.system("rm -rf " + folder_name)
                print(traceback.format_exc())
                return None, "Error processing folder " + str(e), None, None

        execute_button.click(process_image,
                             inputs=[image_input, new_width, new_height, keep_aspect_ratio, enable_color_limit,
                                     number_of_colors, quantization_method, dither_method, use_custom_palette,
                                     palette_image, is_grayscale, is_black_and_white, black_and_white_threshold,
                                     reduce_tile_checkbox, reduce_tile_similarity_threshold, limit_4_colors_per_tile,
                                     enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                                     noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size
                                     ],
                             outputs=[image_output, palette_text,
                                      image_output_no_palette, notice_text])

        execute_button_folder.click(process_image_folder,
                                    inputs=[folder_input, new_width, new_height, keep_aspect_ratio, enable_color_limit,
                                            number_of_colors, quantization_method, dither_method, use_custom_palette,
                                            palette_image, is_grayscale, is_black_and_white, black_and_white_threshold,
                                            reduce_tile_checkbox, reduce_tile_similarity_threshold, limit_4_colors_per_tile,
                                            enable_gothic_filter, brightness_threshold, dot_size, spacing,
                                            contrast_boost,
                                            noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size
                                            ],
                                    outputs=[image_output_zip, palette_text,
                                             image_output_no_palette, notice_text])

    return demo


def start_clearing_temporary_files_timer(interval):
    threading.Timer(interval, start_clearing_temporary_files_timer, args=[interval]).start()
    clear_temporary_files()


def clear_temporary_files():
    print("Clearing temporary files at " + str(time.time()))
    for folder in os.listdir(os.getcwd()):
        if folder.startswith("output_"):
            # get last modified date
            last_modified = os.path.getmtime(folder)
            # if the folder was last modified more than 1 hour ago, delete it
            if (time.time() - last_modified) > 600:
                # Delete folder and all files inside
                try:
                    os.system("rm -rf " + folder)
                except Exception as e:
                    print("Error deleting folder " + folder + ": " + str(e))

if __name__ == "__main__":
    interval = 60
    # clear temporary files every 60 seconds
    start_clearing_temporary_files_timer(interval)
    demo: gr.Blocks = create_gradio_interface()
    # use http basic auth with password of boobiess
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
