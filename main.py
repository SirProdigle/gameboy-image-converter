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
from pathlib import Path
from typing import Callable, Optional

from monitoring import HeartbeatMonitor, QueueClearedError, TaskExecutor, TaskMetrics

import gb_pipeline



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

HEARTBEAT_WEBHOOK_URL = os.environ.get(
    "HEARTBEAT_WEBHOOK_URL",
    "https://canary.discord.com/api/webhooks/1425053965934137449/p_K0PCtdgM8uuZw34VxCXBr7NKZdkQVgmj-mJfZDqHim4irOuibcLfUFLI3J2_KWVYJb",
)
HEARTBEAT_MESSAGE_FILE = Path("heartbeat_status.json")
HEARTBEAT_INTERVAL_SECONDS = 10
QUEUE_ALERT_THRESHOLD = 15
QUEUE_CLEAR_THRESHOLD = 50
QUEUE_CLEAR_REASON = "Queue length exceeded safety limit"

task_metrics = TaskMetrics()
task_executor = TaskExecutor(task_metrics, max_workers=2)
heartbeat_monitor: Optional[HeartbeatMonitor] = None

active_tasks = 0

@contextmanager
def task_log(task_type="image_convert"):
    global active_tasks
    task_id = str(uuid.uuid4())
    start_time = time.time()
    success = True

    active_tasks += 1
    task_metrics.task_started(task_id, task_type)

    logger.info(json.dumps({
        "event": "task_start",
        "task_id": task_id,
        "task_type": task_type,
        "timestamp": start_time,
        "active_tasks": active_tasks,
    }))

    try:
        yield task_id
    except Exception:
        success = False
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        active_tasks = max(active_tasks - 1, 0)
        task_metrics.task_finished(task_id, task_type, duration, success=success)

        logger.info(json.dumps({
            "event": "task_finish",
            "task_id": task_id,
            "task_type": task_type,
            "timestamp": end_time,
            "duration": duration,
            "active_tasks": active_tasks,
        }))


def run_in_task_executor(fn: Callable):
    def wrapper(*args, **kwargs):
        snapshot = task_metrics.task_submitted()
        queue_size = snapshot.queued_tasks
        monitor = heartbeat_monitor
        if monitor:
            monitor.check_queue_threshold(queue_size)
        if queue_size > QUEUE_CLEAR_THRESHOLD:
            task_metrics.retract_submission()
            cleared = task_executor.clear_pending(QUEUE_CLEAR_REASON)
            if monitor and cleared:
                monitor.record_queue_cleared(
                    cleared,
                    f"{QUEUE_CLEAR_REASON} (dropped {cleared} task(s))",
                )
            raise gr.Error(
                f"System queue was cleared after exceeding {QUEUE_CLEAR_THRESHOLD} pending tasks. Please retry shortly."
            )
        future = task_executor.submit(fn, *args, **kwargs)
        try:
            return future.result()
        except QueueClearedError as exc:
            raise gr.Error(str(exc)) from None

    return wrapper


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
        notice = f"✅ {len(unique_tiles)}/{max_unique_tiles} tiles used."
    else:
        notice = f"⚠️ Hit {max_unique_tiles} tile limit! Try: Lower similarity threshold • Use simpler images • Reduce detail/noise"

    return new_image, notice


def is_nearly_solid_tile(tile, tolerance=0.95):
    """Check if a tile is mostly one color (95%+ pixels the same)"""
    data = list(tile.getdata())
    if not data:
        return False
    color_counts = Counter(data)
    most_common_count = color_counts.most_common(1)[0][1]
    return most_common_count / len(data) >= tolerance

def create_solid_color_tile(tile_size, color_index, palette):
    """Create a solid color tile with a specific palette index"""
    solid_tile = Image.new('P', (tile_size, tile_size))
    solid_tile.putpalette(palette)
    solid_tile.putdata([color_index] * (tile_size * tile_size))
    return solid_tile

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
    
    # Add solid color reference tiles first (using negative coordinates to mark them as references)
    # Get the most common colors in the image
    color_counts = image.getcolors(width * height)
    if color_counts:
        # Sort by frequency and take top 4-8 colors for solid references
        top_colors = sorted(color_counts, key=lambda x: x[0], reverse=True)[:min(8, len(color_counts))]
        for i, (count, color_idx) in enumerate(top_colors):
            solid_tile = create_solid_color_tile(tile_size, color_idx, image_palette)
            # Use negative coordinates to indicate reference tiles
            unique_tiles.append((-(i+1)*tile_size, -(i+1)*tile_size, solid_tile))

    notice = ''
    for x, y, tile in tiles:
        # Check if this tile is nearly solid
        tile_is_nearly_solid = is_nearly_solid_tile(tile, tolerance=0.95)
        
        best_similarity = -1
        best_match = None
        best_is_solid_ref = False
        
        for unique_x, unique_y, unique_tile in unique_tiles:
            # Check if the unique tile is a solid reference (negative coordinates)
            is_solid_ref = unique_x < 0 and unique_y < 0
            
            sim = tile_similarity_indexed(tile, unique_tile)
            
            # If current tile is nearly solid and we're comparing to a solid reference,
            # boost the similarity score to prefer solid matches
            if tile_is_nearly_solid and is_solid_ref:
                sim = sim * 1.1  # Boost similarity for solid-to-solid matches
            
            if sim > best_similarity:
                best_similarity = sim
                best_match = (unique_x, unique_y, unique_tile)
                best_is_solid_ref = is_solid_ref

        # For nearly solid tiles, use a higher threshold to prefer solid references
        effective_threshold = similarity_threshold
        if tile_is_nearly_solid and best_is_solid_ref:
            effective_threshold = similarity_threshold * 0.9  # Lower threshold for solid matches
        
        if best_similarity > effective_threshold:
            tile_mapping[(x, y)] = (best_match[0], best_match[1])
        elif len(unique_tiles) < max_unique_tiles:
            # Only add non-solid tiles with single pixel differences if they have meaningful variance
            if not tile_is_nearly_solid or tile_variance(tile) > 0.5:
                unique_tiles.append((x, y, tile))
                tile_mapping[(x, y)] = (x, y)
            else:
                # Nearly solid tile - map to best solid reference instead
                if best_match:
                    tile_mapping[(x, y)] = (best_match[0], best_match[1])
                else:
                    unique_tiles.append((x, y, tile))
                    tile_mapping[(x, y)] = (x, y)
        else:
            best_match = min(unique_tiles, key=lambda ut: tile_similarity_indexed(tile, ut[2]))
            tile_mapping[(x, y)] = (best_match[0], best_match[1])

    # Paint the new image
    for (x, y), (ux, uy) in tile_mapping.items():
        original_tile = image.crop((x, y, x + tile_size, y + tile_size))  # Get the original tile
        
        # Check if we're using a reference tile (negative coordinates)
        if ux < 0 and uy < 0:
            # Find the reference tile in unique_tiles
            ref_tile = next((ut for ux2, uy2, ut in unique_tiles if ux2 == ux and uy2 == uy), None)
            if ref_tile:
                # For solid reference tiles, just use them directly
                new_tile = map_pattern_to_palette(ref_tile, original_tile)
            else:
                new_tile = original_tile
        else:
            pattern_tile = image.crop((ux, uy, ux + tile_size, uy + tile_size))  # Get the tile to copy the pattern from
            new_tile = map_pattern_to_palette(pattern_tile,
                                              original_tile)  # Create a new tile with the original palette but new pattern
        new_image.paste(new_tile, (x, y))

    if len(unique_tiles) < max_unique_tiles:
        notice = f"✅ {len(unique_tiles)}/{max_unique_tiles} tiles used. Try raising similarity threshold to get more details!"
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
    palette_image = Image.new("P", (1, 1))
    max_colors = min(len(color_list), 256)
    selected_colors = color_list[:max_colors]
    flat_palette = [component for color in selected_colors for component in color]
    flat_palette.extend([0] * (768 - len(flat_palette)))
    palette_image.putpalette(flat_palette)
    return palette_image


def convert_to_grayscale(image):
    return image.convert("L").convert("RGB")


def convert_to_black_and_white(image: Image, threshold: int = 128, is_inversed: bool = False):
    apply_threshold = lambda x: 0 if x > threshold else 255 if is_inversed else 255 if x > threshold else 0
    return image.convert('L').point(apply_threshold, mode='1').convert("RGB")




# ---------------------------------------------------------------------------
# Gradio UI and processing functions
# ---------------------------------------------------------------------------
#
# Mode-first layout: a top-level radio picks Artistic (today's free-form flow,
# unchanged) or one of the three GB Studio hardware modes, which route through
# gb_pipeline.convert_for_hardware instead of the old similarity-threshold /
# tile-reduction machinery. `original_width`/`original_height` and the
# `quantize_for_GBC`/`use_tile_variance` flags are no longer module-level
# `gr.State` globals (which raced across concurrent users on the hosted app);
# they flow as ordinary per-session `gr.State` components wired as event
# inputs/outputs instead.

MODE_ARTISTIC = "Artistic"
MODE_COLOR = "GB Studio: Color"
MODE_MONO = "GB Studio: Mono"
MODE_LOGO = "GB Studio: Logo"
HARDWARE_MODES = (MODE_COLOR, MODE_MONO, MODE_LOGO)

HW_DITHER_METHODS = {"None": "none", "Bayer": "bayer"}


def _hardware_preset_for_mode(mode: str, logo_subtype: str) -> str:
    """Map a UI mode (+ logo sub-toggle) to a gb_pipeline.PRESETS key."""
    if mode == MODE_COLOR:
        return "color_only"
    if mode == MODE_MONO:
        return "mono"
    if mode == MODE_LOGO:
        return "logo_mono" if logo_subtype == "Mono" else "logo_color"
    raise ValueError(f"not a hardware mode: {mode!r}")


def _image_unique_colors(image: Image.Image) -> np.ndarray:
    """(N, 3) uint8 array of an RGB image's unique colors."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.unique(arr.reshape(-1, 3), axis=0)


def _custom_palette_array(palette_image):
    """Custom palette restriction set for hardware Color mode, or None."""
    if palette_image is None:
        return None
    return _image_unique_colors(palette_image)


def _mono_ramp_array(palette_image):
    """4-color mono ramp extracted from the palette/ramp image, or None.

    None falls back to gb_pipeline's default DMG ramp. The default value of
    the shared palette image widget is `gb_palette.png`, which already *is*
    the DMG reference ramp, so the common case is a no-op passthrough.
    """
    if palette_image is None:
        return None
    quant = palette_image.convert("RGB").quantize(
        colors=4, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    colors = quant.convert("RGB").getcolors(maxcolors=4)
    if not colors:
        return None
    return np.array([color for _count, color in colors], dtype=np.uint8)


def _format_hardware_notice(result: "gb_pipeline.ConversionResult") -> str:
    stats = result.stats
    if stats["tile_budget"] is None:
        tiles_part = f"{stats['tiles_used']} tiles (no tile limit)"
    else:
        tiles_part = f"{stats['tiles_used']}/{stats['tile_budget']} tiles"
    notice = f"✅ {tiles_part} · {stats['palettes_used']} palettes · {stats['n_merges']} merges"
    if result.warnings:
        notice += "\n" + "\n".join(result.warnings)
    return notice


def _format_palette_text(palette_hex) -> str:
    if not palette_hex:
        return "None"
    lines = [f"Palette {i + 1}: {colors}" for i, colors in enumerate(palette_hex)]
    return "\n".join(lines)


def _process_artistic(image, color_limit, num_colors, quant_method, dither_method,
                      use_palette, custom_palette, grayscale, black_and_white, bw_threshold,
                      enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                      noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size):
    """The free-form Artistic conversion flow -- unchanged from before the
    hardware pipeline redesign, minus the removed hardware-only controls
    (similarity threshold, tile-variance sort, 4-colors-per-tile, tile
    reduction), which are subsumed by the GB Studio hardware modes instead.
    """
    quant_method_key = quant_method if quant_method in QUANTIZATION_METHODS else 'Median cut'
    dither_method_key = dither_method if dither_method in DITHER_METHODS else 'None'

    image_for_reference_palette = image.copy()
    if color_limit:
        image_for_reference_palette = limit_colors(
            image_for_reference_palette, limit=num_colors,
            quantize=QUANTIZATION_METHODS[quant_method_key],
            dither=DITHER_METHODS[dither_method_key],
        )
        image_for_reference_palette = image_for_reference_palette.convert('RGB')

    palette_colors = image_for_reference_palette.getcolors(maxcolors=num_colors)
    if palette_colors is None:
        palette_colors = image_for_reference_palette.quantize(colors=num_colors).convert('RGB').getcolors(maxcolors=num_colors)
    palette_colors = [color for _count, color in (palette_colors or [])]
    palette_color_values = ["#{0:02x}{1:02x}{2:02x}".format(*color) for color in palette_colors]

    if use_palette and custom_palette is not None:
        image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                             dither=DITHER_METHODS[dither_method_key], palette_image=custom_palette)
    else:
        image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                             dither=DITHER_METHODS[dither_method_key])

    text_for_palette = ""
    for i, value in enumerate(palette_color_values):
        text_for_palette += f"Palette {i + 1}: {value}\n"
    if not text_for_palette:
        text_for_palette = "None"

    if enable_gothic_filter:
        image = apply_gothic_filter(image, brightness_threshold, dot_size, spacing, contrast_boost,
                                    edge_enhance, noise_factor, apply_blur, irregular_shape, irregular_size)
        image_for_reference_palette = apply_gothic_filter(
            image_for_reference_palette, brightness_threshold, dot_size, spacing, contrast_boost,
            edge_enhance, noise_factor, apply_blur, irregular_shape, irregular_size,
        )

    if image.mode != "RGB":
        image = image.convert("RGB")
    if image_for_reference_palette.mode != "RGB":
        image_for_reference_palette = image_for_reference_palette.convert("RGB")

    if grayscale:
        image = convert_to_grayscale(image)
    if black_and_white:
        image = convert_to_black_and_white(image, threshold=bw_threshold)

    return image, text_for_palette, image_for_reference_palette, "No Warnings"


def process_image(image, mode, width, height, aspect_ratio,
                  color_limit, num_colors, quant_method, artistic_dither_method,
                  use_custom_palette, custom_palette,
                  grayscale, black_and_white, bw_threshold,
                  enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                  noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size,
                  reserve_ui_palette, hw_dither_method, tile_budget, logo_subtype):
    """Route a single image through the Artistic flow or a GB Studio hardware
    preset, depending on `mode`. Hardware modes call
    `gb_pipeline.convert_for_hardware`; the Artistic path is untouched.
    """
    with task_log("process_image"):
        if image is None:
            raise gr.Error("Please provide an input image.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = downscale_image(image, int(width), int(height), aspect_ratio)

        if mode not in HARDWARE_MODES:
            return _process_artistic(
                image, color_limit, num_colors, quant_method, artistic_dither_method,
                use_custom_palette, custom_palette, grayscale, black_and_white, bw_threshold,
                enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size,
            )

        preset = _hardware_preset_for_mode(mode, logo_subtype)
        is_logo = preset.startswith("logo")
        preset_info = gb_pipeline.PRESETS[preset]

        custom_palette_arr = None
        mono_ramp_arr = None
        if preset_info.mono:
            mono_ramp_arr = _mono_ramp_array(custom_palette)
        elif use_custom_palette:
            custom_palette_arr = _custom_palette_array(custom_palette)

        budget = None if is_logo else int(tile_budget)
        dither_key = HW_DITHER_METHODS.get(hw_dither_method, "none")

        result = gb_pipeline.convert_for_hardware(
            image, preset,
            tile_budget=budget,
            reserve_ui_palette=bool(reserve_ui_palette),
            dither=dither_key,
            custom_palette=custom_palette_arr,
            mono_ramp=mono_ramp_arr,
        )

        notice = _format_hardware_notice(result)
        palette_text = _format_palette_text(result.palette_hex)
        return result.image, palette_text, result.reference, notice


def process_image_folder(input_files, mode, width, height, aspect_ratio,
                         color_limit, num_colors, quant_method, artistic_dither_method,
                         use_custom_palette, custom_palette,
                         grayscale, black_and_white, bw_threshold,
                         enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                         noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size,
                         reserve_ui_palette, hw_dither_method, tile_budget, logo_subtype):
    with task_log("process_image_folder"):
        folder_name = "output_" + str(random.randint(0, 100000))
        while os.path.exists(folder_name):
            folder_name = "output_" + str(random.randint(0, 100000))
        os.makedirs(folder_name)
        try:
            text_for_palette = []
            for index, file in enumerate(input_files):
                if os.path.isdir(file.name):
                    continue
                image_data = Image.open(file.name)
                result = process_image(
                    image_data, mode, width, height, aspect_ratio,
                    color_limit, num_colors, quant_method, artistic_dither_method,
                    use_custom_palette, custom_palette,
                    grayscale, black_and_white, bw_threshold,
                    enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                    noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size,
                    reserve_ui_palette, hw_dither_method, tile_budget, logo_subtype,
                )
                base_name = os.path.basename(input_files[index].name)
                result[0].save(os.path.join(folder_name, base_name))
                result[2].save(os.path.join(
                    folder_name,
                    base_name.replace(".png", "_palette.png").replace(".jpg", "_palette.jpg"),
                ))
                text_for_palette.append(f"File {index + 1}: {base_name}\n{result[1]}")

            with zipfile.ZipFile(os.path.join(folder_name, folder_name + ".zip"), 'w') as zipf:
                for root, _dirs, files in os.walk(folder_name):
                    for file_name in files:
                        if file_name != folder_name + ".zip":
                            zipf.write(
                                os.path.join(root, file_name),
                                os.path.relpath(os.path.join(root, file_name), folder_name),
                            )
                zipf.writestr("palette_info.txt", "\n\n".join(text_for_palette))
            return os.path.join(os.getcwd(), folder_name, folder_name + ".zip"), "\n\n".join(text_for_palette), None, None

        except Exception as e:
            os.system("rm -rf " + folder_name)
            print(traceback.format_exc())
            return None, "Error processing folder " + str(e), None, None


def capture_original_dimensions(image):
    """Report an uploaded image's dimensions as plain return values (no
    module-level state mutation -- see the module docstring above)."""
    if image is None:
        return image, 0, 0
    width, height = image.size
    return image, width, height


def adjust_for_aspect_ratio(keep_aspect, current_width, current_height, orig_width, orig_height):
    if keep_aspect and orig_width and orig_height:
        aspect_ratio = orig_width / orig_height
        new_height = int(current_width / aspect_ratio)
        return current_width, new_height
    return current_width, current_height


def on_gb_screen_click():
    return False, 160, 144


def on_original_resolution_click(orig_width, orig_height):
    return False, orig_width or 160, orig_height or 144


def on_mode_or_logo_change(mode, logo_subtype):
    """Toggle Artistic-vs-hardware panel visibility for a mode/logo-subtype
    change. Order matches the `outputs` list on both `.change()` wirings."""
    is_artistic = mode == MODE_ARTISTIC
    is_color = mode == MODE_COLOR
    is_mono = mode == MODE_MONO
    is_logo = mode == MODE_LOGO
    reserve_visible = is_color or (is_logo and logo_subtype != "Mono")
    tiles_visible = is_color or is_mono
    tile_budget_value = 384 if is_color else 192 if is_mono else 384
    return (
        gr.update(visible=is_artistic),                            # artistic_panel
        gr.update(visible=not is_artistic),                        # hardware_panel
        gr.update(visible=reserve_visible),                        # reserve_ui_palette_checkbox
        gr.update(visible=tiles_visible, value=tile_budget_value),  # tile_budget_number
        gr.update(visible=is_logo),                                # logo_subtype_radio
        gr.update(visible=is_artistic),                            # effects_accordion
    )


def on_mode_change_lock_logo_size(mode):
    """GB Studio: Logo locks size to the fixed 160x144 GB screen automatically."""
    if mode == MODE_LOGO:
        return False, 160, 144
    return gr.update(), gr.update(), gr.update()


def create_gradio_interface():
    header = '<script async defer data-website-id="f5b8324e-09b2-4d56-8c1f-40a1f1457023" src="https://metrics.prodigle.dev/umami.js"></script><script type="module" data-entity="gameboy-image-converter" src="https://analytics.prodigle.dev/script.js"></script>'
    with gr.Blocks(head=header) as demo:
        original_width_state = gr.State(0)
        original_height_state = gr.State(0)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Input Image")
                    folder_input = gr.File(label="Input Folder", file_count='directory')

                mode_radio = gr.Radio(
                    choices=[MODE_ARTISTIC, MODE_COLOR, MODE_MONO, MODE_LOGO],
                    value=MODE_ARTISTIC,
                    label="Mode",
                )

                with gr.Row():
                    new_width = gr.Number(label="Width", value=160)
                    new_height = gr.Number(label="Height", value=144)
                    keep_aspect_ratio = gr.Checkbox(label="Keep Aspect Ratio", value=False)
                with gr.Row():
                    gb_screen_resolution = gr.Button("GB Screen (160x144)")
                    original_resolution = gr.Button("Use Original Resolution(Image)")

                with gr.Group(visible=True) as artistic_panel:
                    with gr.Row():
                        enable_color_limit = gr.Checkbox(label="Limit number of Colors", value=True)
                        number_of_colors = gr.Slider(label="Target Number of colors (32 max for GB Studio)",
                                                     minimum=2, maximum=64, step=1, value=4)
                    with gr.Row():
                        quantization_method = gr.Dropdown(choices=list(QUANTIZATION_METHODS.keys()),
                                                          label="Quantization Method", value="libimagequant")
                        artistic_dither_method = gr.Dropdown(choices=list(DITHER_METHODS.keys()),
                                                             label="Dither Method", value="None")

                with gr.Group(visible=False) as hardware_panel:
                    with gr.Row():
                        reserve_ui_palette_checkbox = gr.Checkbox(
                            label="Reserve palette 8 for dialogue/UI", value=True)
                        hw_dither_method = gr.Dropdown(choices=list(HW_DITHER_METHODS.keys()),
                                                       label="Dither Method", value="None")
                    tile_budget_number = gr.Number(label="Tile budget", value=384, precision=0)
                    logo_subtype_radio = gr.Radio(choices=["Color", "Mono"], value="Color",
                                                  label="Logo Palette Type", visible=False)

                with gr.Group():
                    use_custom_palette = gr.Checkbox(label="Use Custom Color Palette", value=True)
                    palette_image = gr.Image(label="Color Palette Image (custom palette / mono ramp)",
                                             type="pil", visible=True,
                                             value=os.path.join(os.path.dirname(__file__), "gb_palette.png"))

                with gr.Accordion("Effects (Artistic only)", open=False, visible=True) as effects_accordion:
                    with gr.Accordion("Gothic Filter (Experimental)", open=False):
                        enable_gothic_filter = gr.Checkbox(label="Enable Gothic Filter", value=False)
                        brightness_threshold = gr.Slider(label="Brightness Threshold", minimum=0, maximum=255,
                                                         value=0, step=1)
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
                        black_and_white_threshold = gr.Slider(label="Black and White Threshold", minimum=0,
                                                              maximum=255, value=128, visible=False)

                is_black_and_white.change(lambda x: gr.update(visible=x),
                                         inputs=[is_black_and_white], outputs=[black_and_white_threshold])

                image_input.change(
                    fn=capture_original_dimensions,
                    inputs=[image_input],
                    outputs=[image_input, original_width_state, original_height_state],
                )

                gb_screen_resolution.click(
                    fn=on_gb_screen_click,
                    outputs=[keep_aspect_ratio, new_width, new_height],
                )

                original_resolution.click(
                    fn=on_original_resolution_click,
                    inputs=[original_width_state, original_height_state],
                    outputs=[keep_aspect_ratio, new_width, new_height],
                )

                keep_aspect_ratio.change(
                    fn=adjust_for_aspect_ratio,
                    inputs=[keep_aspect_ratio, new_width, new_height, original_width_state, original_height_state],
                    outputs=[new_width, new_height],
                )
                new_width.change(
                    fn=adjust_for_aspect_ratio,
                    inputs=[keep_aspect_ratio, new_width, new_height, original_width_state, original_height_state],
                    outputs=[new_width, new_height],
                )

                mode_change_outputs = [artistic_panel, hardware_panel, reserve_ui_palette_checkbox,
                                       tile_budget_number, logo_subtype_radio, effects_accordion]
                mode_radio.change(fn=on_mode_or_logo_change, inputs=[mode_radio, logo_subtype_radio],
                                  outputs=mode_change_outputs)
                logo_subtype_radio.change(fn=on_mode_or_logo_change, inputs=[mode_radio, logo_subtype_radio],
                                         outputs=mode_change_outputs)
                mode_radio.change(fn=on_mode_change_lock_logo_size, inputs=[mode_radio],
                                  outputs=[keep_aspect_ratio, new_width, new_height])

            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            image_output = gr.Image(type="pil", label="Output Image", height=300)
                        with gr.Column():
                            image_output_no_palette = gr.Image(type="pil", label="Output Image (Natural Palette)",
                                          height=300)
                    notice_text = gr.Text(value="No Warnings", lines=3, max_lines=3, autoscroll=False, interactive=False, label="Warnings", show_label=False)
                    with gr.Row():
                        with gr.Column():
                            kofi_html = gr.HTML(
                                "<a href='https://ko-fi.com/prodigle' target='_blank'><img height='36' style='border:0px; margin:auto; padding: 5px; width: 100%' src='https://cdn.ko-fi.com/cdn/kofi1.png?v=2' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>")
                        with gr.Row():
                            with gr.Column():
                                discord_html = gr.HTML(
                                    "<a href='https://discord.gg/rq8UHnqN8e' target='_blank' style='display: flex; justify-content: center; align-items: center; background-color: #5865F2; border-radius: 8px; padding: 8px; text-decoration: none; color: white; font-weight: 600; height: 36px;'><svg width='24' height='24' viewBox='0 0 24 24' fill='white' style='margin-right: 8px;'><path d='M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z'/></svg>Join our Discord</a>")
                            with gr.Column():
                                github_html = gr.HTML(
                                    "<a href='https://github.com/SirProdigle/gameboy-image-converter' target='_blank' style='display: flex; justify-content: center; align-items: center; background-color: #24292e; border-radius: 8px; padding: 8px; text-decoration: none; color: white; font-weight: 600; height: 36px;'><svg width='24' height='24' viewBox='0 0 24 24' fill='white' style='margin-right: 8px;'><path d='M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z'/></svg>View on GitHub</a>")
                    palette_text = gr.Textbox(label="Custom Palette Info", value="None", interactive=False,
                                          show_copy_button=True, lines=4, max_lines=4, autoscroll=False)
                with gr.Row():
                    execute_button = gr.Button("Convert Image")
                    execute_button_folder = gr.Button("Convert Folder")
                image_output_zip = gr.File(label="Output Folder Zip", type="filepath")

        use_custom_palette.change(lambda x: gr.update(visible=x),
                                  inputs=[use_custom_palette], outputs=[palette_image])

        shared_inputs = [
            new_width, new_height, keep_aspect_ratio,
            enable_color_limit, number_of_colors, quantization_method, artistic_dither_method,
            use_custom_palette, palette_image,
            is_grayscale, is_black_and_white, black_and_white_threshold,
            enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
            noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size,
            reserve_ui_palette_checkbox, hw_dither_method, tile_budget_number, logo_subtype_radio,
        ]

        execute_button.click(run_in_task_executor(process_image),
                             inputs=[image_input, mode_radio] + shared_inputs,
                             outputs=[image_output, palette_text,
                                      image_output_no_palette, notice_text])

        execute_button_folder.click(run_in_task_executor(process_image_folder),
                                    inputs=[folder_input, mode_radio] + shared_inputs,
                                    outputs=[image_output_zip, palette_text,
                                             image_output_no_palette, notice_text])

    return demo


def start_clearing_temporary_files_timer(interval):
    threading.Timer(interval, start_clearing_temporary_files_timer, args=[interval]).start()
    clear_temporary_files()


def clear_temporary_files():
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
    if HEARTBEAT_WEBHOOK_URL:
        heartbeat_monitor = HeartbeatMonitor(
            metrics=task_metrics,
            webhook_url=HEARTBEAT_WEBHOOK_URL,
            message_store=HEARTBEAT_MESSAGE_FILE,
            interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
            queue_alert_threshold=QUEUE_ALERT_THRESHOLD,
        )
        heartbeat_monitor.start()
    else:
        logger.warning("Heartbeat monitor disabled because HEARTBEAT_WEBHOOK_URL is not set")
    demo: gr.Blocks = create_gradio_interface()
    # use http basic auth with password of boobiess
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
