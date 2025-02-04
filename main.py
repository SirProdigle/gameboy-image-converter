from collections import Counter
import gradio as gr
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, deltaE_cie76
import random
import os
import zipfile
import threading
import time
import traceback

# Constants
TILE_SIZE = 8
MAX_UNIQUE_TILES = 192

# Dithering and Quantization Methods
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

# Utility Functions
def tile_variance(tile):
    """Compute the variance of a tile based on luminance."""
    if tile.mode == 'RGB':
        arr = np.array(tile.convert('L'))  # Convert to grayscale for luminance
    else:
        arr = np.array(tile)
    return np.var(arr)

def tile_similarity(tile1, tile2):
    """Calculate the Euclidean similarity between two tiles in LAB color space."""
    arr1 = np.array(tile1.convert('LAB')).flatten()
    arr2 = np.array(tile2.convert('LAB')).flatten()
    euclidean_distance = np.linalg.norm(arr1 - arr2)
    max_distance = np.sqrt(3 * (255 ** 2))  # Maximum possible distance in LAB
    return 1 - (euclidean_distance / max_distance)

def dominant_color(tile):
    """
    Find the dominant color in a tile.
    """
    arr = np.array(tile)
    if arr.ndim == 2:
        # Grayscale
        values, counts = np.unique(arr, return_counts=True)
    elif arr.ndim == 3:
        # Color
        values, counts = np.unique(arr.reshape(-1, arr.shape[-1]), axis=0, return_counts=True)
    else:
        raise ValueError("Unexpected image format!")
    dominant = values[np.argmax(counts)]
    if arr.ndim == 2:
        return (int(dominant),) * 3
    else:
        return tuple(dominant.astype(int))

# Gothic Filter
def apply_gothic_filter(image, threshold, dot_size, spacing, contrast_boost=1.5, edge_enhance=True, noise_factor=0.1,
                        apply_blur=True, irregular_shape=True, irregular_size=True):
    """Applies a gothic-style filter to the image."""
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
        for _ in range(int(image.width * image.height * 0.1)):
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
                    for i in range(8):
                        angle = i * (2 * np.pi / 8) + random.uniform(0, np.pi / 4)
                        r = adjusted_dot_size * (1 + random.uniform(-0.2, 0.2))
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
        original_colors = image.getcolors()
        original_num_colors = len(original_colors) if original_colors else 256
        result = result.quantize(colors=original_num_colors, method=Image.MEDIANCUT)

    return result

# Tile Processing Functions
def get_most_common_color(tile):
    """Returns the most common color in a tile."""
    colors = tile.getcolors(tile.size[0] * tile.size[1])
    return max(colors, key=lambda item: item[0])[1] if colors else (0, 0, 0)

def get_adjacent_common_color(main_image, x, y, default_color):
    """Returns the most common color in adjacent pixels, excluding default_color."""
    adjacent_colors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < main_image.width and 0 <= ny < main_image.height:
                adjacent_color = main_image.getpixel((nx, ny))
                if adjacent_color != default_color:
                    adjacent_colors.append(adjacent_color)
    return max(set(adjacent_colors), key=adjacent_colors.count) if adjacent_colors else default_color

def adjust_tile_colors(tile, surrounding_colors):
    """Adjusts tile colors based on surrounding colors."""
    if not surrounding_colors:
        return tile
    most_common_color = Counter(surrounding_colors).most_common(1)[0][0]
    return np.full_like(tile, most_common_color)

def most_common_border_color(image, x, y, tile_size, default_color):
    """Calculates the most common color in the bordering pixels of a tile."""
    border_colors = []
    border_positions = [(x + i, y) for i in range(-1, tile_size + 1)] + \
                       [(x + i, y + tile_size - 1) for i in range(-1, tile_size + 1)] + \
                       [(x, y + i) for i in range(tile_size)] + \
                       [(x + tile_size - 1, y + i) for i in range(tile_size)]
    for bx, by in border_positions:
        if 0 <= bx < image.width and 0 <= by < image.height:
            color = image.getpixel((bx, by))
            if color != (0, 0, 0):
                border_colors.append(color)
    return max(set(border_colors), key=border_colors.count) if border_colors else default_color

def normalize_tile(tile):
    """
    Convert a tile to a normalized form based on the frequency of each color,
    disregarding the specific colors themselves.
    """
    flat_tile = list(tile.getdata())
    color_counts = Counter(flat_tile)
    sorted_colors = sorted(color_counts.keys(), key=lambda color: (-color_counts[color], color))
    color_map = {color: rank for rank, color in enumerate(sorted_colors)}
    normalized_pixels = [color_map[color] for color in flat_tile]
    new_tile = Image.new('P', tile.size)
    new_tile.putpalette(tile.getpalette())
    new_tile.putdata(normalized_pixels)
    return new_tile

def tile_similarity_indexed(tile1, tile2):
    """Compare two tiles based on their normalized forms."""
    norm_tile1 = normalize_tile(tile1)
    norm_tile2 = normalize_tile(tile2)
    data1 = norm_tile1.getdata()
    data2 = norm_tile2.getdata()
    matches = sum(1 for p1, p2 in zip(data1, data2) if p1 == p2)
    return matches / len(data1)

def map_pattern_to_palette(source_tile, target_tile):
    """Maps the pattern of the source tile to the palette of the target tile."""
    norm_source_tile = normalize_tile(source_tile)
    norm_source_data = list(norm_source_tile.getdata())
    target_data = list(target_tile.getdata())
    pattern_to_color = {}
    for norm_val, target_val in zip(norm_source_data, target_data):
        if norm_val not in pattern_to_color:
            pattern_to_color[norm_val] = target_val

    new_tile_data = [pattern_to_color.get(norm_val, 0) for norm_val in norm_source_data] # Replace with `0` if not found
    new_tile = Image.new('P', target_tile.size)
    new_tile.putpalette(target_tile.getpalette())
    new_tile.putdata(new_tile_data)
    return new_tile

def reduce_tiles_index(image, tile_size=TILE_SIZE, max_unique_tiles=MAX_UNIQUE_TILES, similarity_threshold=0.7, use_tile_variance=False):
    """Reduces the number of unique tiles in an image based on indexed color patterns."""
    width, height = image.size
    width -= width % tile_size
    height -= height % tile_size
    image = image.crop((0, 0, width, height)).convert('P', colors=256, dither=Image.NONE)

    tiles = [(x, y, image.crop((x, y, x + tile_size, y + tile_size)))
             for y in range(0, height, tile_size)
             for x in range(0, width, tile_size)]

    if use_tile_variance:
        tiles.sort(key=lambda x: tile_variance(x[2]))

    unique_tiles = []
    tile_mapping = {}
    new_image = Image.new('P', (width, height))
    new_image.putpalette(image.getpalette())

    for x, y, tile in tiles:
        best_similarity = -1
        best_match = None
        for unique_x, unique_y, unique_tile in unique_tiles:
            sim = tile_similarity_indexed(tile, unique_tile)
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

    for (x, y), (ux, uy) in tile_mapping.items():
        original_tile = image.crop((x, y, x + tile_size, y + tile_size))
        pattern_tile = image.crop((ux, uy, ux + tile_size, uy + tile_size))
        new_tile = map_pattern_to_palette(pattern_tile, original_tile)
        new_image.paste(new_tile, (x, y))

    notice = f"Unique tiles used: {len(unique_tiles)}/{max_unique_tiles}\n"
    if len(unique_tiles) < max_unique_tiles:
        notice += "Consider increasing Tile Similarity Threshold."
    else:
        notice += "Out of spare tiles. Consider reducing Tile Similarity Threshold."

    return new_image, notice

def reduce_tiles(image, tile_size=TILE_SIZE, max_unique_tiles=MAX_UNIQUE_TILES, similarity_threshold=0.7, use_tile_variance=False):
    """Reduces the number of unique tiles in an image."""
    width, height = image.size
    width -= width % tile_size
    height -= height % tile_size
    image = image.crop((0, 0, width, height)).convert('P')

    tiles = [(x, y, image.crop((x, y, x + tile_size, y + tile_size)))
             for y in range(0, height, tile_size)
             for x in range(0, width, tile_size)]

    if use_tile_variance:
        tiles.sort(key=lambda x: tile_variance(x[2]))

    unique_tiles = []
    tile_mapping = {}
    new_image = Image.new('RGB', (width, height))

    for x, y, tile in tiles:
        best_similarity = -1
        best_match = None
        for unique_x, unique_y, unique_tile in unique_tiles:
            sim = tile_similarity(tile, unique_tile)
            if sim > best_similarity:
                best_similarity = sim
                best_match = (unique_x, unique_y, unique_tile)

        if best_similarity > similarity_threshold:
            tile_mapping[(x, y)] = (best_match[0], best_match[1])
        elif len(unique_tiles) < max_unique_tiles:
            unique_tiles.append((x, y, tile))
            tile_mapping[(x, y)] = (x, y)
        else:
            best_match = min(unique_tiles, key=lambda ut: tile_similarity(tile, ut[2]))
            tile_mapping[(x, y)] = (best_match[0], best_match[1])

    for (x, y), (ux, uy) in tile_mapping.items():
        tile = image.crop((ux, uy, ux + tile_size, uy + tile_size))
        new_image.paste(tile, (x, y))

    notice = f"Unique tiles used: {len(unique_tiles)}/{max_unique_tiles}\n"
    if len(unique_tiles) < max_unique_tiles:
        notice += "Consider increasing Tile Similarity Threshold."
    else:
        notice += "Out of spare tiles. Consider reducing Tile Similarity Threshold."

    return new_image, notice

# Image Manipulation Functions
def downscale_image(image: Image, new_width: int, new_height: int, keep_aspect_ratio: bool) -> Image:
    """Downscales the image to the given dimensions, optionally preserving aspect ratio."""
    if keep_aspect_ratio:
        image = image.copy()
        image.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def limit_colors(image, limit=16, quantize=None, dither=None, palette_image=None):
    """Limits the number of colors in the image."""
    if palette_image:
        color_palette = palette_image.quantize(colors=len(set(palette_image.getcolors())))
    else:
        color_palette = image.quantize(colors=limit, kmeans=limit if limit else 0, method=quantize, dither=dither)
    return image.quantize(palette=color_palette, dither=dither)

def create_palette_from_colors(color_list):
    """Creates a palette image from a list of colors."""
    palette_image = Image.new("RGB", (1, len(color_list)))
    for i, color in enumerate(color_list):
        palette_image.putpixel((0, i), color)
    return palette_image.convert("P", palette=Image.ADAPTIVE)

def convert_to_grayscale(image):
    """Converts the image to grayscale."""
    return image.convert("L").convert("RGB")

def convert_to_black_and_white(image: Image, threshold: int = 128, is_inversed: bool = False):
    """Converts the image to black and white based on a threshold."""
    fn = lambda x: 0 if (is_inversed and x > threshold) or (not is_inversed and x < threshold) else 255
    return image.convert('L').point(fn, mode='1').convert("RGB")

# Tile Extraction and Palette Generation
def extract_tiles(image, tile_size=(TILE_SIZE, TILE_SIZE)):
    """Extract tiles from the image."""
    tiles = []
    if image.width <= 0 or image.height <= 0:
        return tiles # Return empty list if image has no width or height

    for y in range(0, image.height, tile_size[1]):
        for x in range(0, image.width, tile_size[0]):
            box = (x, y, x + tile_size[0], y + tile_size[1])
            # Add check to ensure box coordinates are valid
            if box[3] <= box[1] or box[2] <= box[0]:
                continue # Skip invalid box
            tiles.append(image.crop(box))
    return tiles

def generate_palette(tile, num_colors=4):
    """Generate a palette for a tile using K-means clustering."""
    if not isinstance(tile, np.ndarray):
        tile = np.array(tile)
    data = tile.reshape((-1, 3))
    unique_colors = np.unique(data, axis=0)

    if len(unique_colors) <= num_colors:
        # If the tile has fewer unique colors than requested, use those colors directly
        return unique_colors.round().astype(int)
    else:
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10).fit(data)
        return kmeans.cluster_centers_.round().astype(int)

def get_color_distribution(tile):
    """Analyzes the tile and returns a frequency distribution of its colors."""
    data = tile.reshape(-1, 3)
    colors, counts = np.unique(data, axis=0, return_counts=True)
    total = counts.sum()
    return {tuple(color): count / total for color, count in zip(colors, counts)}

# Palette Matching and Application
def find_best_matching_palette(tile_distribution, existing_palettes, adjacent_distributions=None,
                               balance_factor=0.5, key_color_weight=1):
    """Finds the best matching palette for a tile, considering adjacent tiles and key colors."""
    if not existing_palettes:
        return 0  # Return default palette index if no palettes exist

    best_score = float('inf')
    best_palette_index = 0 # Default to the first palette if none is better
    key_colors = [(255, 255, 255), (0, 0, 0)]

    for palette_index, palette in enumerate(existing_palettes):
        palette_colors = np.array(palette)
        tile_score = 0

        for color, frequency in tile_distribution.items():
            distances = np.linalg.norm(palette_colors - np.array(color), axis=1)
            closest_distance = np.min(distances)
            weight = key_color_weight if color in key_colors else 1
            tile_score += frequency * closest_distance * weight

        context_score = 0
        if adjacent_distributions:
            for adj_dist in adjacent_distributions:
                adj_score = 0
                for adj_color, adj_freq in adj_dist.items():
                    distances = np.linalg.norm(palette_colors - np.array(adj_color), axis=1)
                    closest_distance = np.min(distances)
                    weight = key_color_weight if adj_color in key_colors else 1
                    adj_score += adj_freq * closest_distance * weight
                context_score += adj_score
            context_score /= len(adjacent_distributions)

        combined_score = (1 - balance_factor) * tile_score + balance_factor * context_score

        if combined_score < best_score:
            best_score = combined_score
            best_palette_index = palette_index

    return best_palette_index

def apply_palette(tile, palette, dither=False):
    """Applies a palette to a tile, optionally using dithering."""
    tile_data = np.array(tile, dtype=np.uint8)
    tile_lab = rgb2lab(tile_data)
    palette_lab = rgb2lab(np.array(palette, dtype=np.uint8).reshape((1, -1, 3))).reshape((-1, 3))
    new_tile_data = np.zeros_like(tile_data)

    for i in range(tile_data.shape[0]):
        for j in range(tile_data.shape[1]):
            original_color_lab = tile_lab[i, j]
            distances = np.linalg.norm(palette_lab - original_color_lab, axis=1)
            closest_palette_index = np.argmin(distances)
            new_tile_data[i, j] = palette[closest_palette_index]

            if dither:
                error = original_color_lab - palette_lab[closest_palette_index]
                if j + 1 < tile_data.shape[1]:
                    tile_lab[i, j + 1] += error * 7 / 16
                if i + 1 < tile_data.shape[0]:
                    if j > 0:
                        tile_lab[i + 1, j - 1] += error * 3 / 16
                    tile_lab[i + 1, j] += error * 5 / 16
                    if j + 1 < tile_data.shape[1]:
                        tile_lab[i + 1, j + 1] += error * 1 / 16

    return Image.fromarray(new_tile_data, 'RGB')

# Palette Refinement and Analysis
def create_refined_palettes(cluster_centers, tiles, num_palettes=8, colors_per_palette=4):
    """Creates refined palettes based on color frequency and distribution."""
    flat_cluster_centers = np.vstack(cluster_centers)
    cluster_centers_rgb = (np.clip(lab2rgb(flat_cluster_centers), 0, 1) * 255).astype(np.uint8)

    refined_palettes_rgb = [[] for _ in range(num_palettes)]
    color_frequencies = np.zeros(len(cluster_centers_rgb), dtype=int)

    for tile in tiles:
        lab_tile = rgb2lab(np.array(tile.convert('RGB'), dtype=np.float64) / 255).reshape(-1, 3)
        for color in lab_tile:
            distances = np.linalg.norm(cluster_centers_rgb - color, axis=1)
            nearest_color_index = np.argmin(distances)
            color_frequencies[nearest_color_index] += 1

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

    def color_distance(c1, c2):
        """Calculates the CIELAB Delta E (1976) distance between two colors."""
        c1_lab = rgb2lab(np.array(c1).reshape(1, 1, 3) / 255)
        c2_lab = rgb2lab(np.array(c2).reshape(1, 1, 3) / 255)
        return deltaE_cie76(c1_lab, c2_lab)[0, 0]

    global_color_usage = {tuple(color): 0 for color in cluster_centers_rgb}
    for palette in refined_palettes_rgb:
        for color in palette:
            global_color_usage[tuple(color)] += 1

    for palette_index, palette in enumerate(refined_palettes_rgb):
        while len(palette) < colors_per_palette:
            best_color = None
            best_color_score = -np.inf
            existing_colors_tuples = [tuple(color) for color in palette]
            for color in cluster_centers_rgb:
                color_tuple = tuple(color)
                if color_tuple not in existing_colors_tuples:
                    usage_score = -global_color_usage[color_tuple]
                    diversity_score = min([color_distance(np.array(color), np.array(existing_color)) for existing_color in palette] or [np.inf])
                    total_score = usage_score + diversity_score
                    if total_score > best_color_score:
                        best_color_score = total_score
                        best_color = color
            if best_color is not None:
                refined_palettes_rgb[palette_index].append(best_color)
                global_color_usage[tuple(best_color)] += 1

    return refined_palettes_rgb

def analyze_and_construct_palettes(tiles, max_palettes=8, max_colors=32, local_influence=0.5):
    """Analyzes tiles and constructs palettes based on color distribution and frequency."""
    unique_colors_set = set()
    for tile in tiles:
        rgb_tile = np.array(tile.convert('RGB'), dtype=np.uint8)
        unique_colors = set(tuple(color) for row in rgb_tile for color in row)
        unique_colors_set.update(unique_colors)

    num_unique_colors = len(unique_colors_set)
    num_clusters = min(max_colors, num_unique_colors)

    lab_tiles = [rgb2lab(np.array(tile.convert('RGB'), dtype=np.float64) / 255) for tile in tiles]
    all_tiles_lab = np.vstack([tile.reshape(-1, 3) for tile in lab_tiles])

    global_kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    global_kmeans.fit(all_tiles_lab)
    broad_palette_lab = global_kmeans.cluster_centers_

    global_labels = global_kmeans.labels_
    global_color_weights = np.bincount(global_labels, minlength=num_clusters) / float(len(global_labels))

    for tile_lab in lab_tiles:
        local_labels = global_kmeans.predict(tile_lab.reshape(-1, 3))
        local_color_weights = np.bincount(local_labels, minlength=num_clusters) / float(len(local_labels))
        global_color_weights = (1 - local_influence) * global_color_weights + local_influence * local_color_weights

    if not np.any(global_color_weights):
        global_color_weights = np.ones_like(global_color_weights) / len(global_color_weights)

    weighted_colors = np.repeat(broad_palette_lab, np.maximum(global_color_weights.astype(int), 1), axis=0)

    if weighted_colors.size == 0:
        weighted_colors = broad_palette_lab

    final_kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10)
    final_kmeans.fit(weighted_colors)

    refined_palettes_lab = [final_kmeans.cluster_centers_[i:i + 4] for i in range(0, len(final_kmeans.cluster_centers_), 4)]
    refined_palettes_rgb = create_refined_palettes(refined_palettes_lab, tiles)

    return refined_palettes_rgb

def process_tiles(tiles, max_palettes=8, tile_width=TILE_SIZE, tile_height=TILE_SIZE, image_width=None, num_colors=4, should_dither=False, enhanced_palettes=None):
    """Processes tiles to limit them to the best palettes."""
    tile_palette_mapping = []
    if not enhanced_palettes:
        enhanced_palettes = analyze_and_construct_palettes(tiles, max_palettes, num_colors)

    if not enhanced_palettes: # Handle empty palettes case
        return tiles, [], "", []

    tiles_per_row = image_width // tile_width if image_width else None
    tile_palettes = []
    palette_for_tile_text = ""

    for index, tile in enumerate(tiles):
        np_tile = np.array(tile)
        tile_distribution = get_color_distribution(np_tile)
        adjacent_distributions = []

        if tiles_per_row:
            positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dy, dx in positions:
                adj_index = index + dy * tiles_per_row + dx
                if 0 <= adj_index < len(tiles) and (dy == 0 or (index // tiles_per_row == adj_index // tiles_per_row)):
                    adjacent_tile = tiles[adj_index]
                    adj_distribution = get_color_distribution(np.array(adjacent_tile))
                    adjacent_distributions.append(adj_distribution)

        closest_palette_index = find_best_matching_palette(
            tile_distribution, enhanced_palettes, adjacent_distributions
        )

        tile_palettes.append(closest_palette_index)
        x, y = index % tiles_per_row, index // tiles_per_row
        palette_for_tile_text += f"Tile at ({x},{y}) uses palette {closest_palette_index + 1}\n"
        tile_palette_mapping.append(closest_palette_index)

    processed_tiles = []
    for i in range(len(tiles)):
        try:
            processed_tiles.append(apply_palette(tiles[i], enhanced_palettes[tile_palettes[i]], dither=should_dither))
        except IndexError as e:
            print(f"IndexError: tile_palettes[{i}] = {tile_palettes[i]}, len(enhanced_palettes) = {len(enhanced_palettes)}")
            raise e


    return processed_tiles, enhanced_palettes, palette_for_tile_text, tile_palette_mapping

def apply_mapped_colors_to_tile(tile, tile_palette, custom_palette_palette):
    """Applies mapped colors to a tile based on a custom palette."""
    if tile.mode != "RGB":
        tile = tile.convert("RGB", palette=Image.ADAPTIVE, dither=Image.Dither.NONE)
    new_tile = Image.new('RGB', tile.size)
    pixels = tile.load()
    new_pixels = new_tile.load()

    tile_palette_tuples = [tuple(color) for color in tile_palette]

    for y in range(tile.size[1]):
        for x in range(tile.size[0]):
            original_color = pixels[x, y]
            if original_color in tile_palette_tuples:
                color_index = tile_palette_tuples.index(original_color)
                new_color = custom_palette_palette[color_index]
                new_pixels[x, y] = new_color
            else:
                new_pixels[x, y] = original_color

    return new_tile

# Main Image Processing Function (updated to accept quantize_for_GBC and use_tile_variance)
def process_image(image, width, height, aspect_ratio, color_limit, num_colors, quant_method, dither_method,
                  use_palette, custom_palette, grayscale, black_and_white, bw_threshold, reduce_tile_flag,
                  reduce_tile_threshold, limit_4_colors_per_tile, enable_gothic_filter, brightness_threshold,
                  dot_size, spacing, contrast_boost, noise_factor, edge_enhance, apply_blur, irregular_shape,
                  irregular_size, quantize_for_GBC, use_tile_variance):
    """Processes the image based on the given parameters."""
    if num_colors <= 4:
        limit_4_colors_per_tile = False
    text_for_palette = ""
    text_for_palette_tile_application = ""

    image = downscale_image(image, int(width), int(height), aspect_ratio)
    notice = None
    image_for_reference_palette = image.copy()

    if color_limit:
        quant_method_key = quant_method if quant_method in QUANTIZATION_METHODS else 'Median cut'
        dither_method_key = dither_method if dither_method in DITHER_METHODS else 'None'

        image_for_reference_palette = limit_colors(
            image_for_reference_palette,
            limit=num_colors,
            quantize=QUANTIZATION_METHODS[quant_method_key],
            dither=DITHER_METHODS[dither_method_key]
        )
        image_for_reference_palette = image_for_reference_palette.convert('RGB')

        palette_color_values = []
        enhanced_palettes = None
        tile_palette_mapping = None

        if limit_4_colors_per_tile and not reduce_tile_flag:
            tiles = extract_tiles(image_for_reference_palette)
            processed_tiles, enhanced_palettes, text_for_palette_tile_application, tile_palette_mapping = process_tiles(
                tiles,
                image_width=image_for_reference_palette.width,
                num_colors=num_colors,
                should_dither=(dither_method_key != "None")
            )
            new_image = Image.new('RGB', image_for_reference_palette.size)
            tile_index = 0
            for y in range(0, image_for_reference_palette.height, TILE_SIZE):
                for x in range(0, image_for_reference_palette.width, TILE_SIZE):
                    new_image.paste(processed_tiles[tile_index], (x, y))
                    tile_index += 1
            image_for_reference_palette = new_image
            palette_color_values = [[f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in palette] for palette in enhanced_palettes]
        else:
            palette_colors = image_for_reference_palette.getcolors(maxcolors=num_colors)
            palette_colors = [color for count, color in palette_colors]
            palette_color_values = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in palette_colors]

        if use_palette and custom_palette is not None:
            if quantize_for_GBC and not reduce_tile_flag:
                if limit_4_colors_per_tile:
                    image = image_for_reference_palette.copy()
                    if not enhanced_palettes or not tile_palette_mapping:
                        processed_tiles, enhanced_palettes, text_for_palette_tile_application, tile_palette_mapping = process_tiles(
                            extract_tiles(image),
                            image_width=image_for_reference_palette.width,
                            num_colors=num_colors,
                            should_dither=(dither_method_key != "None")
                        )
                    custom_palette = custom_palette.quantize(
                        colors=num_colors,
                        method=QUANTIZATION_METHODS[quant_method_key],
                        dither=DITHER_METHODS[dither_method_key]
                    )
                    image = image.convert("P", palette=Image.ADAPTIVE, dither=Image.Dither.NONE if dither_method_key == "None" else Image.Dither.FLOYDSTEINBERG)
                    image_tiles = processed_tiles
                    custom_palette_palette = [(r, g, b) for r, g, b in zip(custom_palette.getpalette()[0::3],
                                                                            custom_palette.getpalette()[1::3],
                                                                            custom_palette.getpalette()[2::3])]
                    mapped_image = Image.new('RGB', image.size)
                    for index, tile in enumerate(image_tiles):
                        tile_palette = enhanced_palettes[tile_palette_mapping[index]]
                        recolored_tile = apply_mapped_colors_to_tile(tile, tile_palette, custom_palette_palette)
                        mapped_image.paste(recolored_tile, (index % (image.width // TILE_SIZE) * TILE_SIZE,
                                                            index // (image.width // TILE_SIZE) * TILE_SIZE))
                    image = mapped_image
                else:
                    image = limit_colors(
                        image_for_reference_palette,
                        limit=num_colors,
                        quantize=QUANTIZATION_METHODS[quant_method_key],
                        dither=DITHER_METHODS[dither_method_key],
                        palette_image=custom_palette
                    )
            else:
                image = limit_colors(
                    image,
                    limit=num_colors,
                    quantize=QUANTIZATION_METHODS[quant_method_key],
                    dither=DITHER_METHODS[dither_method_key],
                    palette_image=custom_palette
                )
        else:
            image = limit_colors(
                image,
                limit=num_colors,
                quantize=QUANTIZATION_METHODS[quant_method_key],
                dither=DITHER_METHODS[dither_method_key]
            )

        if reduce_tile_flag and limit_4_colors_per_tile:
            image = image_for_reference_palette.copy()
            image = limit_colors(
                image,
                limit=num_colors,
                quantize=QUANTIZATION_METHODS[quant_method_key],
                dither=DITHER_METHODS[dither_method_key]
            )
            custom_palette_info = [color for _, color in custom_palette.getcolors()] if use_palette else None
            enhanced_palettes = analyze_and_construct_palettes(
                extract_tiles(image),
                max_palettes=8,
                max_colors=num_colors
            )
            image, notice = reduce_tiles_index(
                image,
                similarity_threshold=reduce_tile_threshold,
                use_tile_variance=use_tile_variance
            )
            image = image.convert("RGB")
            tiles = extract_tiles(image)
            processed_tiles, enhanced_palettes, text_for_palette_tile_application, tile_palette_mapping = process_tiles(
                tiles,
                image_width=image_for_reference_palette.width,
                num_colors=len(image.getcolors(maxcolors=65536)),
                enhanced_palettes=enhanced_palettes
            )

            if use_palette:
                image_tiles = processed_tiles
                mapped_image = Image.new('RGB', image.size)
                for index, tile in enumerate(image_tiles):
                    image_for_reference_palette.paste(tile, (index % (image.width // TILE_SIZE) * TILE_SIZE,
                                                               index // (image.width // TILE_SIZE) * TILE_SIZE))
                    tile_palette = enhanced_palettes[tile_palette_mapping[index]]
                    recolored_tile = apply_mapped_colors_to_tile(tile, tile_palette, custom_palette_info)
                    mapped_image.paste(recolored_tile, (index % (image.width // TILE_SIZE) * TILE_SIZE,
                                                        index // (image.width // TILE_SIZE) * TILE_SIZE))
                image = mapped_image
                palette_color_values = [[f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in palette] for palette in enhanced_palettes]
        elif reduce_tile_flag:
            image, notice = reduce_tiles(image, similarity_threshold=reduce_tile_threshold, use_tile_variance=use_tile_variance)
            image_for_reference_palette, notice = reduce_tiles_index(
                image_for_reference_palette,
                similarity_threshold=reduce_tile_threshold,
                use_tile_variance=use_tile_variance
            )

        if enable_gothic_filter:
            image = apply_gothic_filter(
                image, brightness_threshold, dot_size, spacing, contrast_boost,
                edge_enhance, noise_factor, apply_blur, irregular_shape, irregular_size
            )
            image_for_reference_palette = apply_gothic_filter(
                image_for_reference_palette, brightness_threshold, dot_size, spacing, contrast_boost,
                edge_enhance, noise_factor, apply_blur, irregular_shape, irregular_size
            )

        for i in range(len(palette_color_values)):
            text_for_palette += f"Palette {i + 1}: {palette_color_values[i]}\n"
        text_for_palette += f"\n\n{text_for_palette_tile_application}"

        image = image.convert("RGB")
        image_for_reference_palette = image_for_reference_palette.convert("RGB")

        return image, text_for_palette, image_for_reference_palette, notice

    if grayscale:
        image = convert_to_grayscale(image)
    if black_and_white:
        image = convert_to_black_and_white(image, threshold=bw_threshold)
    if reduce_tile_flag:
        image, notice = reduce_tiles(image, similarity_threshold=reduce_tile_threshold, use_tile_variance=use_tile_variance.value)
    return image, text_for_palette, None, notice

# File Processing Function (updated to accept quantize_for_GBC and use_tile_variance)
def process_image_folder(input_files, width, height, aspect_ratio, color_limit, num_colors, quant_method,
                         dither_method, use_palette, custom_palette, grayscale, black_and_white, bw_threshold,
                         reduce_tile_flag, reduce_tile_threshold, limit_4_colors_per_tile, enable_gothic_filter,
                         brightness_threshold, dot_size, spacing, contrast_boost, noise_factor, edge_enhance,
                         apply_blur, irregular_shape, irregular_size, quantize_for_GBC, use_tile_variance):
    """Processes a folder of images."""
    folder_name = f"output_{random.randint(0, 100000)}"
    while os.path.exists(folder_name):
        folder_name = f"output_{random.randint(0, 100000)}"
    os.makedirs(folder_name)

    try:
        file_listing = []
        text_for_palette = []
        for index, file in enumerate(input_files):
            if os.path.isdir(file.name):
                continue
            image_data = Image.open(file.name)
            result = process_image(
                image_data, width, height, aspect_ratio, color_limit, num_colors, quant_method,
                dither_method, use_palette, custom_palette, grayscale, black_and_white, bw_threshold,
                reduce_tile_flag, reduce_tile_threshold, limit_4_colors_per_tile, enable_gothic_filter,
                brightness_threshold, dot_size, spacing, contrast_boost, noise_factor, edge_enhance,
                apply_blur, irregular_shape, irregular_size, quantize_for_GBC, use_tile_variance
            )
            base_name = os.path.basename(file.name)
            result[0].save(os.path.join(folder_name, base_name))
            if result[2] is not None:
                palette_base_name = base_name.replace(".png", "_palette.png").replace(".jpg", "_palette.jpg")
                result[2].save(os.path.join(folder_name, palette_base_name))
            text_for_palette.append(f"File {index + 1}: {base_name}\n{result[1]}")

        zip_file_name = f"{folder_name}.zip"
        with zipfile.ZipFile(os.path.join(folder_name, zip_file_name), 'w') as zipf:
            for root, _, files in os.walk(folder_name):
                for file in files:
                    if file != zip_file_name:
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_name))
            zipf.writestr("palette_info.txt", "\n\n".join(text_for_palette))

        return os.path.join(os.getcwd(), folder_name, zip_file_name), "\n\n".join(text_for_palette), None, None

    except Exception as e:
        for the_file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as ex:
                print(f"Error deleting file: {ex}")
        os.rmdir(folder_name)
        print(traceback.format_exc())
        return None, f"Error processing folder: {e}", None, None

# Gradio UI Setup (states moved inside create_gradio_interface)
def capture_original_dimensions(image):
    """Captures the original dimensions of the uploaded image."""
    if image is None:
        return None
    width, height = image.size
    return width, height, image

def adjust_for_aspect_ratio(keep_aspect, current_width, current_height, orig_w, orig_h):
    """Adjusts the dimensions to maintain aspect ratio if the option is selected."""
    if keep_aspect and orig_w.value and orig_h.value:
        aspect_ratio = orig_w.value / orig_h.value
        new_height = int(current_width / aspect_ratio)
        return current_width, new_height
    return current_width, current_height

def on_original_resolution_click(orig_w, orig_h):
    return False, orig_w.value, orig_h.value

def on_limit_4_colors_per_tile_change(x, quantize_state):
    quantize_state = x
    return x


def on_use_tile_variance_click(x, use_tile_state):
    use_tile_state = x
    return x


def create_gradio_interface():
    """Creates the Gradio interface."""
    with gr.Blocks() as demo:
        # Define state variables inside the interface
        original_width = gr.State(value=0)
        original_height = gr.State(value=0)
        quantize_for_GBC = gr.State(False)
        use_tile_variance = gr.State(False)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_input = gr.Image(label="Input Image", type="pil")
                    folder_input = gr.File(label="Input Folder", file_count='directory')
                with gr.Row():
                    new_width = gr.Number(label="Width", value=160, precision=0)
                    new_height = gr.Number(label="Height", value=144, precision=0)
                    keep_aspect_ratio = gr.Checkbox(label="Keep Aspect Ratio", value=False)
                with gr.Row():
                    logo_resolution = gr.Button("Use Logo Resolution")
                    original_resolution = gr.Button("Use Original Resolution(Image)")
                with gr.Row():
                    enable_color_limit = gr.Checkbox(label="Limit number of Colors", value=True)
                    number_of_colors = gr.Slider(label="Target Number of colors (32 max for GB Studio)", minimum=2,
                                                 maximum=64, step=1, value=4)
                    limit_4_colors_per_tile = gr.Checkbox(
                        label="Limit to 4 colors per tile, 8 palettes (For GB Studio development only)",
                        value=False, visible=True)
                with gr.Group():
                    with gr.Row():
                        reduce_tile_checkbox = gr.Checkbox(
                            label="Reduce to 192 unique 8x8 tiles (Not needed for LOGO scene mode)", value=False)
                        use_tile_variance_checkbox = gr.Checkbox(
                            label="Sort by tile complexity (Complex tiles get saved first)", value=False)
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

                # Event Handlers
                is_black_and_white.change(lambda x: gr.update(visible=x),
                                          inputs=[is_black_and_white], outputs=[black_and_white_threshold])

                limit_4_colors_per_tile.change(on_limit_4_colors_per_tile_change, inputs=[limit_4_colors_per_tile, quantize_for_GBC],
                                               outputs=[quantize_for_GBC])

                use_tile_variance_checkbox.change(on_use_tile_variance_click, inputs=[use_tile_variance_checkbox, use_tile_variance],
                                                  outputs=[use_tile_variance])

                image_input.change(capture_original_dimensions,
                                   inputs=[image_input],
                                   outputs=[original_width, original_height, image_input])

                logo_resolution.click(lambda: (False, 160, 144),
                                      outputs=[keep_aspect_ratio, new_width, new_height])

                original_resolution.click(on_original_resolution_click,
                                          inputs=[original_width, original_height],
                                          outputs=[keep_aspect_ratio, new_width, new_height])

                keep_aspect_ratio.change(adjust_for_aspect_ratio,
                                         inputs=[keep_aspect_ratio, new_width, new_height, original_width, original_height],
                                         outputs=[new_width, new_height])

                new_width.change(adjust_for_aspect_ratio,
                                 inputs=[keep_aspect_ratio, new_width, new_height, original_width, original_height],
                                 outputs=[new_width, new_height])

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
                            notice_text = gr.Text(value="No Warnings", lines=3, max_lines=3, autoscroll=False,
                                                  interactive=False, label="Warnings")
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

        reduce_tile_checkbox.change(lambda x: gr.update(visible=x),
                                    inputs=[reduce_tile_checkbox], outputs=[reduce_tile_similarity_threshold])

        use_custom_palette.change(lambda x: gr.update(visible=x),
                                  inputs=[use_custom_palette], outputs=[palette_image])

        # Pass state variables quantize_for_GBC and use_tile_variance to the processing functions
        execute_button.click(process_image,
                             inputs=[image_input, new_width, new_height, keep_aspect_ratio, enable_color_limit,
                                     number_of_colors, quantization_method, dither_method, use_custom_palette,
                                     palette_image, is_grayscale, is_black_and_white, black_and_white_threshold,
                                     reduce_tile_checkbox, reduce_tile_similarity_threshold, limit_4_colors_per_tile,
                                     enable_gothic_filter, brightness_threshold, dot_size, spacing, contrast_boost,
                                     noise_factor, edge_enhance, apply_blur, irregular_shape, irregular_size,
                                     quantize_for_GBC, use_tile_variance],
                             outputs=[image_output, palette_text, image_output_no_palette, notice_text])

        execute_button_folder.click(process_image_folder,
                                    inputs=[folder_input, new_width, new_height, keep_aspect_ratio, enable_color_limit,
                                            number_of_colors, quantization_method, dither_method, use_custom_palette,
                                            palette_image, is_grayscale, is_black_and_white, black_and_white_threshold,
                                            reduce_tile_checkbox, reduce_tile_similarity_threshold, limit_4_colors_per_tile,
                                            enable_gothic_filter, brightness_threshold, dot_size, spacing,
                                            contrast_boost, noise_factor, edge_enhance, apply_blur, irregular_shape,
                                            irregular_size, quantize_for_GBC, use_tile_variance],
                                    outputs=[image_output_zip, palette_text, image_output_no_palette, notice_text])

    return demo

# Temporary File Management
def start_clearing_temporary_files_timer(interval):
    """Starts a timer to clear temporary files periodically."""
    threading.Timer(interval, start_clearing_temporary_files_timer, args=[interval]).start()
    clear_temporary_files()

def clear_temporary_files():
    """Clears temporary files older than 10 minutes."""
    print(f"Clearing temporary files at {time.time()}")
    for folder in os.listdir(os.getcwd()):
        if folder.startswith("output_"):
            last_modified = os.path.getmtime(folder)
            if (time.time() - last_modified) > 600:
                try:
                    for the_file in os.listdir(folder):
                        file_path = os.path.join(folder, the_file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                os.rmdir(file_path)
                        except Exception as ex:
                            print(f"Error deleting file: {ex}")
                    os.rmdir(folder)
                except Exception as e:
                    print(f"Error deleting folder {folder}: {e}")

# Main Function
if __name__ == "__main__":
    start_clearing_temporary_files_timer(60)
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)