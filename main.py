from collections import Counter

import gradio as gr
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from numpy import std
from PIL import Image, features
import random
import os
import zipfile
import threading
import time
from sklearn.cluster import KMeans

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
                          f"Tiles left to process: {remaining_tiles}   \n"
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
        notice = (f"Unique tiles used : {len(unique_tiles)}/{max_unique_tiles}\n\n"
                  f"\n\nConsider increasing Tile Similarity Threshold.")
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
    header = '<script async defer data-website-id="f5b8324e-09b2-4d56-8c1f-40a1f1457023" src="https://metrics.prodigle.dev/umami.js"></script>'
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
                    enable_color_limit = gr.Checkbox(label="Limit number of colors", value=True)
                    number_of_colors = gr.Slider(label="Number of colors", minimum=2, maximum=256, step=1, value=4)
                with gr.Group():
                    with gr.Row():
                        limit_4_colors_per_tile = gr.Checkbox(label="Limit to 4 colors per tile, 8 palettes",
                                                              value=False, visible=True)
                        reduce_tile_checkbox = gr.Checkbox(label="Reduce to 192 unique 8x8 tiles", value=False)
                        use_tile_variance_checkbox = gr.Checkbox(label="Sort by tile complexity", value=False)
                    reduce_tile_similarity_threshold = gr.Slider(label="Tile similarity threshold", minimum=0.3,
                                                                 maximum=0.99, value=0.8, step=0.01, visible=False)
                with gr.Row():
                    quantization_method = gr.Dropdown(choices=list(QUANTIZATION_METHODS.keys()),
                                                      label="Quantization Method", value="libimagequant")
                    dither_method = gr.Dropdown(choices=list(DITHER_METHODS.keys()), label="Dither Method",
                                                value="None")
                with gr.Group():
                    use_custom_palette = gr.Checkbox(label="Use Custom Color Palette", value=True)
                    quantize_for_gbc_checkbox = gr.Checkbox(label="Quantize for Game Boy Color Palette Override",
                                                            value=False, visible=True)
                    palette_image = gr.Image(label="Color Palette Image", type="pil", visible=True,
                                             value=os.path.join(os.path.dirname(__file__), "gb_palette.png"))
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

                def on_quantize_for_gbc_click(x):
                    quantize_for_GBC.value = x

                def on_use_tile_variance_click(x):
                    use_tile_variance.value = x

                quantize_for_gbc_checkbox.change(on_quantize_for_gbc_click, inputs=[quantize_for_gbc_checkbox])
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
            """Generate a 4-color palette for an 8x8 tile using K-means clustering."""
            data = np.array(tile).reshape((-1, 3))
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(data)
            palette = kmeans.cluster_centers_.astype(int)
            return palette

        def find_closest_palette(palette, palettes):
            """Find the closest existing palette to the given one."""
            min_distance = float('inf')
            closest_palette = None
            for existing_palette in palettes:
                distance = np.linalg.norm(palette - existing_palette)
                if distance < min_distance:
                    min_distance = distance
                    closest_palette = existing_palette
            return closest_palette

        def apply_palette(tile, palette):
            """Apply a 4-color palette to an 8x8 tile."""
            tile_data = np.array(tile)
            reshaped_tile = tile_data.reshape((-1, 3))
            new_tile_data = np.zeros_like(reshaped_tile)
            for i, color in enumerate(palette):
                distances = np.sqrt(np.sum((reshaped_tile - color) ** 2, axis=1))
                closest_color_indices = distances == np.min(distances, axis=0)
                new_tile_data[closest_color_indices] = color
            return Image.fromarray(new_tile_data.reshape(tile_data.shape).astype('uint8'))

        def process_tiles(tiles, max_palettes=8):
            """Process the tiles to limit them to the best 4-color palettes."""
            unique_palettes = []
            tile_palettes = []
            for tile in tiles:
                palette = generate_palette(tile)
                if not unique_palettes:
                    unique_palettes.append(palette)
                    tile_palettes.append(palette)
                else:
                    closest_palette = find_closest_palette(palette, unique_palettes)
                    if closest_palette is None or len(unique_palettes) < max_palettes:
                        unique_palettes.append(palette)
                        tile_palettes.append(palette)
                    else:
                        tile_palettes.append(closest_palette)
            return [apply_palette(tiles[i], tile_palettes[i]) for i in range(len(tiles))], unique_palettes

        def process_image(image, width, height, aspect_ratio, color_limit, num_colors, quant_method, dither_method,
                          use_palette, custom_palette, grayscale, black_and_white, bw_threshold, reduce_tile_flag,
                          reduce_tile_threshold, limit_4_colors_per_tile):
            text_for_palette = ""
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = downscale_image(image, int(width), int(height), aspect_ratio)
            notice = None
            if color_limit:
                quant_method_key = quant_method if quant_method in QUANTIZATION_METHODS else 'Median cut'
                dither_method_key = dither_method if dither_method in DITHER_METHODS else 'None'

                image_for_reference_palette: Image = image.copy()
                image_for_reference_palette = limit_colors(image_for_reference_palette, limit=num_colors,
                                                           quantize=QUANTIZATION_METHODS[quant_method_key],
                                                           dither=DITHER_METHODS[dither_method_key])
                image_for_reference_palette: Image = image_for_reference_palette.convert('RGB')


                if limit_4_colors_per_tile:
                    tiles = extract_tiles(image_for_reference_palette)
                    processed_tiles, _ = process_tiles(tiles)
                    # Reconstruct the image from the processed tiles
                    new_image = Image.new('RGB', image_for_reference_palette.size)
                    tile_index = 0
                    for y in range(0, image_for_reference_palette.height, 8):
                        for x in range(0, image_for_reference_palette.width, 8):
                            new_image.paste(processed_tiles[tile_index], (x, y))
                            tile_index += 1
                    image_for_reference_palette = new_image

                palette_colors = image_for_reference_palette.getcolors(maxcolors=num_colors)
                palette_colors = [color for count, color in palette_colors]

                palette_color_values = [
                    "#{0:02x}{1:02x}{2:02x}".format(*color) for color in palette_colors
                ]

                if use_palette and custom_palette is not None:
                    if quantize_for_GBC and quantize_for_GBC.value == True:
                        image = limit_colors(image, limit=min(num_colors, len(custom_palette.getcolors())),
                                             quantize=QUANTIZATION_METHODS[quant_method_key],
                                             dither=DITHER_METHODS[dither_method_key])
                        custom_palette = custom_palette.quantize(colors=num_colors,
                                                                 method=QUANTIZATION_METHODS[quant_method_key],
                                                                 dither=DITHER_METHODS[dither_method_key])
                        image.putpalette(custom_palette.getpalette())
                        image = image.quantize(colors=num_colors, method=QUANTIZATION_METHODS[quant_method_key],
                                               dither=DITHER_METHODS[dither_method_key])
                    else:
                        image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                                             dither=DITHER_METHODS[dither_method_key], palette_image=custom_palette)
                else:
                    image = limit_colors(image, limit=num_colors, quantize=QUANTIZATION_METHODS[quant_method_key],
                                         dither=DITHER_METHODS[dither_method_key])
                if reduce_tile_flag:
                    image, notice = reduce_tiles(image, similarity_threshold=reduce_tile_threshold)

                # Return all necessary components including the processed image and color values
                # set pallete_color_values to exactly 4 values nomatter if there's less or more
                for i in range(len(palette_color_values)):
                    text_for_palette += f"Color {i + 1}: {palette_color_values[i]}\n"

                return image, text_for_palette, image_for_reference_palette, notice

            if grayscale:
                image = convert_to_grayscale(image)
            if black_and_white:
                image = convert_to_black_and_white(image, threshold=bw_threshold)
            if reduce_tile_flag:
                image, _ = reduce_tiles(image, similarity_threshold=reduce_tile_threshold)
            return (
                image,
                text_for_palette, None,
                None
            )

        def process_image_folder(input_files, width, height, aspect_ratio, color_limit, num_colors, quant_method, dither_method,
                                 use_palette, custom_palette, grayscale, black_and_white, bw_threshold, reduce_tile_flag,
                                    reduce_tile_threshold, limit_4_colors_per_tile):
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
                                           reduce_tile_threshold, limit_4_colors_per_tile)
                    result[0].save(os.path.join(folder_name, os.path.basename(input_files[index].name)))
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
                return None, "Error processing folder " + str(e)

        execute_button.click(process_image,
                             inputs=[image_input, new_width, new_height, keep_aspect_ratio, enable_color_limit,
                                     number_of_colors, quantization_method, dither_method, use_custom_palette,
                                     palette_image, is_grayscale, is_black_and_white, black_and_white_threshold,
                                     reduce_tile_checkbox, reduce_tile_similarity_threshold, limit_4_colors_per_tile],
                             outputs=[image_output, palette_text,
                                      image_output_no_palette, notice_text])

        execute_button_folder.click(process_image_folder,
                                    inputs=[folder_input, new_width, new_height, keep_aspect_ratio, enable_color_limit,
                                            number_of_colors, quantization_method, dither_method, use_custom_palette,
                                            palette_image, is_grayscale, is_black_and_white, black_and_white_threshold,
                                            reduce_tile_checkbox, reduce_tile_similarity_threshold, limit_4_colors_per_tile],
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
                os.system("rm -rf " + folder)

if __name__ == "__main__":
    interval = 60
    # clear temporary files every 60 seconds
    start_clearing_temporary_files_timer(interval)
    demo: gr.Blocks = create_gradio_interface()
    # use http basic auth with password of boobiess
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
