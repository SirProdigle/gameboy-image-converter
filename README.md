# 🖼️ Gameboy Image Converter

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

An powerful and versatile image processing tool with a user-friendly Gradio interface. Perfect for game developers, digital artists, and retro enthusiasts!

## 🌟 Features

- 🎨 Color palette reduction and custom palette application
- 🧩 Tile-based image processing for retro game development
- 🖋️ Gothic filter for unique artistic effectsz
- 📐 Image resizing with aspect ratio preservation
- 🔄 Batch processing for multiple images
- 🎮 Specialized features for Game Boy-style graphics

## 🛠️ Technologies Used

- Python
- Gradio
- Pillow (PIL)
- NumPy
- scikit-image
- scikit-learn
- SciPy

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processing-tool.git
   cd image-processing-tool
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. Open your web browser and navigate to `http://localhost:7860` to access the Gradio interface.

## 🖥️ Usage

1. **Upload an Image**: Click on the image upload area to select your input image.

2. **Adjust Settings**: 
   - Set the desired output dimensions
   - Choose color reduction options
   - Enable/disable special filters like the Gothic filter
   - Configure tile reduction settings for retro-style graphics

3. **Process**: Click the "Convert Image" button to apply your selected transformations.

4. **Results**: View the processed image and download it directly from the interface.

## 🎓 Advanced Features

### Tile Reduction
Perfect for retro game development, this feature reduces the number of unique 8x8 pixel tiles in your image, optimizing it for limited memory systems like the Game Boy.

### Custom Palette
Upload your own color palette image to apply specific color schemes to your processed images.

### Gothic Filter
Create unique, stylized images with customizable dot patterns and contrasts.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/image-processing-tool/issues).

## 📜 License

This project is [MIT](https://opensource.org/licenses/MIT) licensed.

## 🙏 Acknowledgements

- [Gradio](https://www.gradio.app/) for the amazing web interface
- [scikit-image](https://scikit-image.org/) for advanced image processing capabilities
- All the open-source contributors whose libraries made this project possible

---

Made with ❤️ by [Prodigle](https://github.com/sirprodigle)
