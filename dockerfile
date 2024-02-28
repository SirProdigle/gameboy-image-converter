# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Clone the required repository
RUN apt-get update && apt-get install -y git \
    && git clone https://github.com/SirProdigle/gameboy-image-converter.git .

# Install dependencies for building Python packages and libimagequant
RUN apt-get install -y build-essential libimagequant-dev libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev

# Install Python dependencies
COPY requirements.txt ./
RUN python -m venv venv
RUN . venv/bin/activate
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install libimagequant

# Install Pillow with imagequant enabled
RUN pip uninstall Pillow -y
RUN python -m pip cache purge
Run python3 -m pip install --upgrade Pillow  --global-option="-C" --global-option="imagequant=enable" --no-binary --no-cache-dir :all:

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
