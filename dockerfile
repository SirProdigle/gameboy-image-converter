# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Clone the required repository
RUN apt-get update && apt-get install -y git \
    && git clone https://github.com/SirProdigle/gameboy-image-converter.git .

# Install dependencies for building Python packages and libimagequant
RUN apt-get install -y build-essential libimagequant-dev libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev libtiff5-dev tk-dev tcl-dev libwebp-dev libharfbuzz-dev libfribidi-dev libxcb1-dev libxml2-dev libxslt1-dev libexif-dev libraqm-dev

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install libimagequant

# Install Pillow with libimagequant enabled
RUN pip uninstall Pillow -y
RUN python -m pip cache purge
RUN CFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib" python -m pip install --upgrade Pillow --no-cache-dir --no-binary :all:

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
