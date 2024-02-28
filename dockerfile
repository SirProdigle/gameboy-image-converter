# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Clone the required repository
RUN apt-get update && apt-get install -y git \
    && git clone https://github.com/SirProdigle/gameboy-image-converter.git .

# Install dependencies for building Python packages and libimagequant
RUN apt-get install -y build-essential libimagequant-dev

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Pillow with imagequant enabled
RUN CFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib" python3 -m pip install imagequant==enable
RUN pip install --upgrade Pillow --global-option="-C" --global-option="imagequant=enable" --no-binary :all:

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
