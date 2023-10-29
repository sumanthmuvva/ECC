# Use a specific version of the TensorFlow Docker image
FROM tensorflow/tensorflow:2.7.0

WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use ENTRYPOINT to always run the script when the container starts
ENTRYPOINT [ "python", "./image_classification.py" ]