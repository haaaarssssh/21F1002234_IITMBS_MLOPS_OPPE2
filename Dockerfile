# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY src/ /app/src/
COPY data/ /app/data/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV GCP_PROJECT=$PROJECT_ID

# Run api.py when the container launches
# Bind to 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "$PORT"]