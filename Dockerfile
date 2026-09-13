FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY Combined_App/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
# We bind to 0.0.0.0 so Render can route traffic to it
CMD ["streamlit", "run", "Combined_App/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
