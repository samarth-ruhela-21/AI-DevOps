# 1. Use a standard Python "Hardware" image
FROM python:3.9-slim

# 2. Create a folder inside the container
WORKDIR /app

# 3. Copy your project files from your PC into the container
COPY . /app

# 4. Install the software libraries needed
RUN pip install mlflow scikit-learn pandas

# 5. Tell the container to run your AI script when it starts
CMD ["python", "train.py"]