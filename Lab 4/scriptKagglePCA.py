import kagglehub

# Download latest version
path = kagglehub.dataset_download("vikrishnan/iris-dataset")

print("Path to dataset files:", path)