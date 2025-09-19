import kagglehub

# Download latest version
path = kagglehub.dataset_download("rmisra/clothing-fit-dataset-for-size-recommendation")

print("Path to dataset files:", path)