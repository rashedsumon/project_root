import kagglehub

def download_data():
    # Download the latest dataset version from KaggleHub
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    print("Path to dataset files:", path)
    return path
