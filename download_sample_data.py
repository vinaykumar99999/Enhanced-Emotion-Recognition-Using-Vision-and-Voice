import os
import urllib.request
import zipfile
import shutil

def download_sample_data():
    # Create data directory if it doesn't exist
    if not os.path.exists('sample_data'):
        os.makedirs('sample_data')
    
    # Download sample images
    urls = {
        'sample_data/face1.jpg': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face.jpg',
        'sample_data/face2.jpg': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face2.jpg',
        'sample_data/face3.jpg': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face3.jpg'
    }
    
    for file_path, url in urls.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {file_path}")

if __name__ == "__main__":
    download_sample_data() 