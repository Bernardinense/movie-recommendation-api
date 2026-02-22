import os
import urllib.request
import zipfile

url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
zip_path = 'ml-100k.zip'
extract_path = 'ml-100k'

if not os.path.exists(extract_path):
    print("Baixando MovieLens 100k...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

    print("✅ Dataset pronto!")
else:
    print("✅ Dataset já existe!")