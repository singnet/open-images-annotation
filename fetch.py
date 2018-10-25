import requests
import os
import sys
import json
import math
import pathlib

from tqdm import tqdm

def download_and_extract(download_dir, dest_dir, url, file_filter):
    local_name = os.path.join(download_dir, url.split('/')[-1])
    print("Downloading %s => %s" % (url, local_name))

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if os.path.exists(local_name) and ('content-length' not in response.headers or int(response.headers.get('content-length')) == os.path.getsize(local_name)):
        print(local_name + " is already downloaded.")
        del response
    else:
        block_size = 1024
        wrote = 0

        with open(local_name, 'wb') as handle:
            for data in tqdm(response.iter_content(block_size), total=math.ceil(total_size//block_size), unit='KB'):
                if data:
                    wrote = wrote + len(data)
                    handle.write(data)

        if total_size != 0 and wrote != total_size:
            raise Exception("ERROR, total size does not match total written bytes")

    if local_name.endswith('.bz2'):
        if '.tar.' in local_name:
            import tarfile
            tar = tarfile.open(local_name, "r:bz2")  
            for tar_item in tar:
                if file_filter is None or tar_item.name in file_filter:
                    print("Extracting %s : %s => %s" % (local_name, tar_item.name, os.path.join(model_dir, tar_item.name)))
                    tar.extract(tar_item, path=model_dir)
            tar.close()
        else:
            import bz2
            local_name_unzip = os.path.join(model_dir, '.'.join(url.split('/')[-1].split('.')[:-1]))
            print("Extracting %s => %s" % (local_name, local_name_unzip))
            with bz2.BZ2File(local_name) as f:
                with open(local_name_unzip, 'wb') as dest:
                    dest.write(f.read())
    elif local_name.endswith('.zip'):
        from zipfile import ZipFile
        zfile = ZipFile(local_name)
        for filename in zfile.namelist():
            if file_filter is None or filename in file_filter:
                print("Extracting %s : %s => %s" % (local_name, filename, os.path.join(model_dir, filename)))
                zfile.extract(filename, path=model_dir)
    else:
        import shutil
        dest_local_name = os.path.join(model_dir, url.split('/')[-1])
        print("Copying %s => %s" % (local_name, dest_local_name))
        shutil.copyfile(local_name, dest_local_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--download-dir", action="store", default="downloads")
    args = parser.parse_args()

    with open("models_and_data.json", "r") as f:
        obj = json.load(f)
        model_urls = obj.get("model_urls", [])
        data_urls = obj.get("data_urls", [])

    my_path = os.getcwd()

    download_dir = os.path.join(my_path, args.download_dir)
    pathlib.Path(download_dir).mkdir(parents=True, exist_ok=True)
    if args.models:
        pathlib.Path(os.path.join(my_path, "models")).mkdir(parents=True, exist_ok=True)
        for url, file_filter in model_urls:
            model_dir = os.path.join(my_path, "models")
            download_and_extract(download_dir, model_dir, url, file_filter)
    if args.annotations:
        pathlib.Path(os.path.join(my_path, "data")).mkdir(parents=True, exist_ok=True)
        for url, file_filter in model_urls:
            data_dir = os.path.join(my_path, "data")
            download_and_extract(download_dir, data_dir, url, file_filter)
