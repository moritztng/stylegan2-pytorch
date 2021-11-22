from os import remove
from os.path import isdir, join
from pathlib import Path
from gdown import download
from zipfile import ZipFile

def download_ffhq(path):
    if not isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        path_zip = join(path, 'ffhq.zip')
        download(id='1EL0pQnON0SFOY8XXn8DX4T6cIcKf4CNu', output=path_zip)
        with ZipFile(path_zip, 'r') as f:
            f.extractall(path)
        remove(path_zip)

def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad
