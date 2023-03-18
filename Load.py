import wget
import pandas as pd
import requests
import io
import os


def LoadDataSet(path, url):
    download = requests.get(url).content
    my_file = open(path, "w+")
    my_file.write(download.decode('utf-8'))
    my_file.close()


print("Load")
if not os.path.isdir('train'):
    os.mkdir('train')
if not os.path.isdir('test'):
    os.mkdir('test')
if not os.path.isdir('content'):
        os.mkdir('content')

LoadDataSet("test/test.csv", "https://raw.githubusercontent.com/ksmk99/Mlops2023-Data-Set/main/Data/test.csv")
LoadDataSet("train/train.csv", "https://raw.githubusercontent.com/ksmk99/Mlops2023-Data-Set/main/Data/train.csv")
