import os
import numpy as np
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

extensions = np.array(["jpg","png","tiff"])

def GetAllImagesOnPath(path):
    files = []
    for ext in extensions:
        files.extend(GetFileWithSpecificExtension(path,ext))

    for f in files:
        print(f)

    return files

def Reshape(src,rows,columns):
    return src.reshape(rows,columns)

def GetFileWithSpecificExtension(path,extension):
    for f in os.listdir(path):
        if f.endswith("." + extension) and os.path.isfile(os.path.join(path,f)):
            yield f

def GetFullPathOfFile(filename):
        return os.path.abspath(filename)

def GetLabelsFromFile(labelsPath):
    with open(labelsPath,"r") as f:
        labels = []
        
        for line in f:
            labels.append(line.rstrip('\n'))

    return labels

def FileExist(path):
    return os.path.exists(path)

def DevideArrayintoEqualsParts(array,size):
    """Yield successive n-sized parts from array."""
    for i in range(0, len(array), size):
        yield array[i:i + size]

def GetGraySkaledImage(img):
    g_img = []
    i =0
    for i in range(int(img.size/3)):
        avg = (img[i] + img[1024+i] + img[2048 + i]) /3 
        g_img.append(int(avg))
    
    return np.array(g_img)

def Normalize_0to1(array):
    normalized_X = preprocessing.normalize([array])
    return normalized_X