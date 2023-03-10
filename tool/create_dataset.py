import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from multiprocessing import Pool


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            # txn.put(k, v)
            if type(v) is str:
                txn.put(str(k).encode(), v.encode())
            else:
                txn.put(str(k).encode(), v)

def createDataset(outputPath, imageDirPath, lexiconList=None, checkValid=True, db_volume=100995116):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    nSamples = 0
    env = lmdb.open(outputPath, map_size=db_volume)
    cache = {}
    cnt = 1
    for _, _, imageNames in os.walk(imageDirPath):
        imageNames.sort(key=lambda x: len(x.split('_')[0]))
        nSamples = len(imageNames)
        
        for i, imageName in enumerate(imageNames):
            imagePath = os.path.join(imageDirPath, imageName)
            label = imageName.split('_')[0]
            
            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue
            with open(imagePath, 'rb') as fd:
                imageBin = fd.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image. remove it' % imagePath)
                    os.remove(imagePath)
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label
            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[i])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
            
        nSamples = cnt-1
        cache['num-samples'] = str(nSamples)
        writeCache(env, cache)
        print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    createDataset('train_crnn_images_lmdb', 'train_crnn_images', db_volume=1009951160)
    createDataset('val_crnn_images_lmdb', 'val_crnn_images', db_volume=100995116)
    pass
