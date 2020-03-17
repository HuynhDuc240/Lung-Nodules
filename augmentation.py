import re
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from utils import readCsv, readMhd, extractCube
from tqdm import tqdm
from scipy import ndimage

def createCube(cubeSize=80):
    csvLines = readCsv("csv/trainSet.csv")
    header = csvLines[0]
    nodules = csvLines[1:]

    for i,n in tqdm(enumerate(nodules[:1])):
        x = int(n[header.index("x")])
        y = int(n[header.index("y")])
        z = int(n[header.index("z")])

        lnd = int(n[header.index("LNDbID")])

        ctr = np.array([x,y,z])
        [scan,spacing,_,_] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
        cube = extractCube(scan,spacing,ctr,cubeSize)

        np.save("cube/{:0}.npy".format(i),cube)

    # REVIEW needs avaiable for cube 3D
def rotate(image, angle):
    rotated = ndimage.rotate(image,angle,reshape=False,mode='reflect')
    return rotated

def zoom(image ,rate ,mode = "constant"):
    h,w  = image.shape[:2]
    if rate < 1:
        # REVIEW follow modes: constant, nearest, reflect, mirror
        if mode == "nearest":
            out = zoom_nearest(image,rate)
        elif mode == "reflect":
            out = zoom_reflect(image,rate)
        elif mode == "mirror" :
            out = zoom_mirror(image,rate)
        else:
            zh = int(np.round(h * rate))
            zw = int(np.round(w * rate))
            top = (h - zh) // 2
            left = (w - zw) // 2
            # Zero-padding
            out = np.zeros_like(image)
            out[top:top+zh, left:left+zw] = ndimage.zoom(image,rate)
    elif rate > 1:
        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / rate))
        zw = int(np.round(w / rate))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(image[top:top+zh, left:left+zw],rate)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = image

    return out

# ANCHOR zoom option
def zoom_nearest(image,rate):
    h,w  = image.shape[:2]
    zh = int(np.round(h * rate))
    zw = int(np.round(w * rate))
    top = (h - zh) // 2
    left = (w - zw) // 2
    # Zero-padding
    out = np.zeros_like(image)
    out[top:top+zh, left:left+zw] = ndimage.zoom(image,rate)
    # The vertical
    v = 0
    while v < left:
        out[:,v] = out[:,left]
        v+=1
    v = left+zw
    while v < w:
        out[:,v] = out[:,left+zw-1] 
        v+=1
    ho = 0
    while ho < top:
        out[ho,:] = out[top,:]
        ho+=1
    ho = top+zh
    while ho < h:
        out[ho,:] = out[top+zh-1,:]
        ho+=1
    return out
def zoom_reflect(image,rate):
    h,w  = image.shape[:2]
    zh = int(np.round(h * rate))
    zw = int(np.round(w * rate))
    top = (h - zh) // 2
    left = (w - zw) // 2
    # Zero-padding
    out = np.zeros_like(image)
    out[top:top+zh, left:left+zw] = ndimage.zoom(image,rate)
    # The vertical
    v = left - 1
    i = left
    while v >= 0:
        out[:,v] = out[:,i]
        v-=1 
        i+=1
    
    v = left + zw
    i = left + zw - 1
    while v < w:
        out[:,v] = out[:,i]
        v+=1
        i-=1
    # The horizontal
    ho = top - 1
    i = top
    while ho >= 0:
        out[ho,:] = out[i,:]
        ho-=1
        i+=1
    
    ho = top + zh
    i = top + zh -1
    while ho < h:
        out[ho,:] = out[i,:]
        ho+=1
        i-=1
    return out
def zoom_mirror(image,rate):
    h,w  = image.shape[:2]
    zh = int(np.round(h * rate))
    zw = int(np.round(w * rate))
    top = (h - zh) // 2
    left = (w - zw) // 2
    # Zero-padding
    out = np.zeros_like(image)
    out[top:top+zh, left:left+zw] = ndimage.zoom(image,rate)
    # The vertical
    v = left - 1
    i = left + 1
    while v >= 0:
        out[:,v] = out[:,i]
        v-=1 
        i+=1
    
    v = left + zw
    i = left + zw - 2
    while v < w:
        out[:,v] = out[:,i]
        v+=1
        i-=1
    # The horizontal
    ho = top - 1
    i = top + 1
    while ho >= 0:
        out[ho,:] = out[i,:]
        ho-=1
        i+=1
    
    ho = top + zh
    i = top + zh - 2
    while ho < h:
        out[ho,:] = out[i,:]
        ho+=1
        i-=1
    return out
def augmet_cube(cube, id):
    for rate in np.arange(0.8,1.5,0.1):
        for angle in range(0,360,30):
            augmented_cube = []
            for s in cube:
                zoomed = zoom(s,rate,mode="mirror")
                rotated = rotate(zoomed,angle)
                augmented_cube.append(rotated)
        
if __name__ == "__main__":
    cube = np.load("cube/0.npy")
    image = cube[299//2,:,:]
    for rate in np.arange(0.8,1.5,0.1):
        zoomed = zoom(image,rate,mode="mirror")
        for angle in range(0,360,30):
            rotated = rotate(zoomed,angle)
            plt.imshow(rotated)
            plt.show()
