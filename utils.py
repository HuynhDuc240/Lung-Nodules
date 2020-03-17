import csv
import os
import sys
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm
import random
class TrainSet():
    def __init__(self, cube_size=80,cube_size_mm=False, n_class=3):
        self.cube_size_mm = cube_size_mm
        self.cube_size = cube_size
        self.n_class = n_class
        
    def createTrainCsv(self):
        print("creating Csv file from TrainNodules_gt...")
        trainSet = pd.DataFrame(columns=['LNDbID', 'FindingID', 'x', 'y', 'z','Volume', 'Text'])    
        csvLines = readCsv("csv/trainNodules_gt.csv")
        header = csvLines[0]
        nodules = csvLines[1:]
        for n in tqdm(nodules):
            lnd = int(n[header.index("LNDbID")])
            # rad = n[header.index("RadID")]
            find = n[header.index("FindingID")]
            # nodule = n[header.index("Nodule")]
            volume = n[header.index("Volume")]
            #### texture ####
            texture = float(n[header.index("Text")])
            texthr = [7/3,11/3]
            if texture>=texthr[0] and texture<=texthr[1]:
                texture = 1
            elif texture > texthr[1]:
                texture = 2
            else:
                texture = 0
            #### coordinates ####
            x = float(n[header.index("x")])
            y = float(n[header.index("y")])
            z = float(n[header.index("z")])

            ctr = np.array([x,y,z])
            [_,spacing,origin,transfmat] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
            transfmat_toimg,_ = getImgWorldTransfMats(spacing,transfmat)
            ctr = convertToImgCoord(ctr,origin,transfmat_toimg) 
            trainSet = trainSet.append({header[0]:lnd, header[3]:find, header[4]:int(ctr[0]), header[5]:int(ctr[1]), header[6]:int(ctr[2]), header[header.index('Volume')]:volume, header[10]:texture},ignore_index=True)

        trainSet = trainSet.sample(frac=1).reset_index(drop=True)
        trainSet.to_csv("csv/trainSet.csv",index=False)


    ##### Get data slice  ##### 
    def getTrainNonSegmentation(self):
        print("getting train with non segmentation...\n")
        # create_csv("csv/trainSet.csv")
        csvLines = readCsv("csv/trainSet.csv")
        header = csvLines[0]
        nodules = csvLines[1:]

        TrainSet = []

        for n in tqdm(nodules):
            lnd = int(n[header.index("LNDbID")])
            x = int(n[header.index("x")])
            y = int(n[header.index("y")])
            z = int(n[header.index("z")])
            volume = float(n[header.index("Volume")])
            if volume >= 51:
                volume = 51
            if self.cube_size_mm == False:
                volume = 51
            #### get cube ####
            ctr = np.array([x,y,z])
            [scan,spacing,_,_] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
            cube = extractCube(scan,spacing,ctr,self.cube_size,volume)

            #### create slices and segment lung ####
            sliceX = cube[int(cube.shape[0]/2),:,:]
            sliceY = cube[:,int(cube.shape[1]/2),:]
            sliceZ = cube[:,:,int(cube.shape[2]/2)]
            nodule = np.array([sliceX,sliceY,sliceZ])
            
            #### show slices ####
            # fig,axs = plt.subplots(1,3)
            # axs[0].imshow(sliceX)
            # axs[1].imshow(sliceY)
            # axs[2].imshow(sliceZ)
            # plt.show()

            TrainSet.append(nodule)
        TrainSet = np.array(TrainSet)
        
        with open("pickle/"+str(self.n_class)+"-classes/nonSegTrainSet-"+str(self.cube_size)+".pkl","wb") as file:
            pickle.dump(TrainSet,file) 

    def getTrainSegmented(self):
        print("getting train set segmented...")
        csvLines = readCsv("csv/trainSet.csv")
        header = csvLines[0]
        nodules = csvLines[1:]

        TrainSet = []

        for n in tqdm(nodules):
            lnd = int(n[header.index("LNDbID")])
            x = int(n[header.index("x")])
            y = int(n[header.index("y")])
            z = int(n[header.index("z")])
            volume = n[header.index("Volume")]
            if volume >= 51:
                volume = 51
            if self.cube_size_mm == False:
                volume = 51
            #### get cube ####
            ctr = np.array([x,y,z])
            [_,spacing,_,_] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
            with open('pickle/LNDb-{:04}.pkl'.format(lnd),"rb") as file:
                scan = pickle.load(file)
            cube = extractCube(scan,spacing,ctr,self.cube_size,volume)

            #### create slices and segment lung ####
            sliceX = cube[int(cube.shape[0]/2),:,:]
            sliceY = cube[:,int(cube.shape[1]/2),:]
            sliceZ = cube[:,:,int(cube.shape[2]/2)]
            nodule = np.array([sliceX,sliceY,sliceZ])
            
            #### show slices ####
            # fig,axs = plt.subplots(1,3)
            # axs[0].imshow(sliceX)
            # axs[1].imshow(sliceY)
            # axs[2].imshow(sliceZ)
            # plt.show()

            TrainSet.append(nodule)
        TrainSet = np.array(TrainSet)
        
        with open("pickle/"+str(self.n_class)+"-classes/segmentedTrainSet-"+str(self.cube_size)+".pkl","wb") as file:
            pickle.dump(TrainSet,file)

    def getLabel(self):
        print("getting label of train set...\n")
        csvLines = readCsv("csv/trainSet.csv")
        header = csvLines[0]
        nodules = csvLines[1:]
        
        Label = []
        if self.n_class==3:
            print("label for "+str(self.n_class)+"class\n")
            for n in tqdm(nodules):
                textture = int(n[header.index("Text")])
                Label.append(textture)
            Label = np.array(Label)
        else:
            print("label for "+str(self.n_class)+"class\n")
            for n in tqdm(nodules):
                textture = int(n[header.index("Text")])
                if textture >= 1:
                    textture = 1
                Label.append(textture)
            Label = np.array(Label)

        with open("pickle/"+str(self.n_class)+"-classes/labelTrain-"+str(self.cube_size)+".pkl","wb") as file:
            pickle.dump(Label,file)

class TestSet():
    def __init__(self,cube_size = 80,n_class=3):
        self.cube_size = cube_size
        self.n_class = n_class

    def createTestCSV(self):
        print("creating testSet ...") 
        csvLines = readCsv("csv/predictedNodulesC.csv")
        header = csvLines[0]
        nodules = csvLines[1:]
        testSet = pd.DataFrame(columns=['LNDbID', 'FindingID', 'x', 'y', 'z', 'Text'])
        for n in tqdm(nodules):
            lnd = int(n[header.index("LNDbID")])
            find = int(n[header.index("FindingID")])
            #### get coordination ####
            x = float(n[header.index("x")])
            y = float(n[header.index("y")])
            z = float(n[header.index("z")])
            texture = np.argmax([float(n[5]),float(n[6]),float(n[7])])
            
            ctr = np.array([x,y,z])
            [_,spacing,origin,transfmat] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
            transfmat_toimg,_ = getImgWorldTransfMats(spacing,transfmat)
            ctr = convertToImgCoord(ctr,origin,transfmat_toimg)
            testSet = testSet.append({'LNDbID':lnd, 'FindingID':find, 'x':int(ctr[0]), 'y':int(ctr[1]), 'z':int(ctr[2]), 'Text':texture},ignore_index=True)
        testSet = testSet.sample(frac=1).reset_index(drop=True)
        testSet.to_csv("csv/testSet.csv",index=False)
    
    def getTestNonSegmentation(self):
        print("getting test non segmentation ...")
        csvLines = readCsv("csv/testSet.csv")
        header = csvLines[0]
        nodules = csvLines[1:]

        testSet = []

        for n in tqdm(nodules):
            lnd = int(n[header.index("LNDbID")])
            x = int(n[header.index("x")])
            y = int(n[header.index("y")])
            z = int(n[header.index("z")])

            #### get cube ####
            ctr = np.array([x,y,z])
            [scan,spacing,_,_] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
            cube = extractCube(scan,spacing,ctr,self.cube_size)

            #### create slices and segment lung ####
            sliceX = cube[int(cube.shape[0]/2),:,:]
            sliceY = cube[:,int(cube.shape[1]/2),:]
            sliceZ = cube[:,:,int(cube.shape[2]/2)]
            nodule = np.array([sliceX,sliceY,sliceZ])
            
            #### show slices ####
            # fig,axs = plt.subplots(1,3)
            # axs[0].imshow(sliceX)
            # axs[1].imshow(sliceY)
            # axs[2].imshow(sliceZ)
            # plt.show()

            testSet.append(nodule)
        testSet = np.array(testSet)
        
        with open("pickle/"+str(self.n_class)+"-classes/nonSegTestSet-"+str(self.cube_size)+".pkl","wb") as file:
            pickle.dump(testSet,file)
    
    
    def getTestSegmented(self):
        print("getting test segmented")
        csvLines = readCsv("csv/testSet.csv")
        header = csvLines[0]
        nodules = csvLines[1:]
        testSet = []

        for n in tqdm(nodules):
            lnd = int(n[header.index("LNDbID")])
            x = int(n[header.index("x")])
            y = int(n[header.index("y")])
            z = int(n[header.index("z")])

            #### get cube ####
            ctr = np.array([x,y,z])
            [_,spacing,_,_] = readMhd('data/LNDb-{:04}.mhd'.format(lnd))
            with open('pickle/LNDb-{:04}.pkl'.format(lnd),"rb") as file:
                scan = pickle.load(file)
            cube = extractCube(scan,spacing,ctr,self.cube_size)

            #### create slices and segment lung ####
            sliceX = cube[int(cube.shape[0]/2),:,:]
            sliceY = cube[:,int(cube.shape[1]/2),:]
            sliceZ = cube[:,:,int(cube.shape[2]/2)]
            nodule = np.array([sliceX,sliceY,sliceZ])
            
            #### show slices ####
            # fig,axs = plt.subplots(1,3)
            # axs[0].imshow(sliceX)
            # axs[1].imshow(sliceY)
            # axs[2].imshow(sliceZ)
            # plt.show()

            testSet.append(nodule)
        testSet = np.array(testSet)
        
        with open("pickle/"+str(self.n_class)+"-classes/segmentedTestSet-"+str(self.cube_size)+".pkl","wb") as file:
            pickle.dump(testSet,file)

    def getLabel(self):
        print("getting label of test set...")
        csvLines = readCsv("csv/testSet.csv")
        header = csvLines[0]
        nodules = csvLines[1:]
        
        Label = []
        if self.n_class==3:
            print("label for "+str(self.n_class)+"class\n")
            for n in tqdm(nodules):
                textture = int(n[header.index("Text")])
                Label.append(textture)
            Label = np.array(Label)
        else:
            print("label for "+str(self.n_class)+"class\n")
            for n in tqdm(nodules):
                textture = int(n[header.index("Text")])
                if textture >= 1:
                    textture = 1
                Label.append(textture)
            Label = np.array(Label)
        with open("pickle/"+str(self.n_class)+"-classes/labelTest-"+str(self.cube_size)+".pkl","wb") as file:
            pickle.dump(Label,file)


def distort(image, sin=1, cos=0):
    A = image.shape[0] / 40
    w = 2 / (1.5 * image.shape[1])
    rs = np.zeros((image.shape[0],image.shape[1]))
    if cos == 1:
        shift = lambda x: A * np.cos(4.5*np.pi*x * w)
    elif sin == 1:
        shift = lambda x: A * np.sin(4.5*np.pi*x * w)
    for i in range(image.shape[1]):
        rs[:,i] = np.roll(image[:,i], int(shift(i)))
    return np.uint8(rs)

def convertTo3Channel(dataSet):
    print("Convert to 3 channel... \n")
    data3Channel = []
    for d in tqdm(dataSet):
        sliceX = np.dstack([d[0],d[0],d[0]])
        sliceY = np.dstack([d[1],d[1],d[1]])
        sliceZ = np.dstack([d[2],d[2],d[2]])

        data3Channel.append([sliceX,sliceY,sliceZ])

    data3Channel = np.array(data3Channel)
    return data3Channel

def shuffle(dataSet,label):
    import random
    index = list(range(len(dataSet)))

    random.shuffle(index)
    Xtrain = []
    Ytrain = []
    for i in index:
        Xtrain.append(dataSet[i])
        Ytrain.append(label[i])
    
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    return Xtrain,Ytrain

def getTrainValSet(Xtrain,Ytrain,rate):
    Xval = Xtrain[int(len(Xtrain)*rate):]
    Yval = Ytrain[int(len(Ytrain)*rate):]

    Xtrain = Xtrain[:int(len(Xtrain)*rate)]
    Ytrain = Ytrain[:int(len(Ytrain)*rate)]

    return Xtrain,Ytrain,Xval,Yval

def getIndexsData(listLabel, label):
    indexs = []
    for i in range(len(listLabel)):
        if listLabel[i] == label:
            indexs.append(i)
    return indexs

####### original #########

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def writeCsv(csfname,rows):
    # write csv from list of lists
    with open(csfname, 'w', newline='') as csvf:
        filewriter = csv.writer(csvf)
        filewriter.writerows(rows)
        
def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan,spacing,origin,transfmat

def writeMhd(filename,scan,spacing,origin,transfmat):
    # write mhd/raw image
    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)    

def getImgWorldTransfMats(spacing,transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3],transfmat[3:6],transfmat[6:9]])
    for d in range(3):
        transfmat[0:3,d] = transfmat[0:3,d]*spacing[d]
    transfmat_toworld = transfmat #image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat) #world to image coordinates conversion matrix
    
    return transfmat_toimg,transfmat_toworld

def convertToImgCoord(xyz,origin,transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg,xyz))    
    return xyz
    
def convertToWorldCoord(xyz,origin,transfmat_toworld):
    # convert image to world coordinates
    xyz = np.matmul(transfmat_toworld,xyz)
    xyz = xyz + origin
    return xyz

def extractCube(scan,spacing,xyz,cube_size=80,cube_size_mm=51):
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2,1,0]],np.int)
    spacing = np.array([spacing[i] for i in [2,1,0]])
    scan_halfcube_size = np.array(cube_size_mm/spacing/2,np.int)
    if np.any(xyz<scan_halfcube_size) or np.any(xyz+scan_halfcube_size>scan.shape): # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan,((maxsize,maxsize,)),'constant',constant_values=0)
        xyz = xyz+maxsize
    
    scancube = scan[xyz[0]-scan_halfcube_size[0]:xyz[0]+scan_halfcube_size[0], # extract cube from scan at xyz
                    xyz[1]-scan_halfcube_size[1]:xyz[1]+scan_halfcube_size[1],
                    xyz[2]-scan_halfcube_size[2]:xyz[2]+scan_halfcube_size[2]]

    sh = scancube.shape
    scancube = zoom(scancube,(cube_size/sh[0],cube_size/sh[1],cube_size/sh[2]),order=2) #resample for cube_size
    
    return scancube




if __name__ == "__main__":
    #Extract and display cube for example nodule
    lnd = 1
    rad = 1
    finding = 1
    # Read scan
    [scan,spacing,origin,transfmat] =  readMhd('data/LNDb-{:04}.mhd'.format(lnd))
    print(spacing,origin,transfmat)
    # Read segmentation mask
    [mask,spacing,origin,transfmat] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))
    print(spacing,origin,transfmat)
    # Read nodules csv
    csvlines = readCsv('trainNodules.csv')
    header = csvlines[0]
    nodules = csvlines[1:]
    for n in nodules:
        if int(n[header.index('LNDbID')])==lnd and int(n[header.index('RadID')])==rad and int(n[header.index('FindingID')])==finding:
            ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
            break
    
    # Convert coordinates to image
    transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
    ctr = convertToImgCoord(ctr,origin,transfmat_toimg)
    
    # Display nodule scan/mask slice
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(scan[int(ctr[2])])
    axs[1].imshow(mask[int(ctr[2])])
    plt.show()
    
    # Extract cube around nodule
    scan_cube = extractCube(scan,spacing,ctr)
    mask[mask!=finding] = 0
    mask[mask>0] = 1
    mask_cube = extractCube(mask,spacing,ctr)
    
    # Display mid slices from resampled scan/mask
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
    axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
    axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
    axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
    axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
    axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
    plt.show()
    
    