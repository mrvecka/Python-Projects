import numpy as np
import _pickle as cp
import Common as com
import cv2

np.random.seed(1)

def CreateDataSets(path,mode,trainDataInPercent):
    imageData,labeldata = LoadData(path,mode)
    # train_data_x = np.empty((200,3072)) #200 train images
    #train_data_y = np.empty((1,200),dtype = np.int16)    #200 train labels
    j = 0
    l = 0
    i = 0
    cats = []    
    catsLabels = []
    noCats = []
    noCatsLabels = []
    imsize = 3072
    numOfFiles = len(imageData) -1
    f = 0
    for f in range(5000):
        img = imageData[f]
        label =labeldata[f]
        if (label == 1 or label == 2):
            cats.append(com.Reshape(img,1,imsize))
            cats.append(com.Reshape(np.array(cv2.rotate(img,cv2.ROTATE_180)),1,imsize))
            cats.append(com.Reshape(np.array(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)),1,imsize))
            labelarray = []
            if (labeldata[f] == 1):
                labelarray = [1,0]
            elif (labeldata[f] == 2):
                labelarray = [0,1]
            else:
                {}
            catsLabels.append(labelarray)
            catsLabels.append(labelarray)
            catsLabels.append(labelarray)
 
    train_data_x= np.array(cats[:int(len(cats) * (trainDataInPercent)/100)])
    train_data_y = np.array(catsLabels[:int(len(catsLabels) * (trainDataInPercent)/100)])

    test_data_x = np.array(cats[int(len(cats) * (trainDataInPercent)/100):])
    test_data_y = np.array(catsLabels[int(len(catsLabels) * (trainDataInPercent)/100):])

    #add some none cats images to train and test data    
    # train_data_x = np.vstack([train_data_x,np.array(noCats[:int(len(noCats) * (trainDataInPercent)/100)])])
    # train_data_y = np.append(train_data_y,np.array(noCatsLabels[:int(len(noCatsLabels) * (trainDataInPercent)/100)]),axis=0)

    # test_data_x = np.vstack([test_data_x,np.array(noCats[int(len(noCats) * (trainDataInPercent)/100):])])
    # test_data_y = np.append(test_data_y,np.array(noCatsLabels[int(len(noCatsLabels) * (trainDataInPercent)/100):]),axis=0)

    train_data_x = train_data_x.reshape(train_data_x.shape[0],3072)
    test_data_x = test_data_x.reshape(test_data_x.shape[0],3072)
        
    return train_data_x, train_data_y,test_data_x,test_data_y          
        


def LoadData(path,mode):

    numOfFiles = 1
    images = []
    labels = []    
    while (com.FileExist(path+ str(numOfFiles))):
        f = open(path+str(numOfFiles), 'rb')
        dict = cp.load(f,encoding='latin1')
        f.close()
        images.append(dict['data'])
        #images = np.reshape(images, (10000, 3, 32, 32))
        labels.append(dict['labels'])
        numOfFiles+=1

    g_images = []

    imagesArray = np.array(images)   #   (5, 10000, 3072)
    imagesArray = com.Reshape(imagesArray,imagesArray.shape[0]*imagesArray.shape[1],imagesArray.shape[2]) # 50000 , 3072
    # g_images = []
    # i = 0
    # for i in range(imagesArray.shape[0]):
    #     img = imagesArray[i]
    #     img = com.GetGraySkaledImage(img)
    #     img = com.Normalize_0to1(img)
    #     g_images.append(img)


    # images = np.array(g_images)
    # images = images.reshape((images.shape[0],1024))
    
    labelsArray = np.array(labels)   #   (10000,)
    labelsArray = np.reshape(labelsArray,(labelsArray.shape[0]*labelsArray.shape[1]))
    return imagesArray, labelsArray
        
def LoadCustomImages(path,labelsPath = ""):
    files = com.GetAllImagesOnPath(path)
    imagesArray = np.empty((len(files),3072))
    labelsArray = []
    i = 0
    for f in files:
        img = cv2.imread(path + "\\" + f)
        result = GetImageAsArray(img)        
        imagesArray[i] = result
        labelsArray.append(f)
        i+=1

    if labelsPath == None:
        final = (imagesArray,labelsArray)
    else:
        labels = com.GetLabelsFromFile(labelsPath)
        final = (imagesArray,labels)

    print(final)

def GetImageAsArray(img):
    """
    Methode return array of image data, image will be resized to 32*32
    Array will be of shape (1,x) x: img.height * img.width * 3

    @param: img Source image 
    """
    img2 = cv2.resize(img,(32,32),interpolation = cv2.INTER_AREA)
    red = np.array
    green = np.array
    blue = np.array
    for x in img2:
        red = np.append(red,x[:,0])
        green = np.append(green,x[:,1])
        blue = np.append(blue,x[:,2])   
    
    result = np.append(red[1:],green[1:])
    result = np.append(result,blue[1:])

    return result

        
            

    #cv2.imshow("loaded image", img2)
    #k = cv2.waitKey(0)
    #lineImage = cv2.PCA_DATA_AS_ROW(img2)
    #print (lineImage)
    #cv2.destroyAllWindows()
    
    


