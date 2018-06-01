import cv2, sys, numpy, os,time
def train_lbph(images):
    images_lbp=localBinaryPattern(images)
    images_histogramed=getHistograms(images_lbp)
    return images_histogramed
    
def localBinaryPattern_wh(images):
    (s1,s2,s3)=images.shape
    images_processed=numpy.zeros((s1,(s2-4),(s3-4)))
    window=3
    for i in range(s1):
        temp2=numpy.zeros((s2,s3))
        for j in range(s2-window-1):
            for k in range(s3-window-1):
                temp2[i,j:j+window,k:k+window]=images[i,j:j+window,k:k+window]
                temp2[i,j:j+window,k:k+window]=getPattern_wh(temp2[i,j:j+window,k:k+window])
        images_processed[i]=temp2
    return images_processed

def localBinaryPattern(images):
    (s1,s2,s3)=images.shape
    images_processed=numpy.zeros((s1,(s2-4),(s3-4)))
    count=0
    temp1=numpy.zeros((1,(s2-4)*(s3-4)))
    temp2=numpy.zeros((s2,s3))
    window=3
    for i in range(s1):
        count=0
        temp1=numpy.zeros((1,(s2-4)*(s3-4)))
        for j in range(s2-window-1):
            for k in range(s3-window-1):
                temp2=images[i,j:j+window,k:k+window]
                temp1[0,count]=int(getPattern(temp2))
                count=count+1
        temp1=temp1[0].reshape((s2-4),(s3-4))
        images_processed[i]=temp1
    return images_processed

def bin2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def getElementsClockwise(mat):
    (d,r)=mat.shape
    l=0
    u=0
    count=0
    theta=numpy.zeros((1,r*d))
    direction=0
    while  l<=(r-1) and u<=(d-1):
        if direction==0:
            for i in range(l,r):
                theta[0,count]=mat[u,i]
                count=count+1
            u=u+1
            direction=direction+1
        elif direction==1:
            for i in range(u,d):
                theta[0,count]=mat[i,r-1]
                count=count+1
            r=r-1
            direction=direction+1
        elif direction==2:
            for i in range(r-1,l-1,-1):
                theta[0,count]=mat[d-1,i]
                count=count+1
            d=d-1
            direction=direction+1
        elif direction==3:
            for i in range(d-1,u-1,-1):
                theta[0,count]=mat[i,l]
                count=count+1
            l=l+1
            direction=direction+1
        if direction%4 ==0:
            direction=0    
    return theta

def getPattern_wh(im_portion):
    temp=numpy.zeros((3,3))
    #im_portion1=getElementsClockwise(im_portion)
    for i in range(3):
        for j in range(3):
            if im_portion[i,j]>=im_portion[1,1]:
                temp[i,j]=1
    return temp
    
def getPattern(im_portion):
    temp=numpy.zeros((1,8))
    im_portion1=getElementsClockwise(im_portion)
    for i in range(8):
        if im_portion1[0,i]<im_portion1[0,8]:
            temp[0,i]=1
    temp_int=temp.tolist()
    temp2_bool=[bool(m) for m in temp_int[0]]
    temp2=bin2int(temp2_bool[::-1])
    return temp2

def getHistograms(images_lbp):
    (s1,s2,s3)=images_lbp.shape
    sd=(int)((s2/8)*(s3/8)*256)
    result=numpy.zeros((s1,sd))
    for i in range(s1):
        result[i]=getVals(images_lbp[i],s2,s3)
    return result

def getVals(img_por,s2,s3):
    sd=(int)((s2/8)*(s3/8))
    temp=numpy.zeros((sd,256))
    idx=0
    for j in range(0,s2-8,8):
        for k in range(0,s3-8,8):            
            unique,counts=numpy.unique(img_por[j:j+8,k:k+8].reshape(1,8*8,order='F'),return_counts=True)
            unique_int=[int(l) for l in unique.tolist()]
            temp[idx,unique_int]=counts
            idx=idx+1
    return temp.reshape((1,sd*256))

def predict_lbph(input_image,recognizer,labels):
    (s1,s2)=input_image.shape
    (d1,d2)=recognizer.shape
    temp=numpy.zeros((1,s1,s2))
    temp[0]=input_image
    input_histogramed=train_lbph(temp)
    (minval,index,distance)=(10000,0,0)
    for i in range(d1):
        distance=numpy.linalg.norm(recognizer[i,:]-input_histogramed)
        if distance<minval:
            index=i
            minval=distance
    return (labels[index],minval)

def predict_lbph_multi(input_image,recognizer,labels):
    (s3,s1,s2)=input_image.shape
    (d1,d2)=recognizer.shape
    predictions=numpy.zeros((s3))
    minval_confidence = numpy.zeros((s3))
    input_histogramed=train_lbph(input_image)
    for j in range(s3):
        (minval, index, distance) = (10000, 0, 0)
        for i in range(d1):
            distance=numpy.linalg.norm(recognizer[i,:]-input_histogramed[j,:])
            if distance<minval:
                index=i
                minval=distance
        predictions[j]=labels[index]
        minval_confidence[j]=minval
    return (predictions)