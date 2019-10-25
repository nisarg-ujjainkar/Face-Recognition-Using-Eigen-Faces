import numpy as np
import os
import cv2
import sys
from sklearn.preprocessing import normalize

def read_dataset():
    '''this reads training images flattens them and created a 2D matrix of all images'''
    #images = np.empty([0,0])
    images=[]
    for i in range(1,41):
        folder = os.getcwd()+'/s'+str(i)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.flatten()#'''[0]*img.shape[1]''')
                images.append(img)
    #print(img.shape)
    images = np.asarray(images)#.astype(np.float64)
    return images.transpose()


def eig_faces(images,k):
    '''this function takes matrix containing training images and calculates mean, k best eig vectors
    and projects the image data into that eigen vector space'''
    #print(images.shape)
    mean = np.mean(images,axis=1).reshape(images.shape[0],1)
    images = images-mean
    #print('shifted images',images)
    #print('mean',mean)
    cov = np.matmul(images.transpose(),images)
    w,v = np.linalg.eig(cov)
    u = np.matmul(images,v)
    #u=[]
    #for i in v:
    #    u.append(np.dot(images,i))
    #u=np.asarray(u)
    #print('shape of u',u.shape)
    nu = normalize(u, axis=1, norm='l2')
    indexes = np.argsort(w)[::-1][:k]
    k_eig_vect = nu.transpose()[indexes]
    eig_face = np.matmul(k_eig_vect,images)
    #eig_face=[]
    #for val in k_eig_vect:
    #    eig_face.append(np.dot(val,images))
    #eig_face=np.asarray(eig_face)
    #print("Eigen faces",eig_face)            #this is somehow wrong
    #print("selected vectors",(k_eig_vect.transpose()))     #this also is wrong
    return eig_face,mean,k_eig_vect

def read_test_data(mean,k_eig_vect):
    '''This functionreads all the test images and projectsit into K_eigen vector space'''
    test_img_path = os.getcwd()+'/test_images/s'
    test_data = []
    for i in range(1,41):
        temp_path = test_img_path+str(i)
        image = cv2.imread(os.path.join(temp_path,os.listdir(temp_path)[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()
        test_data.append(image)
    test_data = np.asarray(test_data).transpose()#.astype(np.float64).transpose()
    test_data = test_data-mean
    #print("test data",test_data)
    #test_data=test_data.transpose()
    #print(test_data.shape)
    test_data = np.matmul(k_eig_vect,test_data)
    #test_d=[]
    #for i in test_data:
    #    test_d.append(np.dot(k_eig_vect,i))
    #return np.asarray(test_d)#ata
    #print("test data",(test_data.transpose()))
    return test_data
def read_test_img(mean,s,k_eig_vect):
    '''if a particular source is given for testing, this function reads it and projects it into 
    the K_eigen vector space'''
    path = os.getcwd()+'/s'+str(s)
    path = os.path.join(path,os.listdir(path)[0])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.flatten()#'''[0]*img.shape[1]''')
    image = image.reshape(image.shape[0],1)
    #print(image.shape,mean.shape)
    image = image-mean
    #print(k_eig_vect.shape,image.shape)
    image = np.matmul(k_eig_vect,image)
    #print("test data",image)
    return image.reshape(1,image.shape[0])
def main():
    #print(len(sys.argv))
    if(len(sys.argv)==1):
        print("No arguements provided running test on all test cases")
    elif(int(sys.argv[1])<=40 and int(sys.argv[1])>=1):
        print("Running test on test image set s%d" %int(sys.argv[1]))
    else:
        print("Wrong arguement provided")
        exit(1)
    flag = len(sys.argv)
    images = read_dataset()
    eig_face,mean,k_eig_vect = eig_faces(images,20)
    #print(eig_face,mean)
    #print(images.shape,eig_face.shape)
    threshold_value = np.inf
    #min_dist = np.inf
    #min_dindex = -1
    eig_face = eig_face.transpose()
    #print(eig_face)
    if(flag==1):
        test_data = read_test_data(mean,k_eig_vect)
        test_data = test_data.transpose()
        #print("test data",test_data.shape)
        #print("eig face",eig_face.shape)
        indices=[]
        for val in test_data:
            #val=val.reshape(1,val.shape[0])
            """ for index in range(len(eig_face)):
                norm=np.linalg.norm(eig_face[index]-val)    #here I was trying to calculate the
                if norm < min_dist:                         #index for the min norm manually, which 
                    if norm < threshold_value:              #for some reason does not work.
                        min_dist = norm
                        min_dindex = index """
            comp = eig_face-val.reshape(1,val.shape[0])
            norm = np.linalg.norm(comp,axis=1)
            index = np.argmin(norm)
            if(norm[index]<threshold_value):
                indices.append(index)
                #print(val - eig_face[index])
                #print(min_dist)
            #print(int(min_dindex/9)+1,min_dist)
            #indices.append(min_dindex)
        print([(int(x/9)+1) for x in indices])
    else:
        s=int(sys.argv[1])
        img = read_test_img(mean,s,k_eig_vect)
        comp = eig_face-img
        norm = np.linalg.norm(comp,axis=1)
        index = np.argmin(norm)
        print("the image matches to set S%d" %(int(index/9)+1))
        """ for index in range(len(eig_face)):
            if np.linalg.norm(img - eig_face[index]) < min_dist:
                if np.linalg.norm(img - eig_face[index]) < threshold_value:
                    min_dist = np.linalg.norm(img - eig_face[index])
                    min_dindex = index """
        
        #print(int(min_dindex/9)+1,min_dist)


if __name__=="__main__":
    main()