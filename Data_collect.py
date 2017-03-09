import numpy as np
import argparse
import sys
import os
import pickle
from scipy.misc import imread, imresize
from skimage.color import gray2rgb 
np.random.seed(1337)

def main(argv):
    parser = argparse.ArgumentParser('Collect data from images using text files')
    
    parser.add_argument("data_location", help="directory of image data")
    parser.add_argument("text", help="text file containing list of images")
    parser.add_argument("--train", help="For training data", default = False, action = "store_true")
    parser.add_argument("--test", help="For testing data", default = False, action="store_true")
    parser.add_argument("--validation", help="For validation data", default = False, action="store_true")
    parser.add_argument("--random", help="Distribute whole data randomly", default=False, action="store_true")
       
    args=parser.parse_args()    
    
    f = open(args.text,'r')    
    
    X = []
    y = []        
    
    for i in f.readlines():
        im = imresize(imread(os.path.join(args.data_location,i.split()[0])), (224,224)).astype(np.float32)
        if im.ndim == 2:
            im = gray2rgb(im)            
        
        im = im.transpose(2,0,1)    #(channel, width, height)    
        cls = i.split()[1]            
            
        X.append(im)
        y.append(cls)
        print("currnet no. of images: {}".format(len(X)))
            
    X = np.array(X)
    y = np.array(y)        
            
    if args.train:
        pickle.dump((X,y), open("train_data.p","wb"))
    elif args.test:
        pickle.dump((X,y), open("test_data.p","wb"))
    elif args.validation:
        pickle.dump((X,y), open("valid_data.p","wb"))
    elif args.random:
        indx=dice(X.shape[0])  #select train_ratio, test_ratio, valide_ratio      
        pickle.dump(((X[indx==0][:][:],y[indx==0]),(X[indx==1][:][:],y[indx==1]),(X[indx==2][:][:],y[indx==2])), open("all_data.p","wb"))
    else:         
        assert 1, 'Choose mode'    

def dice(size, train_ratio = 0.6, test_ratio=0.2, valid_ratio=0.2):
    indx = []
    for p in np.random.rand(size):
        if p <= train_ratio:
            indx.append(0)
        elif p<= train_ratio+test_ratio:
            indx.append(1)
        else:
            indx.append(2)    
            
    return np.array(indx)

if __name__ == '__main__':
    main(sys.argv)


