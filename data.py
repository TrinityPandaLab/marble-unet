from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage
import math
from skimage.util import img_as_ubyte
import random
import skimage.io as io
import skimage.transform as trans
import shutil
from natsort import natsorted

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,save_to_dir,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,target_size = (256,256),seed = 0):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image,target_size = (256,256),flag_multi_class = False,as_gray = False):
    '''
    Generate the testing images from the specified path,
    resize them to 256*256,
    number of image specified by the user
    set as_gray as False if they are RGB images
    '''
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.tif"%i),as_gray = as_gray)
        #img = img / 255
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

#def resultGenerator(test_path, num_image = 30, target_size = (256,256), flag_multi_class = False, as_gray = False):
#    '''
#    Generate the 30 resulting images from the specified path,
#    resize them to 256*256
#    set as_gray as False if they are RGB images
#    '''
#    for i in range(num_image):
#        img = io.imread(os.path.join(test_path,"%d.tif"%i),as_gray = as_gray)
#        #img = img / 255
#        img = trans.resize(img,target_size)
#        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#        img = np.reshape(img,(1,)+img.shape)
#        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    '''
    from the original image path, read images and convert them into numpy arrays
    change the file extensions as needed in this following line.
    '''
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.bmp"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def kfolderGenerator(k_value,image_path,label_path):
    '''
    Generate the k-folders according to k_value following the concept of k-fold cross validation process
    '''
    count = 0
    count2 = 0
    wkdir = ''
    for path in os.listdir(image_path):
        if os.path.isfile(os.path.join(image_path, path)):
            count += 1
    for path in os.listdir(label_path):
    	if os.path.isfile(os.path.join(label_path, path)):
    	    count2 += 1
    print('The image folder has:', count)
    print('The label folder has:', count2)
    if count == count2:
        item_n = math.floor(count/k_value)
        os.mkdir('Kfolder')
        wkdir = os.path.join(wkdir,'Kfolder')
        print('Temporary folder: Kfolder created')
        imagelis = os.listdir(image_path)
        for i in range(k_value):
            os.mkdir(os.path.join(wkdir,str(i)))
            temp_dir = os.path.join(wkdir,str(i))
            train_dir = os.path.join(temp_dir,'train')
            test_dir = os.path.join(temp_dir,'test')
            os.mkdir(train_dir)
            os.mkdir(test_dir)
            os.mkdir(os.path.join(test_dir,'correctans'))
            os.mkdir(os.path.join(train_dir,'image'))
            os.mkdir(os.path.join(train_dir,'labels'))
            os.mkdir(os.path.join(train_dir,'aug'))
            if i == k_value-1:
            	list_piece = imagelis[i*item_n:]
            else:
            	list_piece = imagelis[i*item_n:i*item_n+item_n]
            diff = set(imagelis).difference(set(list_piece))
            for item in list_piece:
                shutil.copy(os.path.join(image_path,item),test_dir)
                shutil.copy(os.path.join(label_path,item),os.path.join(test_dir,'correctans'))
            for count, filename in enumerate(natsorted(os.listdir(test_dir))):
                if filename != 'correctans':
                    dst = f"{count}{'.tif'}"
                    src1 =f"{test_dir}/{filename}"
                    src2 =f"{test_dir}/{'correctans'}/{filename}"
                    dst1 =f"{test_dir}/{dst}"
                    dst2 =f"{test_dir}/{'correctans'}/{dst}"
                    os.rename(src1, dst1)
                    os.rename(src2, dst2)
            for item in diff:
            	shutil.copy(os.path.join(image_path,item),os.path.join(train_dir,'image'))
            	shutil.copy(os.path.join(label_path,item),os.path.join(train_dir,'labels'))
            for count, filename in enumerate(natsorted(os.listdir(os.path.join(train_dir,'labels')))):
                if filename != 'correctans':
                    dst = f"{count}{'.tif'}"
                    src1 =f"{train_dir}/{'image'}/{filename}"
                    src2 =f"{train_dir}/{'labels'}/{filename}"
                    dst1 =f"{train_dir}/{'image'}/{dst}"
                    dst2 =f"{train_dir}/{'labels'}/{dst}"
                    os.rename(src1, dst1)
                    os.rename(src2, dst2)
        os.mkdir(os.path.join(wkdir,'total'))
        os.mkdir(os.path.join(wkdir,'total','ans'))
        os.mkdir(os.path.join(wkdir,'total','image'))
        os.mkdir(os.path.join(wkdir,'total','labels'))
        #print(sample)
    else:
        print('Unmatched number of files.')

def kfolderGenerator_double(k_value,label_path,image_path,marble_path,sawdust_path):
    '''
    Generate the k-folders according to k_value following the concept of k-fold cross validation process
    this method is used specifically for the double-network method
    it puts everything needed to the folders and use more space
    '''
    count = 0
    count2 = 0
    count3 = 0
    wkdir = ''
    # Iterate directory
    for path in os.listdir(image_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(image_path, path)):
            count += 1
    for path in os.listdir(marble_path):
    	if os.path.isfile(os.path.join(marble_path, path)):
    	    count2 += 1
    for path in os.listdir(sawdust_path):
    	if os.path.isfile(os.path.join(sawdust_path, path)):
    	    count3 += 1
    print('The image folder has:', count)
    print('The marble folder has:', count2)
    print('The sawdust folder has:', count3)
    if (count == count2) and (count == count3):
        item_n = math.floor(count/k_value)
        os.mkdir('Kfolder')
        wkdir = os.path.join(wkdir,'Kfolder')
        print('Temporary folder: Kfolder created')
        imagelis = os.listdir(image_path)
        for i in range(k_value):
            os.mkdir(os.path.join(wkdir,str(i)))
            temp_dir = os.path.join(wkdir,str(i))
            train_dir = os.path.join(temp_dir,'train')
            test_dir = os.path.join(temp_dir,'test')
            os.mkdir(train_dir)
            os.mkdir(test_dir)
            os.mkdir(os.path.join(test_dir,'correctans'))
            os.mkdir(os.path.join(train_dir,'image'))
            os.mkdir(os.path.join(train_dir,'marble'))
            os.mkdir(os.path.join(train_dir,'sawdust'))
            os.mkdir(os.path.join(train_dir,'labels'))
            os.mkdir(os.path.join(train_dir,'aug'))
            if i == k_value-1:
            	list_piece = imagelis[i*item_n:]
            else:
            	list_piece = imagelis[i*item_n:i*item_n+item_n]
            diff = set(imagelis).difference(set(list_piece))
            for item in list_piece:
                shutil.copy(os.path.join(image_path,item),test_dir)
                shutil.copy(os.path.join(label_path,item),os.path.join(test_dir,'correctans'))
            for count, filename in enumerate(natsorted(os.listdir(test_dir))):
                if filename != 'correctans':
                    dst = f"{count}{'.tif'}"
                    src1 =f"{test_dir}/{filename}"
                    src2 =f"{test_dir}/{'correctans'}/{filename}"
                    dst1 =f"{test_dir}/{dst}"
                    dst2 =f"{test_dir}/{'correctans'}/{dst}"
                    os.rename(src1, dst1)
                    os.rename(src2, dst2)
            for item in diff:
            	shutil.copy(os.path.join(image_path,item),os.path.join(train_dir,'image'))
            	shutil.copy(os.path.join(marble_path,item),os.path.join(train_dir,'marble'))
            	shutil.copy(os.path.join(sawdust_path,item),os.path.join(train_dir,'sawdust'))
            	shutil.copy(os.path.join(label_path,item),os.path.join(train_dir,'labels'))
            for count, filename in enumerate(natsorted(os.listdir(os.path.join(train_dir,'image')))):
                dst = f"{count}{'.tif'}"
                src1 =f"{train_dir}/{'image'}/{filename}"
                src2 =f"{train_dir}/{'marble'}/{filename}"
                src3 =f"{train_dir}/{'sawdust'}/{filename}"
                src4 =f"{train_dir}/{'labels'}/{filename}"
                dst1 =f"{train_dir}/{'image'}/{dst}"
                dst2 =f"{train_dir}/{'marble'}/{dst}"
                dst3 =f"{train_dir}/{'sawdust'}/{dst}"
                dst4 =f"{train_dir}/{'labels'}/{dst}"
                os.rename(src1, dst1)
                os.rename(src2, dst2)
                os.rename(src3, dst3)
                os.rename(src4, dst4)
        os.mkdir(os.path.join(wkdir,'total'))
        os.mkdir(os.path.join(wkdir,'total','ans'))
        os.mkdir(os.path.join(wkdir,'total','image'))
        os.mkdir(os.path.join(wkdir,'total','marble'))
        os.mkdir(os.path.join(wkdir,'total','sawdust'))
        os.mkdir(os.path.join(wkdir,'total','labels'))
        #print(sample)
    else:
        print('Unmatched number of files.')

def saveResult(current_kfold,save_path,npyfile,flag_multi_class = False,num_class = 2):
    '''
    Save the result from the generated numpy files, into the current k-folder
    '''
    wkpath = os.path.join('Kfolder',str(current_kfold))
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
    for item in os.listdir(os.path.join(wkpath,'test')):
        if item != 'correctans':
            shutil.copy(os.path.join(wkpath,'test',item),os.path.join('Kfolder','total',str(current_kfold)+'_'+item))
            if item[-4:] == '.tif':
                shutil.copy(os.path.join(wkpath,'test','correctans',item),os.path.join('Kfolder','total','ans',str(current_kfold)+'_'+item))


def saveResult_double(current_kfold,is_marble,save_path,npyfile,flag_multi_class = False,num_class = 2):
        '''
        Save the result from the generated numpy files, into the current k-folder
        this method is used specifically for the double-network method
        '''
    wkpath = os.path.join('Kfolder',str(current_kfold))
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
    for item in os.listdir(os.path.join(wkpath,'test')):
        if item != 'correctans':
            if is_marble:
                shutil.copy(os.path.join(wkpath,'test',item),os.path.join('Kfolder','total','marble',str(current_kfold)+'_'+item))
                if item[-4:] == '.tif':
                    shutil.copy(os.path.join(wkpath,'test','correctans',item),os.path.join('Kfolder','total','ans',str(current_kfold)+'_'+item))
            else:
                shutil.copy(os.path.join(wkpath,'test',item),os.path.join('Kfolder','total','sawdust',str(current_kfold)+'_'+item))
