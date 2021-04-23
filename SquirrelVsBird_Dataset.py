import os
import cv2 
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle 

#Method for loading and scaling images
#input params:
    #folder - root image directory
    #name_list - list of folders below the root directory that should be included in the dataset
    #label - the label that will be given to these images
    #img_size = the size the image will be scaled to, currently makes the image square. Default value is 224.
def LoadAndScaleImages(folder, name_list, bin_label, label, img_size = 64):
    data_samples = []
    data_labels = []
 
    for name in name_list:

        #retrieve image folder paths
        img_folders = glob.glob(folder+'*'+name+'*//')

        for img_folder in img_folders:
            #retrieve image files paths in the folder
            img_names = glob.glob(img_folder+'*')

            #iterate through the images
            for img_name in img_names:
                img = cv2.imread(img_name)

                #To prevent distortion, the image will be padded along each dimension until it can be divided evenly by the desired image size.
                h,w,c = img.shape
                max_dem = max([h,w])
                r = max_dem % img_size
                new_dem = max_dem + (img_size - r)
                pad_top = math.ceil((new_dem - h)/2)
                pad_bottom = math.floor((new_dem - h)/2)
                pad_left = math.ceil((new_dem - w)/2)
                pad_right = math.floor((new_dem - w)/2)

                #Use OpenCV to resize then scall the image
                resized_img = cv2.copyMakeBorder(img,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT)
                scaled_img = cv2.resize(resized_img,(img_size,img_size))

                #append the image and labels to arrays 
                file_count = len(data_samples)
                file_name = f'{label}_{file_count:05}.jpg'
                data_samples.append(scaled_img)
                data_labels.append([file_name,bin_label])

    #shuffle the samples prior to returning. The image and label lists are zipped to maintain their relationship before being randomized.
    zipped_data = list(zip(data_samples, data_labels))
    random.shuffle(zipped_data)
    shuffled_samples, shuffled_labels = zip(*zipped_data)
    return shuffled_samples, shuffled_labels

#Creates the destination folder for the validation set
#input params:
    #samples - list of images
    #labels - list of image labels
    #dest_dir - the destination directory where the validation images and label file will be saved.
def CreateValidationFolder(samples, labels, dest_dir):

    #Create the destination folder if it does not exist.
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    #Changed the excution directory to the desingation folder.
    os.chdir(dest_dir)
    
    #Simultaneously write the image file and save the sample images to a jpg.
    with open('val.txt','w') as label_file:
        for sample, label in zip(samples,labels):
            cv2.imwrite(label[0],sample)
            label_file.write(f'{label[0]} {label[1]}' +'\n')

#Main method, executes dataset amalagamation and transformation process
    #caltech_bird_dir - path of the root folder for the caltech bird dataset
    #animal_10_dir - path of the root folder for the aninal-10 dataset
def Main(caltech_bird_dir,animal_10_dir):
    #set a set a seed for the randomizer to keep randomization consistent between executions.
    random.seed(517)

    #name of the birds that will be included from the Caltech_Birds_2011 dataset
    bird_names = ['Warbler','Wren','Crow','Woodpecker','Swallow','Sparrow','Shrike','Oriole','Mockingbird','Meadowlark','Jay','Goldfinch','Finch','Flycatcher','Cuckoo','Blackbird','Catbird','Cardinal']

    #name of the folder for squirrel images from the Animals-10 dataset
    squirrel_names = ['Scoiattolo']

    #Call the above method to read the images and load them into memory
    bird_samples, bird_labels = LoadAndScaleImages(caltech_bird_dir,bird_names,1,'bird')
    squirrel_samples, squirrel_labels = LoadAndScaleImages(animal_10_dir,squirrel_names,0,'squirrel')

    #There are more bird images than squirrel images in the dataset. To prevent an unbalanced model, we will undersample the bird_samples.
    #15% of the samples will be reserved for a validation set, while the remaining 85% will be used to train the model.
    max_sample_len = min(len(bird_samples),len(squirrel_samples))
    validation_len = int(math.floor(max_sample_len * 0.15))
    training_len = max_sample_len - validation_len

    #Using the lengths above, the lists will be seperated into training and validation sets.
    bird_train_samples = bird_samples[:training_len-1]
    bird_train_labels = bird_labels[:training_len-1]
    bird_val_samples = bird_samples[training_len:training_len + validation_len-1]
    bird_val_labels = bird_labels[training_len:training_len + validation_len-1]


    squirrel_train_samples = squirrel_samples[:training_len-1]
    squirrel_train_labels = squirrel_labels[:training_len-1]
    squirrel_val_samples = squirrel_samples[training_len:training_len+validation_len-1]
    squirrel_val_labels = squirrel_labels[training_len:training_len+validation_len-1]


    #Combine the bird and squirrel sets 
    training_samples = bird_train_samples + squirrel_train_samples
    training_labels = bird_train_labels + squirrel_train_labels
    validation_samples = bird_val_samples + squirrel_val_samples
    validation_labels = bird_val_labels + squirrel_val_labels

    #Shuffle the training_samples to mix the two classes
    zipped_data = list(zip(training_samples, training_labels))
    random.shuffle(zipped_data)
    training_samples, training_labels = zip(*zipped_data)

    #Use the current director as the root for the validation save location folder
    currdir = os.getcwd()
    CreateValidationFolder(validation_samples, validation_labels, currdir + '//SquirrelVsBird_ValidationSet_224//')

    #Reset the execution directory in case it was updated in the CreateValidationFolder method.
    os.chdir(currdir)

    #The training and validation sets are put into a dictionary to be saved as a pickle file.
    #The pickle file will be read by the script for creating the model.
    squirrel_vs_birds_dict = {'training_samples' : training_samples}
    squirrel_vs_birds_dict.update({'training_labels' : training_labels})
    squirrel_vs_birds_dict.update({'validation_samples' : validation_samples})
    squirrel_vs_birds_dict.update({'validation_labels' : validation_labels})

    with open("squirrelvsbirddataset_224.p","wb") as picklefile:
        pickle.dump(squirrel_vs_birds_dict, picklefile)

if __name__ == "__main__":
    #Define root directories for the full datasets
    caltech_bird_dir = ''
    animal_10_dir = ''

    Main(caltech_bird_dir,animal_10_dir)
