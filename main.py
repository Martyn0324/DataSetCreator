import os
import numpy as np
from PIL import Image

class DatasetCreator():
    '''A shortcut to create your own datasets. Developed with focus on Machine Learning, specially Neural Networks, but perhaps might be useful
    for Data Science in general.
    None of its operations are complex, they're just kind of boring and might make your code polluted.'''
    
    def __init__(self):
        pass
        
    def images(self, path, width, height, return_rgba=False): # I don't know if this order is correct. In ML, we normally use width = height.
        '''Returns an array with shape (n_samples, width, height, n_channels). Created based on MNIST dataset(which has this shape format).
        
        path: Path where you got the images saved.
        return_rgba: if False, all RGBA images will be converted to RGB. Default: False.
        '''
        pics = []
        
        for directory, _, files in os.walk(path):
            for file in files:
                pics.append(directory+'/'+file)
            
        # Removing any files that aren't images
        pics = [i for i in pics if ('.jpg', '.png') in i]
        
        pics = [Image.open(i) for i in pics]
        
        for i in range(len(pics)):
            pics[i] = pics[i].resize((width, height))
        if pics[i].mode == 'RGBA' and return_rgba == False:
            pics[i] = pics[i].convert('RGB')
        
        # Converting PIL arrays back to numpy arrays
        pics = [np.array(i) for i in pics]
        
        pics = np.array(pics)
        
        # Now, we got an array with shape(n_samples,), where each sample got (width, height, n_channels)
        
        pics = np.stack(pics, axis=0)
        
        # NOW, yes, we got an array with shape (n_samples, width, height, n_channels)
        
        print(f'dataset shape: {pics.shape}\nSamples shape: {pics[0].shape}')
        
        return pics
    
    def audio(self):
        # TODO - Maybe if someday I learn how to create an audio dataset...
        raise NotImplementedError
    
    def preprocess(self, dataset, type='image'):
        '''Normalizes the dataset, returning it with values between -1 and 1.
        If you want to plot the images, you need to denormalize it using deprocess()'''
        if type in ('image', 'img'):
            dataset = dataset/127.5 - 1.0

            print(f"Dataset type: {type(dataset)}, {dataset.dtype}")

            return dataset
        
        elif type == 'audio':
            raise NotImplementedError

    def deprocess(self, dataset, type='image', normalized=True):
        '''Returns the dataset in a way that it can be plotted(images) or even denormalized(normalized=False)'''
        if type in ('image', 'img'):
            if normalized:
                dataset = (dataset+1.0)*0.5
                print('dataset values are now between 0 and 1')

            else:
                dataset = (dataset+1.0)*127.5
                print('dataset values are now between 0 and 255')
            
            return dataset
        
        elif type == 'audio':
            raise NotImplementedError

        
    
    def save_dataset(self, save_path, dataset_name):
        '''In case you forgot how to save a numpy array using np.save()'''
        path_string = save_path + '/' + dataset_name
        
        np.save(path_string, dataset_name)

    def load_dataset(self, load_path, dataset_name):
        '''In case you forgot how to load a numpy array using np.load()'''
        dataset = np.load(load_path + '/' + dataset_name + '.npy')

        return dataset
