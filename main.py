import os
import numpy as np
from PIL import Image
from scipy.io import wavfile

class DatasetCreator():
    '''A shortcut to create your own datasets. Developed with focus on Machine Learning, specially Neural Networks, but perhaps might be useful
    for Data Science in general.
    None of its operations are complex, they're just kind of boring and might make your code polluted.'''
    
    def __init__(self):
        pass
        
    def images(path, width, height, return_rgba=False): # I don't know if this order is correct. In ML, we normally use width = height.
        '''Returns an array with shape (n_samples, width, height, n_channels). Created based on MNIST dataset(which has this shape format).
        
        path: Path where you got the images saved.
        return_rgba: if False, all RGBA images will be converted to RGB. Default: False.
        '''
        pics = []
        
        for directory, _, files in os.walk(path):
            for file in files:
                pics.append(directory+'/'+file)
            
        # Removing any files that aren't images
        pics = [i for i in pics if '.jpg' in i or '.png' in i]
        
        images = []
        for i in pics:
            pic = Image.open(i)
            pic = pic.resize((width, height))
            if pic.mode != 'RGB' and return_rgba == False:
                pic = pic.convert('RGB')
            image = np.array(pic)
            pic.close()
            images.append(image)
        
        pics = images
        
        # Converting list of arrays to array of...arrays.
        
        pics = np.array(pics)
        
        # Now, we got an array with shape(n_samples,), where each sample got (width, height, n_channels)
        
        pics = np.stack(pics, axis=0)
        
        # NOW, yes, we got an array with shape (n_samples, width, height, n_channels)
        
        print(f'dataset shape: {pics.shape}\nSamples shape: {pics[0].shape}')
        
        return pics
    
    def audio(path):
        # I think this can follow the same template as image...at least it worked when I tried...
        '''
        Returns an array with shape (n_samples, audio_data, n_channels). Created based on images dataset.
        Make sure that your audio files are in .wav format, all with the same channel(1, Mono, or 2, stereo) and they all have the same Sample Rate
        
        path: Path where you got the audio saved.
        '''
        audios = []
        
        for directory, _, files in os.walk(path):
            for file in files:
                audios.append(directory+'/'+file)
        
        # Removing any files that aren't .wav audio
        audios = [i for i in audios if ".wav" in i]
        
        audio = []
        for i in audios:
            sample_rate, data = wavfile_read(i)
            data.append(audio)
            
        audio = np.array(audio)
        
        try:
            audio = np.stack(audio, axis=0)
        
        except ValueError: # The audios have different shapes. Padding in order to make sure they all have the same shape.
            max_shape = 0
            for i in audio:
                if i.shape[0] > max_shape
                max_shape = i.shape[0]
            
            for i in audio: # There must be a more efficient way to do this...
                if i.shape[0] < max_shape:
                    pad = max_shape-i.shape[0]
                    i = np.pad(i, [(0, pad), (0,0)]) # Now all audios will have the same shape(AKA duration), only with some silence after the sound itself.
            
            audio = np.stack(audio, axis=0)
            
        
        print(f'dataset shape: {audio.shape}\nSample Rate: {sample_rate}')
        
        return audio, sample_rate
    
    def preprocess(dataset, type='image'):
        '''
        Normalizes the dataset, returning it with values between -1 and 1.
        
        If you want to plot the images, you need to denormalize it using deprocess()
        
        If you want to listen to the audio, you need to denormalize it using deprocess()
        '''
        if type in ('image', 'img'):
            dataset = dataset.astype('float32') # Floats are necessary, otherwise the normalization will generate pixels with 0 values.
            if dataset.shape[0] > 3000:
                for i in range(len(dataset)):
                    dataset[i] = dataset[i]/127.5 - 1
            
            else:
                dataset = dataset/127.5 - 1.0

            print(f"Dataset type: {type(dataset)}, {dataset.dtype}")

            return dataset
        
        elif type == 'audio':
            dataset = dataset.astype("float64")
            dataset = (dataset - np.min(dataset))*2.0 / (np.max(dataset) - np.min(dataset))-1.0
            
            print(f"Dataset type: {type(dataset)}, {dataset.dtype}")
            
            return dataset

    def deprocess(dataset, type='image', normalized=True, preprocessed_data = None):
        '''Returns the dataset in a way that it can be plotted(images)/listened(audio) or even denormalized(normalized=False)'''
        if type in ('image', 'img'):
            if normalized:
                dataset = (dataset+1.0)*0.5
                print('dataset values are now between 0 and 1')

            else:
                dataset = (dataset+1.0)*127.5
                print('dataset values are now between 0 and 255')
            
            return dataset
        
        elif type == 'audio':
            if preprocessed_data is None:
                raise ValueError("For audios, you need to pass the preprocessed data as argument")
                
            dataset = ((dataset+1.0)*(np.max(preprocessed_data)-np.min(preprocessed_data))*0.5) + np.min(preprocessed_data)
            dataset = dataset.astype(np.(preprocessed_data.dtype))
            
            print("dataset values are now back to their normal range")
            
            return dataset

        
    
    def save_dataset(dataset, dataset_name, save_path=None):
        '''In case you forgot how to save a numpy array using np.save()'''
        if save_path is None:
            np.save(dataset_name, dataset)
            print(f'Dataset saved as {dataset_name}')
        else:
            path_string = save_path + '/' + dataset_name
        
            np.save(path_string, dataset)
            print(f'Dataset saved in {path_string}')

    def load_dataset(load_path, dataset_name):
        '''In case you forgot how to load a numpy array using np.load()'''
        dataset = np.load(load_path + '/' + dataset_name + '.npy')

        print(f"Loaded dataset from {load_path+'/'+dataset_name}")

        return dataset
