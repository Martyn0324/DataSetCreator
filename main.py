import os
import numpy as np
from PIL import Image

class DatasetCreator():
  
  def __init__(self, path, dataset_name, save_path):
    
    self.path = path
    self.dataset_name = dataset_name
    self.save_path = save_path
    
  def images(self, width, height, return_rgba=False):
    pics = []
    
    for directory, _, files in os.walk(self.path):
      for file in files:
        pics.append(directory+'/'+file)
        
    # Removing any files that aren't images
    pics = [i for i in pics if ('.jpg', '.png') in i]
    
    pics = [Image.open(i) for i in pics]
    
    for i in range(len(pics)):
      pics[i] = pics[i].resize((width, height))
      if images[i].mode == 'RGBA' and return_rgba == False:
        images[i] = images[i].convert('RGB')
        
    pics = [np.array(i) for i in pics]
    
    pics = np.array(pics)
    
    # Now, we got an array with shape(n_samples,), where each sample got (width, height, n_channels)
    
    pics = np.stack(pics, axis=0)
    
    # NOW, yes, we got an array with shape (n_samples, width, height, n_channels)
    
    self.dataset_name = pics
    
    print(f'dataset shape: {self.dataset_name.shape}\nSamples shape: {self.dataset_name[0].shape}')
    
    return dataset
  
  def audio(self):
    # TODO - Maybe if someday I learn how to create an audio dataset...
    pass
  
  def preprocess(self, dataset):
    # TODO - I'm lazy right now
    pass
  
  def save_dataset(self):
    path_string = self.save_path + '/' + self.dataset_name
    
    np.save(path_string, self.dataset_name)
  
