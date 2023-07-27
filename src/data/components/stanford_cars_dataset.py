import os
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image

from src.data.components.transforms import img_classification_transform


class StanfordCarsCustomDataset(Dataset):
    def __init__(self, data_dir='../data/', train=True, transforms=img_classification_transform):
        super().__init__() 
        
        self.stage = 'train' if train else 'test'
        
        # images
        self.directory = f'{data_dir}/stanford-cars-dataset/cars_{self.stage}/cars_{self.stage}'
        self.images = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
        
        # transforms
        self.transforms = transforms        
        
        # annotations
        cars_annos_train_mat = loadmat(f'{data_dir}/stanford-cars-dataset-meta/devkit/cars_train_annos.mat')
        cars_annos_test_mat = loadmat(f'{data_dir}/stanford-cars-dataset-meta/cars_test_annos_withlabels (1).mat')
        
        self.training_image_label_dictionary, self.testing_image_label_dictionary = {}, {}

        for arr in cars_annos_train_mat['annotations'][0]:
            image, label = arr[-1][0], arr[-2][0][0] - 1
            self.training_image_label_dictionary[image] = label

        for arr in cars_annos_test_mat['annotations'][0]:
            image, label = arr[-1][0], arr[-2][0][0] - 1
            self.testing_image_label_dictionary[image] = label
            
        if train:
            self.image_label_dict = self.training_image_label_dictionary
        else:
            self.image_label_dict = self.testing_image_label_dictionary

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        image = self.images[index]
        img_pil = Image.open(image).convert('RGB')
        img_trans = self.transforms(img_pil)

        # Parse out the label from cars_meta and cars_x_annos files
        image_stem = image.split("/")[-1]
        img_label = self.image_label_dict[image_stem]

        return img_trans, img_label
    
    
if __name__ == "__main__":
    _ = StanfordCarsCustomDataset()