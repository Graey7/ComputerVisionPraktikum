from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import os
import cv2

def extract_data(path):
    data = []

    file_count = sum(len(files) for _, _, files in os.walk(path))

    with tqdm(desc='extract data', total=file_count) as pbar:
        for subdir, dirs, files in os.walk(path):
            for file in files: 
                if '.DS_Store' not in file:
                    image = cv2.imread(os.path.join(subdir, file))    
                    # opencv uses bgr we need rgb
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if 'carved' in file: 
                        label = 1
                    else:
                        label = 0

                    data.append((image, label))
                    pbar.update(1)
    return data

def make_k_folds(data, k):
    folds = []
    split_size = int(len(data) * 1/k)
        
    split, rest = random_split(data, [split_size, len(data)-split_size])
    folds.append(split)
    for i in range(1,k-1):
        split, rest = random_split(rest, [split_size, len(data)-split_size*(i+1)])
        folds.append(split)
    folds.append(rest)

    return folds

class DatasetSeamCarved(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        label = self.data[index][1]
        
        # transform while getting item
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return (image, label)
