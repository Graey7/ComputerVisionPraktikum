import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import DatasetSeamCarved, make_k_folds, extract_data
from albumentations.pytorch import ToTensorV2
import utils
import albumentations as A
from train import train, test
# GET DATA
data_path = '/Users/emiledehn/Downloads/carved_dataset'
data = extract_data(path=data_path)
folds = make_k_folds(data=data, k=5)

# PRE PROCESS DATA
train_transform = A.Compose([
    A.Resize(256, 256), 
    A.RandomCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.HorizontalFlip(p = 0.5), 
    A.VerticalFlip(p = 0.5),
    A.GaussNoise(mean = 0, var_limit=(0.2), p=0.5),
    ToTensorV2()
])

test_transform = A.Compose([    
    A.Resize(256, 256), 
    A.RandomCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# TRAINING
epochs = 1
# TODO: balance batch in pytorch, keyword: sampler?
batch_size = 16
valid_size = 0.1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

model_name = ''
model_path = f'models/model_name/'
graph_path = f'{model_path}/evaluation/'

training_loss, validation_loss = [], []
training_acc, validation_acc = [], []
testing_acc = []

# k-FOLD-CROSS-VALIDATION LOOP
for i in tqdm(range(len(folds)), desc='step'):
    
    # SPLIT DATA ACCORDING TO FOLD INTO TEST AND TRAINING/VALIDATION DATA
    train_data = []
    for j in range(len(folds)):
        if i == j:
            test_data = folds[j]
        else:
            train_data.extend(folds[j])

    # SPLIT DATA SET INTO TRAINING AND VALIDATION
    train_size = int((1-valid_size) * len(train_data))
    test_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, test_size])
    
    # DATASETS FOR DATA LOADERS
    train_dataset = DatasetSeamCarved(train_data, train_transform)
    valid_dataset = DatasetSeamCarved(valid_data, test_transform)
    test_dataset = DatasetSeamCarved(test_data, test_transform)

    # DATA LOADERS FOR TRAINING
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_data), shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_data), shuffle=True)
    
    # TRAIN MODEL
    model, train_loss, valid_loss, train_acc, valid_acc = train(train_data=train_loader, valid_data=valid_loader, epochs=epochs, device=device)
    # GET MODEL METRICS FOR EVALUATION
    training_loss.append(train_loss)
    validation_loss.append(valid_loss)
    training_acc.append(train_acc)
    validation_acc.append(valid_acc)
    
    # TEST MODEL
    test_acc = test(model, test_data=test_loader, device=device)
    # GET MODEL METRICS FOR EVALUATION
    testing_acc.append(test_acc)

    # SAVE MODELS
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.save(model.state_dict(), f'{model_path}fold{i}')

# PLOT MODEL EVALUATION
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

utils.make_plots(graph_path, 'Loss', training_loss, validation_loss)
utils.make_plots(graph_path, 'Accuracy', training_acc, validation_acc)
utils.make_bars(graph_path, 'Test Accuracy', testing_acc)