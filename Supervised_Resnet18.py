import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import helper_evaluate
import helper_data_reduced
import helper_data_simclr

# Set to run on mps if available (i.e. Apple's GPU).
# mps is a new pytorch feature, so we check that
# it's also available with the user's pytorch install.
DEVICE = 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
        else 'cuda' if hasattr(torch.backends, "cuda") and torch.cuda.is_available() \
        else 'cpu'

# Hyperparameters
RANDOM_SEED = 123
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 100
helper_evaluate.set_deterministic
helper_evaluate.set_all_seeds(RANDOM_SEED)

# Getting the data:
# # For CIFAR10:
# train_loader, test_loader = helper_data.get_dataloaders_cifar10(batch_size=BATCH_SIZE)
# class_names = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# For other data:
# The directory should contain folders of images, with each folder
# having images of a certain class. Example: 2 folders for 2 classes.
# The folder names should be the class names.
parent_dir = os.path.dirname(os.getcwd())
data_location=('/data')

# Getting the cropped plant seedling dataset
if not os.path.exists(data_location):
    import urllib.request
    import zipfile

    # Download the dataset from the URL
    url = 'https://vision.eng.au.dk/?download=/data/WeedData/Segmented.zip'
    filename, headers = urllib.request.urlretrieve(url)

    # Extract the contents of the downloaded file to a folder called "data"
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/data')

    import shutil

    # Move the contents of data/Segmented to data/
    src_dir = '/data/Segmented'
    dst_dir = '/data'
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.move(src_path, dst_path)

    # Delete the Segmented folder
    os.rmdir('/data/Segmented')

augmentations = ['Reduced', 'SimCLR']
for data_aug in augmentations:
    for LEARNING_RATE in [0.0001, 0.0003, 0.01, 0.03, 0.06, 0.1, 1, 10]:
        for NUM_DATAPOINTS in [5, 10, 20, 50, 100, 200, 250]:

            BATCH_SIZE = NUM_DATAPOINTS
            if BATCH_SIZE > 128:
                BATCH_SIZE = 128

            # Use all the data:
            if NUM_DATAPOINTS == 250:
                if data_aug == 'Reduced':
                    model_name=f'Supervised_Resnet18_{LEARNING_RATE}LR_RedAug'
                    train_loader, test_loader = helper_data_reduced.get_dataloaders(data_location, batch_size=BATCH_SIZE)
                if data_aug == 'SimCLR':
                    model_name=f'Supervised_Resnet18_{LEARNING_RATE}LR_SimclrAug'
                    train_loader, test_loader = helper_data_simclr.get_dataloaders(data_location, batch_size=BATCH_SIZE)

            else:
                if data_aug == 'Reduced':
                    model_name = f'Supervised_Resnet18_{LEARNING_RATE}LR_{NUM_DATAPOINTS}DP_RedAug'
                    train_loader, test_loader = helper_data_reduced.get_dataloaders_reduced_data(data_location, batch_size=BATCH_SIZE, num_datapoints=NUM_DATAPOINTS)
                if data_aug == 'SimCLR':
                    model_name = f'Supervised_Resnet18_{LEARNING_RATE}LR_{NUM_DATAPOINTS}DP_SimclrAug'
                    train_loader, test_loader = helper_data_simclr.get_dataloaders_reduced_data(data_location, batch_size=BATCH_SIZE, num_datapoints=NUM_DATAPOINTS)

            class_names = [f.name for f in os.scandir(data_location) if f.is_dir()]

            # Use a pre-trained CNN model for feature extraction
            model = models.resnet18(weights='DEFAULT')
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(class_names))
            model.to(DEVICE)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


            comp_log_dict = {'final_epoch_loss':[],
                            'final_batch_loss': [],
                            'final_accuracy': [],
                            'training_time': [],
                            'model': model.__class__.__name__,
                            'learning_rate': LEARNING_RATE,
                            'batch_size': BATCH_SIZE,
                            'num_epochs': NUM_EPOCHS,
                            'num_datapoints': NUM_DATAPOINTS,
                            'optimizer_type': optimizer.__class__.__name__
                            }

            try:
                os.mkdir(model_name)
            except:
                pass
            os.chdir('/Users/duanemyklejord/Documents/Capstone/PlantAutomatedScripts/'+model_name)

            # If you have an already trained model, it can be loaded. Training steps can then be skipped.
            # model.load_state_dict(torch.load(f'{model_name}.pt')) # to load a pre-trained model

            # Fine-tune the CNN model on the given dataset
            start_time = time.time()
            train_loss_per_epoch = []
            for epoch in range(NUM_EPOCHS):
                epoch_loss = 0
                print(f'Current model: {model_name}')
                for batch_idx, (inputs, _, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    epoch_loss += loss.item()
                                
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if not batch_idx % 4:
                        print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                                % (epoch+1, NUM_EPOCHS, batch_idx,
                                    len(train_loader), loss))
                
                train_loss_per_epoch.append(epoch_loss)
                print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))    
            print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
            comp_log_dict['training_time'] = ((time.time() - start_time)/60)
            torch.save(model.state_dict(), model_name+'.pt')

            # Plot the epoch losses
            plt.plot(np.linspace(1, NUM_EPOCHS, NUM_EPOCHS), train_loss_per_epoch)
            plt.savefig(model_name+'_EpochLossPlot.png')

            # Evaluate the model on the test dataset
            print('Evaluating the model on the test dataset...')
            correct, total = 0, 0
            pred_labels, labels = [], []
            model.eval().to('cpu')
            with torch.no_grad():
                for images, _, batch_labels in test_loader:

                    outputs = model(images)
                    _, batch_preds = torch.max(outputs.data, 1)
                    
                    total += batch_labels.size(0)
                    correct += (batch_preds == batch_labels).sum().item()
                    
                    pred_labels.extend(batch_preds.numpy())
                    labels.extend(batch_labels.numpy())


            accuracy = correct / total
            print(f'Accuracy of the network on the test images: {accuracy}')
            comp_log_dict['final_accuracy'] = accuracy

            w = csv.writer(open(model_name+'_Meta.csv', "w"))
            for key, val in comp_log_dict.items():
                w.writerow([key, val])

            # Calculate classification statistics
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
            import matplotlib.pyplot as plt

            confusion = confusion_matrix(labels, pred_labels)
            report = classification_report(labels, pred_labels)

            # Print the classification statistics
            print('Confusion Matrix:')
            print(confusion)
            print('\nClassification Report:')
            print(report)

            helper_evaluate.visualize_confusion_matrix(confusion, accuracy, class_names, model_name)

            # plt.rcParams["figure.autolayout"]= True # to make sure the labels don't get cut off
            # display = ConfusionMatrixDisplay(confusion, display_labels=class_names)
            # fig, ax = plt.subplots(figsize=(10, 10))
            # display.plot(ax=ax, xticks_rotation=45)
            # # plt.xticks(rotation = 45, ha='right') # Rotates X-Axis Ticks by 45-degrees
            # plt.show()
            # plt.savefig(model_name+'_ConfusionMatrix.png')

            os.chdir('/Users/duanemyklejord/Documents/Capstone/PlantAutomatedScripts/')
