
# Note: The models are not true to their respective papers, but are close.
# Choices were made so that they could be run on single-GPU consumer hardware.


import torch
from torch import nn
import torchvision
import copy
import numpy as np
import os
import matplotlib.pyplot as plt


from lightly.models.modules import SimCLRProjectionHead, MoCoProjectionHead
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from sklearn.manifold import TSNE

import helper_train
import helper_evaluate
import helper_data

# Hyperparameters
RANDOM_SEED = 123
# LEARNING_RATE = 0.1
# BATCH_SIZE = 128
NUM_EPOCHS = 100
# NUM_DATAPOINTS=5

# Set to run on mps if available (i.e. Apple's GPU).
# mps is a new pytorch feature, so we check that
# it's also available with the user's pytorch install.
DEVICE = 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu'
DEVICE = 'cuda'
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
data_location=('/data')
# data_location = ('/Users/duanemyklejord/Documents/Capstone/PlantAutomatedScripts/data')

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


# train_loader, test_loader = helper_data.get_dataloaders(data_location, batch_size=BATCH_SIZE)
# train_loader, test_loader = helper_data.get_dataloaders_reduced_data(data_location, batch_size=BATCH_SIZE, num_datapoints=NUM_DATAPOINTS)
# class_names = [f.name for f in os.scandir(data_location) if f.is_dir()]

# The models themselves:
class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    # i.e. the online branch
    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    # i.e. the target branch
    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

for LEARNING_RATE in [0.0001, 0.0003, 0.01, 0.03, 0.06, 0.1, 1, 10]:
    for NUM_DATAPOINTS in [5, 10, 20, 50, 100, 200, 250]:

        # Building the backbone. Here is where you can change it to whatever you
        # want, for example a resent50. weights=DEFAULT initializes to ImageNet1k weights:
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        backbone = nn.Sequential(*list(resnet.children())[:-1]) # removes FC layer

        # In this case, we want to use MoCo with a SGD optimizer:
        model = MoCo(backbone).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        BATCH_SIZE = NUM_DATAPOINTS
        if BATCH_SIZE > 128:
            BATCH_SIZE = 128

        # Use all the data:
        if NUM_DATAPOINTS == 250:
            model_name=f'{model.__class__.__name__}_{LEARNING_RATE}LR_RedAug'
            train_loader, test_loader = helper_data.get_dataloaders(data_location, batch_size=BATCH_SIZE)
        else:
            model_name = f'{model.__class__.__name__}_{LEARNING_RATE}LR_{NUM_DATAPOINTS}DP_RedAug'
            train_loader, test_loader = helper_data.get_dataloaders_reduced_data(data_location, batch_size=BATCH_SIZE, num_datapoints=NUM_DATAPOINTS)
        class_names = [f.name for f in os.scandir(data_location) if f.is_dir()]
        
        x1, x2, label = next(iter(train_loader))
        print(f'shape x1: {np.shape(x1)}')


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
        os.chdir(model_name)
        
        # Train the model, returns a dict of the training loss. Change the
        # function call to train the model you want.
        training_log_dict = helper_train.train_moco(num_epochs=NUM_EPOCHS, model=model,
                            optimizer=optimizer, device=DEVICE,
                            train_loader=train_loader,
                            save_model=model_name,
                            logging_interval=2,
                            save_epoch_states=False)

        # Saving the log of training loss to a csv
        import csv
        w = csv.writer(open(model_name+'_TrainingLossLog.csv', "w"))
        for key, val in training_log_dict.items():
            w.writerow([key, val])

        plt.plot(np.linspace(1, NUM_EPOCHS, NUM_EPOCHS), training_log_dict['train_loss_per_epoch'])
        plt.savefig(model_name+'_EpochLossPlot.png')

        # Inference:
        # Passing images through the trained backbone to get their embeddings/features/latent_space etc.
        train_X, train_y, test_X, test_y, test_images, train_images = helper_evaluate.get_features(model, train_loader, test_loader, DEVICE)
        np.savez(model_name+'_features', train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, test_images=test_images, train_images=train_images)

        # Trains a linear classifier on the data to determine accuracy and makes a confusion matrix.
        # Lets us see exactly what's going on under the hood, vs using sklearn.
        # returns dict of training and final losses and accuracies.
        eval_log_dict = helper_evaluate.lin_eval(train_X, train_y, test_X, test_y, model_name, class_names, DEVICE='cpu')

        comp_log_dict['final_epoch_loss'] = training_log_dict['train_loss_per_epoch'][-1]
        comp_log_dict['final_batch_loss'] = training_log_dict['train_loss_per_batch'][-1]
        comp_log_dict['training_time'] = training_log_dict['total_time'][-1]
        comp_log_dict['final_accuracy'] = eval_log_dict['final_accuracy'][-1]

        w = csv.writer(open(model_name+'_Meta.csv', "w"))
        for key, val in comp_log_dict.items():
            w.writerow([key, val])

        # BUT, it's slow, so we can use sklearn instead if we want:
        # pred_labels = helper_evaluate.linear_classifier(train_X, train_y, test_X, test_y)
        # confusion_matrix, accuracy = helper_evaluate.make_confusion_matrix(pred_labels, test_y, len(class_names))
        # helper_evaluate.visualize_confusion_matrix(confusion_matrix, accuracy, class_names, model_name)

        # TSNE analysis and visualization:
        tsne_xtest = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, n_iter=1000).fit_transform(test_X)
        helper_evaluate.visualize_tsne(model_name, tsne_xtest, class_names, test_y, close_fig=True)

        # Visualize TSNE with predicted labels:
        if len(np.unique(test_y))>1:
            pred_labels = helper_evaluate.linear_classifier(train_X, train_y, test_X, test_y)

        # If data is unlabeled, we use K-means:
        else:
            pred_labels = helper_evaluate.kmeans_classifier(test_X, k=10)
            class_names = np.unique(pred_labels)
            test_y = None

        helper_evaluate.visualize_hover_images(model_name, tsne_xtest, test_images, pred_labels, class_names, test_y, showplot=True)

        os.chdir('..')

quit()


'''
# Extra functions that may be handy:

pred_labels = helper_evaluate.kmeans_classifier_2class(test_X, test_y)
pred_labels = helper_evaluate.kmeans_classifier(test_X, k=10)
pred_labels = helper_evaluate.knn_classifier(train_X, train_y, test_X, test_y, k=100)

# For saving time with pre-trained model:
model.load_state_dict(torch.load(f'{model_name}.pt')) # to load a pre-trained model
features = np.load(model_name+'_features.npz')          # to load the embedded space that you've already found
features = np.load('MoCo_10EP_128BS_0.1LR_features.npz')
train_X, train_y, test_X, test_y, test_images = features['train_X'], features['train_y'], features['test_X'], features['test_y'], features['test_images']   # Gets the embedded space into a usable format.
# '''

import numpy as np
model_name = 'MoCo_0.5LR'
features = np.load(os.listdir()[0]+'/MoCo_0.5LR_features.npz')
train_X, train_y, test_X, test_y, test_images = features['train_X'], features['train_y'], features['test_X'], features['test_y'], features['test_images']   # Gets the embedded space into a usable format.


feature_file_dirs = os
import os
os.listdir()

dirs = [x.name for x in os.scandir() if x.is_dir()]
next(os.scandir()).name

# To compile all the accuracy numbers:
acc_dict = {}
import csv
import os
dirs = [x.name for x in os.scandir() if x.is_dir()]
for dir in dirs:
    os.chdir('/Users/duanemyklejord/Documents/Capstone/PlantAutomatedScripts/'+dir)
    if os.path.isfile(dir+'_Meta.csv'):
        print(f'Currently: {dir}')      
        model_name = dir

        with open(model_name+'_Meta.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'final_accuracy':
                    # If it does, add the key-value pair to the dictionary
                    acc_dict[model_name] = row[1]


        # features = np.load(dir+'_features.npz')
        # train_X, train_y, test_X, test_y, test_images = features['train_X'], features['train_y'], features['test_X'], features['test_y'], features['test_images']   # Gets the embedded space into a usable format.
        # eval_log_dict = helper_evaluate.lin_eval(train_X, train_y, test_X, test_y, model_name, class_names, DEVICE='cpu')
        # tsne_xtest = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, n_iter=1000).fit_transform(test_X)
        # helper_evaluate.visualize_tsne(model_name, tsne_xtest, class_names, test_y, close_fig=True)
    else:
        print(f'This is not a valid dir: {dir}')

os.chdir('/Users/duanemyklejord/Documents/Capstone/PlantAutomatedScripts/')
w = csv.writer(open('Model_Accuracies.csv', "w"))
for key, val in acc_dict.items():
    w.writerow([key, val])

### End accuracy compilation

from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state=0)
LogReg.fit(train_X, train_y)
pred_labels = LogReg.predict(test_X)

from importlib import reload  # Python 3.4+
import helper_evaluate
reload(helper_evaluate)
eval_log_dict = helper_evaluate.lin_eval(train_X, train_y, test_X, test_y, model_name, class_names, DEVICE='cpu')

reload(helper_evaluate)
tsne_xtest = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, n_iter=1000).fit_transform(test_X)
helper_evaluate.visualize_tsne(model_name, tsne_xtest, class_names, test_y, close_fig=True)


# features = np.load('Pretrained_SegWeeds_features.npz')
# train_X, train_y, test_X, test_y, test_images, train_images = features['train_X'], features['train_y'], features['test_X'], features['test_y'], features['test_images'], features['train_images']   # Gets the embedded space into a usable format.

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=200, random_state=0).fit(train_X)
# pred_labels = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_

# from sklearn.neighbors import NearestNeighbors
# NN = NearestNeighbors(n_neighbors=1).fit(train_X)
# neighbor_idx = NN.kneighbors(cluster_centers)[1]

# min_images = []
# min_classes = []
# for ind in neighbor_idx:
#     min_images.append(train_images[ind])
#     min_classes.append(train_y[ind])
# min_images = np.squeeze(min_images)
# min_images = (min_images*255).astype(np.uint8).transpose(0,2,3,1)
# min_train_y = np.squeeze(min_classes)

# class ImagePair(torch.utils.data.Dataset):
#     """
#     Returns the same image transformed two ways.
#     https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
#     """
#     def __init__(self, data, target, transform=None, target_transform=None):
#         self.data = data
#         self.target = target
#         self.transform = transform
#         self.target_transform = target_transform

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample1 ,sample2, target) where target is class_index of the target class.
#             sample1 and sample2 are augmented/transformed versions of sample (sample==image).
#         """
#         sample1 = self.data[index]
#         sample2 = self.data[index]
#         target = self.target[index]
        
#         if self.transform is not None:
#             sample1 = self.transform(Image.fromarray(sample1))
#             sample2 = self.transform(Image.fromarray(sample2))

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample1, sample2, target

#     def __len__(self):
#         return len(self.data)

# from PIL import Image
# min_dataset = ImagePair(min_images, min_train_y,  transform=helper_data.simclr_transforms.train)
# train_loader = torch.utils.data.DataLoader(dataset=min_dataset, 
#                                              batch_size=BATCH_SIZE,
#                                              num_workers=0,
#                                              drop_last=True,
#                                              shuffle=True)

# model_name='200Centroid_Pretrained_Segweeds_SimCLR_0.0003LR_128BS'
# LEARNING_RATE = 0.0003
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# model = SimCLR(backbone).to(DEVICE)
# log_dict = helper_train.train_simclr(num_epochs=500, model=model,
#                     optimizer=optimizer, device=DEVICE,
#                     train_loader=train_loader,
#                     save_model=model_name,
#                     logging_interval=2,
#                     save_epoch_states=False)

# import csv
# w = csv.writer(open(model_name+'_LossLog.csv', "w"))
# for key, val in log_dict.items():
#     w.writerow([key, val])

    
# NUM_EPOCHS=500
# import matplotlib.pyplot as plt
# plt.plot(np.linspace(1, NUM_EPOCHS, NUM_EPOCHS), log_dict['train_loss_per_epoch'])
# plt.show()

# train_X, train_y, test_X, test_y, test_images, train_images = helper_evaluate.get_features(model, train_loader, test_loader, DEVICE)
# np.savez(model_name+'_features', train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, test_images=test_images)


# log_dict = helper_evaluate.lin_eval(train_X, train_y, test_X, test_y, model_name, class_names, DEVICE='cpu')


# tsne_xtest = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, n_iter=1000).fit_transform(test_X)
# helper_evaluate.visualize_tsne(model_name, tsne_xtest, class_names, test_y, close_fig=True)
# if len(np.unique(test_y))>1:
#     pred_labels = helper_evaluate.linear_classifier(train_X, train_y, test_X, test_y)

# helper_evaluate.visualize_hover_images(model_name, tsne_xtest, test_images, pred_labels, class_names, test_y, showplot=True)

