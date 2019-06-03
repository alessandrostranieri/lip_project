import copy
from typing import Dict, List

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset, get_class_weights
from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR, DATA_DIR

if __name__ == '__main__':

    torch.manual_seed(42)

    # CUDA
    torch.cuda.init()
    cuda_available: bool = True
    if torch.cuda.is_available():
        print("CUDA available")
    else:
        print("CUDA not available")

    device = torch.device("cuda:0" if cuda_available else "cpu")
    # cuda_available = False

    # DATA
    SPLIT_RATIO: float = 0.7
    BATCH_SIZE: int = 32

    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Dictionary(DATA_DIR / 'dict2000.json'),
                                                              Compose([Resize((299, 299)),
                                                                       ToTensor()]))

    # MODEL

    # LOAD PRE-TRAINED MODEL
    net = inception_v3(pretrained=True)
    # FREEZE FEATURE EXTRACTOR
    for param in net.parameters():
        param.requires_grad = False
    # SET FINAL LAYER
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)
    if cuda_available:
        net.cuda(device)

    data_set_size: int = len(movie_data_set)
    print(f'Size of the data-set: {data_set_size}')

    train_data_set_size: int = int(data_set_size * SPLIT_RATIO)
    val_data_set_size: int = data_set_size - train_data_set_size
    train_dataset, val_dataset = torch.utils.data.random_split(movie_data_set, [train_data_set_size,
                                                                                val_data_set_size])
    weights: np.ndarray = get_class_weights(train_dataset)

    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_data_set_loader: DataLoader = DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   sampler=weighted_sampler,
                                                   drop_last=True)
    val_data_set_loader: DataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)
    data_set_loaders: Dict[str, DataLoader] = {'train': train_data_set_loader,
                                               'val': val_data_set_loader}

    data_set_sizes = {'train': len(train_dataset),
                      'val': len(val_dataset)}
    print(f'Data-set sizes: {data_set_sizes}')

    loss_function: CrossEntropyLoss = CrossEntropyLoss()
    if cuda_available:
        loss_function.cuda(device)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # TRAINING
    NUM_EPOCHS: int = 50

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    best_epoch = 0

    loss_history: Dict[str, List[float]] = {'train': [],
                                            'val': []}
    accuracy_history: Dict[str, List[float]] = {'train': [],
                                                'val': []}

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            print(f'Phase {phase}')
            if phase == 'train':
                exp_lr_scheduler.step(epoch)
                net.train()
            else:
                net.eval()

            running_loss: float = 0.0
            running_corrects: int = 0

            for index, data in enumerate(data_set_loaders[phase]):

                # GET INPUT AND OUTPUT
                X, Xp, y = data
                if cuda_available:
                    X = X.to(device)
                    y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # CALCULATE THE PREDICTION
                    if phase == 'train':
                        y_pred, aux = net(X)
                    else:
                        y_pred = net(X)

                    _, predicted_labels = torch.max(y_pred, 1)
                    correct_tensor = predicted_labels.eq(y.data.view_as(predicted_labels))
                    if not cuda_available:
                        correct = np.squeeze(correct_tensor.numpy())
                    else:
                        correct = np.squeeze(correct_tensor.cpu().numpy())

                    # CALCULATE THE LOSS ON THE PREDICTION
                    loss = loss_function(y_pred, y)

                    # APPLY GRADIENT DESCENT
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_corrects += np.sum(correct)

            # EPOCH STATISTCS
            epoch_loss = running_loss / data_set_sizes[phase]
            epoch_acc = running_corrects / data_set_sizes[phase]

            print(f'Correctly classified: {running_corrects}')
            print(f'{phase} Loss: {epoch_loss:.8f} Acc: {epoch_acc:.8f}')

            loss_history[phase].append(epoch_loss)
            accuracy_history[phase].append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                best_epoch = epoch

    print(f'Finished training')

    print(f'Best Accuracy: {best_acc} at epoch {best_epoch}')

    # SAVE HISTORIES
    loss_df = pd.DataFrame(loss_history)
    accuracy_df = pd.DataFrame(accuracy_history)

    loss_df.to_csv('poster_inception_loss_history.csv')
    accuracy_df.to_csv('poster_inception_accuracy_history.csv')

    torch.save(net.state_dict(), 'poster_inception.model')

