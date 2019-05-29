import copy
from typing import Dict, List

import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.lib.model.plot_net import PlotNet
from lip.lib.model.poster_net import PosterNet
from lip.utils.common import WORKING_IMAGE_SIDE
from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR, DATA_DIR

if __name__ == '__main__':

    # CUDA
    cuda_available: bool = True
    if torch.cuda.is_available():
        print("CUDA available")
    else:
        print("CUDA not available")
    # cuda_available = False

    device = torch.device("cuda:0" if cuda_available else "cpu")

    # DATA
    SPLIT_RATIO: float = 0.7
    BATCH_SIZE: int = 4

    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Dictionary(DATA_DIR / 'dict2000.json'),
                                                              Compose([Resize((WORKING_IMAGE_SIDE,
                                                                               WORKING_IMAGE_SIDE)),
                                                                       ToTensor()]))

    data_set_size: int = len(movie_data_set)
    print(f'Size of the data-set: {data_set_size}')

    train_data_set_size: int = int(data_set_size * SPLIT_RATIO)
    val_data_set_size: int = data_set_size - train_data_set_size
    train_dataset, val_dataset = torch.utils.data.random_split(movie_data_set, [train_data_set_size,
                                                                                val_data_set_size])

    train_data_set_loader: DataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_data_set_loader: DataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    data_set_loaders: Dict[str, DataLoader] = {'train': train_data_set_loader,
                                               'val': val_data_set_loader}

    data_set_sizes = {'train': len(train_dataset),
                      'val': len(val_dataset)}
    print(f'Data-set sizes: {data_set_sizes}')

    # MODEL
    net: PosterNet = PlotNet(vocab_size=2001)
    if cuda_available:
        net.cuda(device)

    loss_function: BCELoss = BCELoss()
    if cuda_available:
        loss_function.cuda(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # TRAINING
    NUM_EPOCHS: int = 1

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    cutoff = torch.tensor([0.5] * BATCH_SIZE).reshape((BATCH_SIZE, 1)).to(device)

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
                net.train()
            else:
                net.eval()

            running_loss: float = 0.0
            running_corrects: int = 0

            for index, data in enumerate(data_set_loaders[phase]):

                # GET INPUT AND OUTPUT
                X_image, X_plot, y = data
                if cuda_available:
                    X_plot = X_plot.to(device)
                    y = y.to(device)

                # WE NEED TO ZERO THE GRADIENTS BEFORE TRAINING OVER A BATCH
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # CALCULATE THE PREDICTION
                    y_pred = net(X_plot)

                    predicted_labels = (y_pred >= cutoff).float()
                    corrects = (predicted_labels == y).float()

                    # CALCULATE THE LOSS ON THE PREDICTION
                    loss = loss_function(y_pred, y)

                    # APPLY GRADIENT DESCENT
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X_plot.size(0)
                running_corrects += torch.sum(corrects)

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

    print(f'Finished training')