import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import torch.optim as optim
from torchvision.transforms import Compose, RandomCrop, ToTensor

from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.lib.model.poster_net import PosterNet
from lip.utils.common import CROPPED_IMAGE_SIDE
from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR

if __name__ == '__main__':

    # CUDA
    cuda_available: bool = True
    if torch.cuda.is_available():
        print("CUDA available")
    else:
        print("CUDA not available")

    device = torch.device("cuda:0" if cuda_available else "cpu")

    # DATA
    SPLIT_RATIO = 0.7

    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Compose([RandomCrop(CROPPED_IMAGE_SIDE),
                                                                       ToTensor()]))

    data_set_size: int = len(movie_data_set)
    print(f'Size of the data-set: {data_set_size}')

    train_data_set_size: int = int(data_set_size * SPLIT_RATIO)
    test_data_set_size: int = data_set_size - train_data_set_size
    train_dataset, test_dataset = torch.utils.data.random_split(movie_data_set, [train_data_set_size,
                                                                                 test_data_set_size])

    train_data_set_loader: DataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_data_set_loader: DataLoader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # MODEL
    net: PosterNet = PosterNet()
    if cuda_available:
        net.cuda(device)

    loss_function: BCELoss = BCELoss()
    if cuda_available:
        loss_function.cuda(device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # TRAINING
    NUM_EPOCHS: int = 2
    for epoch in range(NUM_EPOCHS):

        running_loss: float = 0.0
        for index, data in enumerate(train_data_set_loader):
            # GET INPUT AND OUTPUT
            X, y = data
            if cuda_available:
                X = Variable(X.cuda())
                y = Variable(y.cuda())

            # WE NEED TO ZERO THE GRADIENTS BEFORE TRAINING OVER A BATCH
            optimizer.zero_grad()
            # CALCULATE THE PREDICTION
            y_pred = net(X)
            # CALCULATE THE LOSS ON THE PREDICTION
            loss = loss_function(y_pred, y)
            # APPLY GRADIENT DESCENT
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if index % 200 == 199:  # print every 1000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {(running_loss / 2000):.6f}')
                running_loss = 0.0

    print(f'Finished training')
