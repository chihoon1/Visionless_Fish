import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from multiclass_performanceMetrics import *


class simpleEarlyStopping():
    def __init__(self, patience):
        # param patience: int denoting the number of epochs required to halt the training with early stoppage
        # if validation loss doesn't improve in number of epochs(== patience), then training stops
        self.patience = patience
        self.last_val_loss = float('inf')
        self.epoch_count = 0


    def compare_last_curr_epoch(self, curr_val_loss):
        # param curr_val_loss: current epoch's validation loss
        if curr_val_loss >= self.last_val_loss:  # validation loss doesn't decrease
            self.epoch_count += 1
        else:  # validation still decreasing, no need to consider early stoppage
            self.epoch_count = 0
        self.last_val_loss = curr_val_loss


def make_NN_block(input_dim, output_dim, activation, is_batchNorm):
    # build a simple feed forforward neural network block (hidden layer and activtation, etc)
    if is_batchNorm:
        block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            activation
        )
    else:
        block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activation
        )
    return block

class simpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, is_dropOut=False, is_batchNorm=False, **kwargs):
        super(simpleNN, self).__init__()
        # batch norm set up
        self.is_batchnorm = is_batchNorm

        self.dropout = nn.Dropout(p=0.2)  # dropout layer
        self.is_dropout = is_dropOut

        layers_dim = kwargs.get('layers_dim', None)  # a list of hidden dimensions for hidden layers in order
        num_layers = len(layers_dim) if layers_dim is not None else 0

        # a list of activation functions for each layer(include the first layer and may include output layer)
        # Default: ReLu
        activations = kwargs.get('activations', nn.ReLU())

        if num_layers:
            self.layer1 = make_NN_block(input_dim, layers_dim[0], activations[0], self.is_batchnorm)
            self.hidden_layers = nn.Sequential()
            for i in range(1, num_layers):
                self.hidden_layers.add_module(f"hidden{i}",
                                              make_NN_block(layers_dim[i - 1], layers_dim[i], activations[i],
                                                            self.is_batchnorm))
            self.output_layer = nn.Linear(layers_dim[num_layers - 1], output_dim)
            self.output_activation = activations[i + 1] if len(activations) > i + 1 else None
        else:
            # single layer network
            self.output_layer = nn.Linear(input_dim, output_dim)
            self.output_activation = activations


    def forward(self, x):
        # param x: an input (vector or tensor)
        # feed forward through the network and return the output
        try:
            # first layer
            out = self.layer1(x)
            for layer in self.hidden_layers:
                out = layer.forward(out)
                if self.is_dropout:
                    out = self.dropout(out)

            output = self.output_layer(out)
            if self.output_activation:
                output = self.output_activation(output)
        except AttributeError:  # this implies a NN is a single layer network
            output = self.output_layer(x)
            if self.output_activation:
                output = self.output_activation(output)
        return output




class vanillaRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, batch_first=False, **kwargs):
        # param input_dim: dimension(size) of input data
        # param output_dim: dimension(size) of output vector
        # param hidden_dim: dimension(size) of output of hidden layer(at t-1) to be used as input of layer(at t)
        # param num_layers: number of RNN layers (default=1)
        # param batch_first:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        super(vanillaRNN, self).__init__()
        # RNN layer
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=batch_first)
        # Output layer
        self.outl = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.output_activation = kwargs.get('output_activation')


    def forward(self, x):
        # param x: an input (vector or tensor)
        batch_size = x.shape[0]
        # create an initial hidden state(all zeros) for first input of RNN
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).float()
        # feed the input and initial hidden state to the RNN model
        output, _ = self.rnn(x, initial_hidden)
        output = self.outl(output)
        if self.output_activation: output = self.output_activation(output)
        return output


def train(model, train_data, val_data, optimizer, loss_f, device, batch_size=128, n_epochs=700, ES=None, **kwargs):
    # training the pytorch neural network model
    # param model: pytorch neural network model
    # param train_data: n-dimensional array that can be loaded by torch DataLoader
    # param val_data: n-dimensional array that can be loaded by torch DataLoader
    # param optimizer: pytorch optimzer to optimize to minimize loss
    # param loss_f: pytorch loss function
    # param device: torch.device for GPU if available. otherwise, cpu
    # param batch_size: number of input data in a batch (default=128)
    # param n_epochs: number of epochs for training (default=700)
    # param ES: early stopping criteria. (default: None, no early stopping)
    # return a tuple in the order of train loss, validation loss, train accuracy, validation accuracy from all epochs
    all_train_loss = np.empty(n_epochs)
    all_val_loss = np.empty(n_epochs)
    all_train_acc = np.empty(n_epochs)
    all_val_acc = np.empty(n_epochs)
    is_binary = kwargs.get('is_binary', False)  # indicate whether binary or multiclass classification (default: multiclass)
    for i in range(n_epochs):
        # load the data in torch data loader
        train_loader = iter(DataLoader(train_data, batch_size=batch_size, shuffle=True))
        val_loader = iter(DataLoader(val_data, batch_size=batch_size, shuffle=False))
        # train step
        train_loss = 0
        train_acc = 0
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()  # empty out the gradient from previous batch iteration
            X_train_batch = data[:, :-1]
            try:
                _ = model.is_batchnorm
            except AttributeError:  # no batch norm is used in the model
                # resizing to fit in the model
                X_train_batch = X_train_batch.reshape(X_train_batch.shape[0], 1, X_train_batch.shape[1])
            X_train_batch = X_train_batch.float()
            X_train_batch.to(device)

            # forward propagation
            output = model(X_train_batch).squeeze()
            y_train_batch = data[:, -1].long() if not is_binary else data[:, -1].float()
            y_train_batch.to(device)
            loss = loss_f(output, y_train_batch)  # compute loss
            loss.backward()  # back propagation and compute gradients
            optimizer.step()  # update weights using the gradients computed above
            # training accuracy computation and training performance evaluation
            y_pred_batch = predict(model, X_train_batch).squeeze()
            cm = confusion_matrix(y_train_batch, y_pred_batch)
            acc = accuracy(cm)
            train_loss += loss.item()
            train_acc += acc

        # validation step
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for j, data in enumerate(val_loader):
                # compute loss of current validation epoch
                X_val_batch = data[:, :-1]
                try:
                    _ = model.is_batchnorm
                except AttributeError:  # no batch norm is used in the model
                    # resizing to fit in the model
                    X_val_batch = X_val_batch.reshape(X_val_batch.shape[0],
                                                          1, X_val_batch.shape[1])
                X_val_batch = X_val_batch.float()
                X_val_batch.to(device)
                # forward propagation
                output = model(X_val_batch).squeeze()
                y_val_batch = data[:, -1].long() if not is_binary else data[:, -1].float()
                y_val_batch.to(device)
                loss = loss_f(output, y_val_batch)  # compute loss
                # validation accuracy computation and validation performance evaluation
                y_pred_batch = predict(model, X_val_batch).squeeze()
                cm = confusion_matrix(y_val_batch, y_pred_batch)
                acc = accuracy(cm)
                val_loss += loss.item()
                val_acc += acc

        # store train and validation performance per epoch sequentially
        all_train_loss[i] = train_loss / len(train_loader)
        all_val_loss[i] = val_loss / len(val_loader)
        all_train_acc[i] = train_acc / len(train_loader)
        all_val_acc[i] = val_acc / len(val_loader)

        if i % 10 == 9:
            print(f"Epoch {i+1}/{n_epochs}")
            print("Train Loss: {:.4f}".format(train_loss / len(train_loader)), end="\t")
            print("Train Accuracy: {:.4f}".format(train_acc / len(train_loader)))
            print("Validation Loss: {:.4f}".format(val_loss / len(val_loader)), end="\t")
            print("Validation Accuracy: {:.4f}".format(val_acc / len(val_loader)))

        if ES is not None:  # If True, early stopping required for training
            # check if early stopping criteria is met
            #if ES.last_val_loss <= all_val_loss[i]:  # debug
            #    print(ES.last_val_loss, all_val_loss[i])  # debug
            ES.compare_last_curr_epoch(all_val_loss[i])
            # print(ES.epoch_count)  # debug
            if ES.epoch_count >= ES.patience:  # halt the training for early stopping
                print("Early Stopping Executed")
                break

    # slicing 0  to i+1 indices required in case where early stopping was executed
    return all_train_loss[:i+1], all_val_loss[:i+1], all_train_acc[:i+1], all_val_acc[:i+1]


def predict(model, x):
    # param x: an input (vector or tensor)
    # return the predicted classes of x. return data type(torch.tensor)
    y_pred = model(x).squeeze().detach()
    try:
        if model.output_activation:
            # model has its own output activation function
            y_pred = torch.round(y_pred)
            return y_pred
    except AttributeError:
        pass
    # In this case, model didn't define the output layer activation function, so apply softmax if multiclass
    y_pred = nn.functional.softmax(y_pred, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)  # torch.max returns two output tensors (max, max_indices)
    return y_pred


def visualize_train_log(train_loss, val_loss, train_acc, val_acc, loss_fname):
    # param train_loss: np array containing training loss per epoch for all epochs
    # param val_loss: np array containing validation loss per epoch for all epochs
    # param train_acc: np array containing training accuracy per epoch for all epochs
    # param val_acc: np array containing validation accuracy per epoch for all epochs
    # param loss_fname: str type. the name of loss function used

    plt.rcParams["figure.figsize"] = (9, 5)
    # plot loss
    x = np.arange(1,len(train_loss)+1)
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss, linestyle='-', label='Train Set', linewidth=3)
    plt.plot(x, val_loss, linestyle='-', label='Validation Set', linewidth=3)
    plt.title(f'Loss based on {loss_fname} throughout the epochs', fontsize=25)
    plt.legend(fontsize=15)
    plt.ylabel('Loss', fontsize=25)
    plt.xlabel('Epoch', fontsize=25)

    # plot accuracy
    x = np.arange(1, len(train_loss) + 1)
    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc, linestyle='-', label='Train Set', linewidth=3)
    plt.plot(x, val_acc, linestyle='-', label='Validation Set', linewidth=3)
    plt.title(f'Accuracy throughout the epochs', fontsize=25)
    plt.legend(fontsize=15)
    plt.ylabel('Accuracy', fontsize=25)
    plt.xlabel('Epoch', fontsize=25)



def test(model, test_data, device, batch_size=128, **kwargs):
    # testing the pytorch neural network model
    # param model: pytorch neural network model
    # param test_data: n-dimensional array that can be loaded by torch DataLoader
    # param device: torch.device for GPU if available. otherwise, cpu
    # param batch_size: number of input data in a batch (default=128)
    # return a tuple of ground truth test labels and predicted classes

    # indicate whether binary or multiclass classification (default: multiclass)
    is_binary = kwargs.get('is_binary',False)

    # load test data in DataLoader
    test_loader = iter(DataLoader(test_data, batch_size=batch_size, shuffle=False))

    # variables to store model predicted class and ground truth class labels
    test_true = np.zeros(test_data.shape[0])
    test_pred = np.zeros(test_data.shape[0])
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            # batch data setup
            X_test_batch = data[:, :-1]
            try:
                _ = model.is_batchnorm
            except AttributeError:  # no batch norm is used in the model
                # resizing to fit in the model
                X_test_batch = X_test_batch.reshape(X_test_batch.shape[0],
                                                1, X_test_batch.shape[1])
            X_test_batch = X_test_batch.float()
            X_test_batch.to(device)
            y_test_batch = data[:, -1].long() if not is_binary else data[:, -1].float()
            y_test_batch.to(device)
            # get predicted classes
            preds = predict(model, X_test_batch).squeeze()
            test_pred[batch_size * j:batch_size * (j + 1)] = preds.numpy()
            # compile the ground truth test labels in test data (same order as predicted classes)
            test_true[batch_size * j:batch_size * (j + 1)] = y_test_batch.numpy()
    return test_true, test_pred
