'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.MLP_Training_Visualization import MLP_Training_Visualizer
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 501
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 5e-3

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, 256) # First hidden layer: 784 input features, 256 output features
        self.activation_func_1 = nn.ReLU() 
        self.fc_layer_2 = nn.Linear(256, 128) # Second hidden layer: 256 input features, 128 output features
        self.activation_func_2 = nn.ReLU() 
        self.fc_layer_3 = nn.Linear(128, 10) # Third layer: 128 input features, 10 output features (output)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x)) # First hidden layer, 256 output features
        h2 = self.activation_func_2(self.fc_layer_2(h1)) # Second hidden layer, 128 output features
        y_pred = self.fc_layer_3(h2) # Output layer, 10 output features

        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        evaluator = Evaluate_Metrics('training evaluator', '')
        visualizer = MLP_Training_Visualizer()
        
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            visualizer.loss_history.append(train_loss.item())

            if epoch%50 == 0:
                evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                scores = evaluator.evaluate()
                print(f'Epoch: {epoch}')
                print('Training Metrics:')
                for metric_name, score in scores.items():
                    print(f'  {metric_name}: {score:.4f}')
                print(f'Loss: {train_loss.item():.5f}\n')
        
        visualizer.plot_training_curve()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            