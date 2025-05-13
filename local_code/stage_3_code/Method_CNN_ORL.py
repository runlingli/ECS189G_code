'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Training_Visualization import Training_Visualizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ORL class names (person IDs)
classes = [str(i) for i in range(40)]  # ORL has 40 people

class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch = 1
    learning_rate = 2e-3
    batch_size = 128

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # CNN architecture for ORL (grayscale images)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation_func_1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.activation_func_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.activation_func_3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.activation_func_4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.activation_func_5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.activation_func_6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 14 * 11, 512)
        self.activation_func_linear_1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.activation_func_linear_2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)
        self.activation_func_linear_3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 40)  # ORL has 40 classes
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        c11 = self.activation_func_1(self.bn1(self.conv1(x)))
        c12 = self.activation_func_2(self.bn2(self.conv2(c11)))
        p1 = self.pool1(c12)

        c21 = self.activation_func_3(self.bn3(self.conv3(p1)))
        c22 = self.activation_func_4(self.bn4(self.conv4(c21)))
        p2 = self.pool2(c22)

        c31 = self.activation_func_5(self.bn5(self.conv5(p2)))
        c32 = self.activation_func_6(self.bn6(self.conv6(c31)))
        p3 = self.pool3(c32)

        f = torch.flatten(p3, 1)
        h1 = self.activation_func_linear_1(self.fc1(f))
        h1 = self.dropout(h1)
        h2 = self.activation_func_linear_2(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = self.activation_func_linear_3(self.fc3(h2))
        y_pred = self.fc4(h3)
        
        return y_pred

    def train_model(self):
        self.train()  # training mode
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss()
        visualizer = Training_Visualizer()

        train_loader = self.data['train_loader']
        print(f"Training on {len(train_loader.dataset)} samples")

        # Training loop
        for epoch in range(self.max_epoch):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}'):
                images = images.to(device)
                labels = labels.to(device).long()

                outputs = self.forward(images)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total

            visualizer.loss_history.append(epoch_loss)
            visualizer.acc_history.append(epoch_acc)

            if epoch % 5 == 0:
                print(f'Epoch: {epoch}')
                print(f'Training Loss: {epoch_loss:.4f}')
                print(f'Training Accuracy: {epoch_acc:.2f}%\n')

        visualizer.plot_training_curve()

    def visualize_test_results(self, test_loader, num_samples=10):
        """visualize test results and save as PNG file"""
        self.eval()
        images, labels = next(iter(test_loader))
        images = images[:num_samples].to(device)
        labels = labels[:num_samples].to(device).long()
        
        with torch.no_grad():
            outputs = self.forward(images)
            _, predicted = outputs.max(1)
        
        # convert image format for display
        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # from (C,H,W) to (H,W,C)
        # denormalize
        images = images * 0.5 + 0.5  # from [-1,1] to [0,1]
        images = np.clip(images, 0, 1)
        
        # create image grid
        fig = plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            ax = fig.add_subplot(1, num_samples, i + 1)
            ax.imshow(images[i].squeeze(), cmap='gray')  # ORL is grayscale
            ax.axis('off')
            color = 'green' if predicted[i] == labels[i] else 'red'
            ax.set_title(f'True: Person {classes[labels[i]]}\nPred: Person {classes[predicted[i]]}', 
                        color=color, fontsize=8)
        plt.tight_layout()
        
        # save image
        plt.savefig('orl_test_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Test results saved to 'orl_test_results.png'")

    def test(self):
        self.eval()  # evaluation mode
        test_loader = self.data['test_loader']
        print(f"Testing on {len(test_loader.dataset)} samples")
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.to(device).long()
                outputs = self.forward(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # visualize some test results
        print("\nVisualizing test results...")
        self.visualize_test_results(test_loader)
        
        return all_preds, all_labels

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model()
        print('--start testing...')
        pred_y, true_y = self.test()
        return {'pred_y': pred_y, 'true_y': true_y}