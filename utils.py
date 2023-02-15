import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn import functional as F
import wandb

def train(model, device, train_loader, optimizer, epoch, steps_per_epoch):
  # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
  model.train()
  train_total = 0
  train_correct = 0

  # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
  for batch_idx, (data, target) in enumerate(train_loader, start=0):
    if batch_idx > steps_per_epoch:
      break
    # Load the input features and labels from the training dataset
    data, target = data.to(device), target.to(device)
    
    # Reset the gradients to 0 for all learnable weight parameters
    optimizer.zero_grad()
    
    # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
    output = model(data)
    
    # Define our loss function, and compute the loss
    loss = F.nll_loss(output, target)

    scores, predictions = torch.max(output.data, 1)
    train_total += target.size(0)
    train_correct += int(sum(predictions == target))
            
    # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
    loss.backward()
    
    # Update the neural network weights
    optimizer.step()

  acc = round((train_correct / train_total) * 100, 2)
  print('Epoch [{}], Loss: {}, Accuracy: {}, '.format(epoch, loss.item(), acc), end='')
  wandb.log({'Train Loss': loss.item(), 'Train Accuracy': acc})


def test(model, device, test_loader, classes):
  # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
  model.eval()
  
  test_loss = 0
  test_total = 0
  test_correct = 0

  with torch.no_grad():
      for data, target in test_loader:
          # Load the input features and labels from the test dataset
          data, target = data.to(device), target.to(device)
          
          # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
          output = model(data)
          
          # Compute the loss sum up batch loss
          test_loss += F.nll_loss(output, target, reduction='sum').item()
          
          scores, predictions = torch.max(output.data, 1)
          test_total += target.size(0)
          test_correct += int(sum(predictions == target))
          
  acc = round((test_correct / test_total) * 100, 2)
  print(' Test_loss: {}, Test_accuracy: {}'.format(test_loss/test_total, acc))
  wandb.log({'Test Loss': test_loss/test_total, 'Test Accuracy': acc})




def plot_results(train_loss, train_acc, test_loss, test_acc, epochs):
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plotting the training loss
    sns.lineplot(x=range(epochs), y=train_loss, ax=ax1, label='Train Loss')
    sns.lineplot(x=range(epochs), y=test_loss, ax=ax1, label='Test Loss')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss vs Epoch')
    ax1.legend()

    # Plotting the accuracy
    sns.lineplot(x=range(epochs), y=train_acc, ax=ax2, label='Train Accuracy')
    sns.lineplot(x=range(epochs), y=test_acc, ax=ax2, label='Test Accuracy')
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy vs Epoch')
    ax2.legend()

    plt.show()
