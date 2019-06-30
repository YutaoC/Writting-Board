import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms


# Define my model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Define the layers and its parameters
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.conv3 = nn.Conv2d(50, 64, 3, 1)
        self.linear1 = nn.Linear(3 * 3 * 64, 128)
        self.linear2 = nn.Linear(128, 47)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        # define the forward pass
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2, 2)
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(self.conv3(out))
        out = self.dropout1(out)
        out = out.view(-1, 3 * 3 * 64)
        out = F.relu(self.linear1(out))
        out = self.dropout2(out)
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


# Define the training function
def train(model, device, train_data, optimizer, epoch):
    model.train()  # enable the train model
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # check the progress and the loss
        if batch_idx % 400 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# Definr the testing function
def test(model, device, test_data):
    model.eval()
    # variables used to store the info about accuracy
    test_loss = 0  # loss
    correct_1 = 0  # top 1 correct number
    correct_5 = 0  # top 5 correct number
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # top 1 accuracy
            pred_1 = output.argmax(dim=1, keepdim=True)
            correct_1 += pred_1.eq(target.view_as(pred_1)).sum().item()
            # top 5 accuracy
            _, pred_5 = output.topk(5, dim=1)
            pred_5 = pred_5.t()
            correct = pred_5.eq(target.view(1, -1).expand_as(pred_5))
            correct_5 += correct[:5].view(-1).float().sum(0)
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Top1 Accuracy: {}/{} ({:.0f}%), '
          'Top5 Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct_1, len(test_loader.dataset),
            100. * correct_1 / len(test_loader.dataset),
            correct_5, len(test_loader.dataset),
            100. * correct_5 / len(test_loader.dataset)))


# load the data set (train and test)
train_data = datasets.EMNIST('../data', split='balanced', train=True, download=True,
                             transform=transforms.ToTensor())
test_data = datasets.EMNIST('../data', split='balanced', train=False, download=True,
                            transform=transforms.ToTensor())

# transform it to DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                           shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                          shuffle=True, num_workers=1, pin_memory=True)
device = torch.device("cuda")
model = LeNet()
model.to(device)

# Declear the parameters
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
n_epoch = 15

total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# actual training and testing loop
for epoch in range(n_epoch):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# save the model
model_name = 'EMNISTModel.pt'
path = '/Users/yutao/Desktop/' + model_name
torch.save(model.state_dict(), path)
