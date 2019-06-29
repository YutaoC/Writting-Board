import torch
import torch.nn as nn
import torch.nn.functional as F


"""Load the pretrained model for recognizing the input image"""


def loadmodel(path, device):
    """path --> the path to the model.
       Ddevice --> 'GPU' or 'CPU'.
       Return --> the model
    """
    # Define the model(same as loaded model)
    class TrainedModel(nn.Module):
        def __init__(self):
            super(TrainedModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 3, 1)
            self.conv3 = nn.Conv2d(50, 64, 3, 1)
            self.linear1 = nn.Linear(3 * 3 * 64, 128)
            self.linear2 = nn.Linear(128, 47)
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.25)

        def forward(self, x):
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

    device = torch.device(device)

    # Declare the model and set the parameters
    model = TrainedModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
