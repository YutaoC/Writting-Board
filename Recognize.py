import cv2
import numpy as np
from torchvision import transforms


"""Process the input image and feed it into the model"""


def recognize(path, model):
    """path --> path to the image.
       model --> the pretrained model.
       Return --> top1 result and top5 result.
    """
    # process the input data
    inputimg = cv2.imread(path)
    inputimg = cv2.resize(inputimg, (28, 28))
    inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)  # to gary
    inputimg = np.fliplr(inputimg)  # flip left/right
    inputimg = np.rot90(inputimg)  # rotate 90 degree
    trans = transforms.ToTensor()  # to tensor transform
    inputimg = trans(inputimg)  # use the transform
    inputimg.unsqueeze_(0)  # add a dimension

    # predict
    output = model(inputimg)
    pred_1 = output.argmax(dim=1, keepdim=True)
    _, pred_5 = output.topk(5, dim=1)
    return pred_1.item(), pred_5.tolist()[0]
