# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import numpy as np


# class LabelSmoothing(nn.Module):
#     # "Implement label smoothing."

#     def __init__(self, size, smoothing=0.0):

#         super(LabelSmoothing, self).__init__()

#         # self.criterion = nn.KLDivLoss(size_average=False)
#         self.criterion = nn.CrossEntropyLoss()
        
#         #self.padding_idx = padding_idx

#         self.confidence = 1.0 - smoothing

#         self.smoothing = smoothing

#         self.size = size

#         self.true_dist = None

#     def forward(self, x, target):
#         """
#         x represents the input (M, N) N samples, M is the total number of classes, each class probability log P
#                  target represents a label (M,)
#         """
#         assert x.size(1) == self.size
#         x = x.log()
#         true_dist = x.data.clone()# First deep copied
#         #print true_dist
#         true_dist.fill_(self.smoothing / (self.size - 1))#otherwise formula
#         #print true_dist
#         # Becomes one-hot encoding, 1 represents a filling columns,
#         # Target.data.unsqueeze (1) represents the index, confidence in the digital filler represents
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

#         self.true_dist = true_dist
#         print(x.shape,true_dist.shape)
        
#         return self.criterion(x, Variable(true_dist, requires_grad=False))



# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# if __name__=="__main__":
# # Example of label smoothing.
 
#     crit = LabelSmoothingLoss(classes=5, smoothing= 0.1)
#     #predict.shape 3 5
#     predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
    
#                                  [0, 0.9, 0.2, 0.1, 0], 
    
#                                  [1, 0.2, 0.7, 0.1, 0]])

#     v = crit(Variable(predict),
    
#              Variable(torch.LongTensor([2, 1, 0])))
#     print(v)
    
#     # Show the target distributions expected by the system.
    
#     plt.imshow(crit.true_dist)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))