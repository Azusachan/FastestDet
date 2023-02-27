import torch
import torch.nn as nn

class Conv1x1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 =  nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(output_channels),
                                      nn.ReLU(inplace=True)
                                     )
    
    def forward(self, x):
        return self.conv1x1(x)

class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups = input_channels, bias = False),
                                     nn.BatchNorm2d(input_channels),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(output_channels)
                                    ) 
    
    def forward(self, x):
        return self.conv5x5(x)

class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
   
 
class DetectHead(nn.Module):
    def __init__(self, input_channels, category_num):
        super(DetectHead, self).__init__()
        self.conv1x1 =  Conv1x1(input_channels, input_channels)

        self.obj_layers = Head(input_channels, 1)
        self.reg_layers = Head(input_channels, 4)
        self.cls_layers = Head(input_channels, category_num)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1x1(x)
        
        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))

        return torch.cat((obj, reg, cls), dim =1)

 #SPD space_to_depth
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
