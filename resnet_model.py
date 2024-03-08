import torch
import torch.nn as nn

#resnet18
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):#################### =1
        super().__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity # skip connection
        x = self.relu(x)

        return x

# 18 layer ResNet
class ResNet(nn.Module): # [ 2, 2, 2, 2 ]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride = 2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride = 2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    

    # Resnet layers, should revisit this at the end
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          out_channels, 
                          kernel_size = 1, 
                          stride = stride,
                          bias = False
                          ),
                        nn.BatchNorm2d(out_channels))

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride))
        # for ResNet 50+
        #self.in_channels = out_channels*4

        self.in_channels = out_channels

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return(nn.Sequential(*layers))

def ResNet18(img_channels = 3, num_classes = 1000):
    return ResNet(block, [2, 2, 2, 2], img_channels, num_classes)


def test():
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet18(img_channels=3, num_classes=1000).to(device)
    y = net(torch.randn(BATCH_SIZE, 3, 224, 224)).to(device)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(net)
    print(y.size())


if __name__ == "__main__":
    test()