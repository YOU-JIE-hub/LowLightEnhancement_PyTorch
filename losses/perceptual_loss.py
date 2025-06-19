import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()
        self.resize = resize

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def forward(self, input, target):
        # 調整尺寸與正規化
        if self.resize:
            input = nn.functional.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        input = self.transform(input)
        target = self.transform(target)

        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return self.criterion(input_features, target_features)
