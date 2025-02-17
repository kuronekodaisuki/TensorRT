#import warnings

#warnings.filterwarnings("error")
import argparse
import torch
import torch.nn as nn

from torchvision import models, transforms
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class MyImageFolder(ImageFolder):
    def __init__(self, img_path, category_path, transform=None):
        super(MyImageFolder, self).__init__(img_path, transform)
        self.classes, self.class_to_idx = self._my_classes()
        self.samples = self._make_dataset(self.samples)
        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]
        with open(category_path, "r") as f:
            self.classes = [s.strip() for s in f.readlines()]
        #print(self.classes)

    def _my_classes(self):
        #classes = ['duck', 'wolf']
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        return self.classes, class_to_idx

    def _make_dataset(self, samples):
        n = len(samples)
        ds = [None] * n

        for i, (img, cls) in enumerate(samples):
            ds[i] = (img, self._custom_class(cls))

        return ds

    def _custom_class(self, cls):
        return cls

def Evaluate(args):
    dataset = args.dataset
    category = args.category
    kernel_size = args.kernel
    resize = args.size
    cropsize = args.crop

    print(kernel_size, resize)

    # デバイスの設定 (GPUが利用可能ならGPUを使用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # WideResNet50_2のロード
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)  # 学習済みのモデル
    print(model.conv1)
    """
    model.conv1 = nn.Conv2d(in_channels=3,
                        out_channels=model.conv1.out_channels,
                        kernel_size=(kernel_size, kernel_size),
                        stride=(2, 2),
                        padding=3,
                        bias=False)
    #print(model.conv1)
    """
    layer3 = model.layer3
    layer4 = model.layer4
    #print(layer3, layer4)

    model = model.to(device)  # モデルをデバイスに転送
    model.eval()  # 評価モードに設定

    # データの準備 (例としてImageFolderでデータをロード)
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop((cropsize, cropsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 例: test_data ディレクトリ内の画像データを読み込む
    test_dataset = MyImageFolder(dataset, category, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # 予測と実際のラベルのリストを格納するためのリスト
    all_preds = []
    all_labels = []

    # 予測の実行
    with torch.no_grad():
        for images, labels in test_loader:
            #print(labels)
            images = torch.tensor(images).clone().detach().to(device)

            # モデルによる予測
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #print(all_preds, all_labels)

    # F1スコアの計算 (バイナリ分類かマルチクラス分類に応じて設定)
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 'macro' や 'micro' も指定可能

    print(f'F1スコア: {f1}')

    # Confusion Matrixの計算
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(conf_matrix)

    # Confusion Matrixの可視化
    if args.show == True:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        #plt.colorbar()

        # ラベルの表示
        classes = test_dataset.classes  # クラスラベル
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        # 各セルに値を表示
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='~/Dataset/imagenet/val')
    parser.add_argument('-c', '--category', default="imagenet_classes.txt")
    parser.add_argument('-s', '--size', default=256, type=int)
    parser.add_argument('--crop', default=224, type=int)
    parser.add_argument('-k', '--kernel', default=7, type =int)
    parser.add_argument('--show', type=bool, default=False)
    Evaluate(parser.parse_args())

