from torch.utils import data
import os
from PIL import Image
import cv2
import numpy as np


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root, task):

        if task == "RGBV":
            pred_names = []
            gt_names = []
            label_root = label_root.replace('/Train/', '/Test/')
            files = os.listdir(label_root)
            for file in files:
                root = os.path.join(label_root, file, 'GT')
                imgs = os.listdir(root)
                for img in imgs:
                    pred_names.append(os.path.join(file, img))
                    gt_names.append(os.path.join(file, 'GT', img))
        elif task == "CODRGB":
            pred_names = []
            gt_names = []
            if "COD10K" in label_root:
                label_root = label_root.replace('/Train/', '/Test/')
                files = os.listdir(label_root)
                for file in files:
                    pred_names.append(os.path.join(file))
                    gt_names.append(os.path.join(file))
            else:
                label_root = label_root.replace('/Train/', '/Test/')
                files = os.listdir(pred_root)
                for file in files:
                    pred_names.append(os.path.join(file))
                    gt_names.append(os.path.join(file))
        elif task == "CODRGBD":
            pred_names = []
            gt_names = []
            if "COD10K" in label_root:
                label_root = label_root.replace('/Train/', '/Test/')
                files = os.listdir(label_root)
                for file in files:
                    pred_names.append(os.path.join(file))
                    gt_names.append(os.path.join(file))
            else:
                label_root = label_root.replace('/Train/', '/Test/')
                files = os.listdir(pred_root)
                for file in files:
                    pred_names.append(os.path.join(file))
                    gt_names.append(os.path.join(file))
        elif task == "CODRGBV":
            pred_names = []
            gt_names = []
            files = os.listdir(pred_root)
            label_root = label_root.replace('/Train/', '/Test/')
            if 'CAD' in label_root:
                for file in files:
                    path = os.path.join(label_root, file, 'GT')
                    imgs = os.listdir(path)
                    for img in imgs:
                        pred_names.append(os.path.join(file, file + '_' + img[:-7] + '.png'))
                        gt_names.append(os.path.join(file, 'GT', img))
            else:
                for file in files:
                    path = os.path.join(label_root, file, 'GT')
                    imgs = os.listdir(path)
                    for img in imgs:
                        pred_names.append(os.path.join(file, img))
                        gt_names.append(os.path.join(file, 'GT', img))

        else:
            pred_names = os.listdir(pred_root)
            gt_names = pred_names
        
        if pred_root.split('/')[-2] == 'PASCAL-S':
            # remove the following image
            if '424.png' in pred_names:
                pred_names.remove('424.png')
            if '460.png' in pred_names:
                pred_names.remove('460.png')
            if '359.png' in pred_names:
                pred_names.remove('359.png')
            if '408.png' in pred_names:
                pred_names.remove('408.png')
            if '622.png' in pred_names:
                pred_names.remove('622.png')
        
        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), pred_names))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), gt_names))
        self.task = task

    def __getitem__(self, item):

        pred = Image.open(self.image_path[item]).convert('L')
        
        if (self.task == "RGBV"):
            gt = Image.open(self.label_path[item][:-4]+'.png').convert('L')
        elif 'VT1000' in self.image_path[item] or 'VT821' in self.image_path[item]:
            gt = Image.open(self.label_path[item][:-4]+'.jpg').convert('L')
        else:
            gt = Image.open(self.label_path[item]).convert('L')
            #gt = cv2.imread(self.label_path[item], cv2.IMREAD_GRAYSCALE)
            #gt = ((gt >= 127).astype(np.uint8))*255
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
