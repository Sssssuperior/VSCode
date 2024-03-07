import os

import torch
from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import collections
collections.Iterable = collections.abc.Iterable


def load_list(dataset_list, data_root, task):

    images = []
    depths = []
    labels = []
    contours = []

    dataset_list = dataset_list.split('+')

    if task == "RGB":
        img_root = data_root + dataset_list[0] + '/DUTS-TR-Image/'
        img_files = os.listdir(img_root)

        for img in img_files:
            images.append(img_root + img[:-4] + '.jpg')
            labels.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Mask/') + img[:-4] + '.png')
            contours.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Contour/') + img[:-4] + '.png')

    elif task == "RGBD":
        for dataset_name in dataset_list:
            depth_root = data_root+ dataset_name + '/trainset/depth/'
            depth_files = os.listdir(depth_root)

            for depth in depth_files:
                images.append(depth_root.replace('/depth/', '/RGB/') + depth[:-4]+'.jpg')
                depths.append(depth_root + depth)
                labels.append(depth_root.replace('/depth/', '/GT/') + depth[:-4]+'.png')
                contours.append(depth_root.replace('/depth/', '/contour/') + depth[:-4]+'.png')

    elif task == "RGBT":
        for dataset_name in dataset_list:

            depth_root = data_root + dataset_name + '/Train/T/'
            depth_files = os.listdir(depth_root)

            for depth in depth_files:
                images.append(depth_root.replace('/T/', '/RGB/') + depth[:-4] + '.jpg')
                depths.append(depth_root + depth)
                labels.append(depth_root.replace('/T/', '/GT/') + depth[:-4] + '.png')
                contours.append(depth_root.replace('/T/', '/Contour/') + depth[:-4] + '.png')

    elif task == "CODRGB":
        for dataset_name in dataset_list:
            if dataset_name == "COD10K":
              img_root = data_root + dataset_name + '/GT/'
              img_files = os.listdir(img_root)

              for img in img_files:
                  labels.append(img_root + img[:-4] + '.png')
                  images.append(img_root.replace('/GT/', '/RGB/') + img[:-4] + '.jpg')
                  contours.append(img_root.replace('/GT/', '/Contour/') + img[:-4] + '.png')
            else:
              img_root = data_root + dataset_name + '/RGB/'
              img_files = os.listdir(img_root)

              for img in img_files:
                  images.append(img_root + img[:-4] + '.jpg')
                  labels.append(img_root.replace('/RGB/', '/GT/') + img[:-4] + '.png')
                  contours.append(img_root.replace('/RGB/', '/Contour/') + img[:-4] + '.png')

    elif task == "CODRGBV":
        img_root = data_root + dataset_list[0] + '/TrainDataset_per_sq/'
        files = os.listdir(img_root)

        for file in files:
            image_root = img_root + "/" + file + "/Flow/"
            image_files = os.listdir(image_root)
            for img in image_files:
                depths.append(image_root + img[:-4] + '.png')
                images.append(image_root.replace('/Flow/', '/Imgs/') + img[:-4] + '.jpg')
                labels.append(image_root.replace('/Flow/', '/GT/') + img[:-4] + '.png')
                contours.append(image_root.replace('/Flow/', '/Contour/') + img[:-4] + '.png')

    else:
        for dataset_name in dataset_list:

            depth_root = data_root + dataset_name
            files = os.listdir(depth_root)

            for file in files:
                depth_root_file = depth_root + "/" + file + '/Flow/'
                depth_files = os.listdir(depth_root_file)
                for depth in depth_files:
                    if dataset_name == "DAVSOD":
                        images.append(depth_root_file.replace('/Flow/', '/RGB/') + depth[:-4] + '.png')
                    else:
                        images.append(depth_root_file.replace('/Flow/', '/RGB/') + depth[:-4] + '.jpg')
                    depths.append(depth_root_file + depth)
                    labels.append(depth_root_file.replace('/Flow/', '/GT/') + depth[:-4] + '.png')
                    contours.append(depth_root_file.replace('/Flow/', '/Contour/') + depth[:-4] + '.png')

    return images, depths, labels, contours


def load_test_list(test_path, data_root, task):

    images = []
    depths = []

    if task == 'RGBD':
        if test_path in ['NJUD', 'NLPR', 'DUTLF-Depth', 'ReDWeb-S']:
            depth_root = data_root + test_path + '/testset/depth/'
        else:
            depth_root = data_root + test_path + '/depth/'

        depth_files = os.listdir(depth_root)

        for depth in depth_files:
            images.append(depth_root.replace('/depth/', '/RGB/') + depth[:-4] + '.jpg')
            depths.append(depth_root + depth)

    elif task == 'RGBT':
        if test_path in ['VT5000']:
            depth_root = data_root + test_path + '/Test/T/'
        else:
            depth_root = data_root + test_path + '/T/'

        depth_files = os.listdir(depth_root)

        for depth in depth_files:
            images.append(depth_root.replace('/T/', '/RGB/') + depth[:-4] + '.jpg')
            depths.append(depth_root + depth)

    else:
        depth_root = (data_root + test_path).replace('/Train/', '/Test/')
        files = os.listdir(depth_root)
        for file in files:
            depth_roott = depth_root + '/' + file + '/OF_FlowNet2/'

            depth_files = os.listdir(depth_roott)

            for depth in depth_files:
                images.append(depth_roott.replace('/OF_FlowNet2/', '/Frame/') + depth[:-4] + '.jpg')
                depths.append(depth_roott + depth)


    return images, depths


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform_RGB, transformCOD_RGB, transformCOD_RGBV, transformCOD_Video, transformdepth_RGB, transform_depth, transformthermal_RGB, transform_thermal, transformvideo_RGB, transform_video,  mode, task=None, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            self.image_path, self.depth_path, self.label_path, self.contour_path = load_list(dataset_list, data_root, task)
        else:
            self.image_path, self.depth_path = load_test_list(dataset_list, data_root, task)

        self.transform_RGB = transform_RGB
        self.transformCOD_RGB = transformCOD_RGB
        self.transformCOD_RGBV = transformCOD_RGBV
        self.transformCOD_Video = transformCOD_Video
        self.transformdepth_RGB = transformdepth_RGB
        self.transform_depth = transform_depth
        self.transformthermal_RGB = transformthermal_RGB
        self.transform_thermal = transform_thermal
        self.transformvideo_RGB = transformvideo_RGB
        self.transform_video = transform_video
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size
        self.task = task


    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]
        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])
        if self.task == "RGB" or self.task == "CODRGB":
            depth = Image.new('RGB', (image_w, image_h), (0,0,0))
        else:
            depth = Image.open(self.depth_path[item]).convert('RGB')

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            contour = Image.open(self.contour_path[item]).convert('L')
            random_size = self.scale_size

            new_img = trans.Scale((random_size, random_size))(image)
            new_depth = trans.Scale((random_size, random_size))(depth)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)
            new_contour = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(contour)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_depth = new_depth.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_contour = new_contour.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_depth = new_depth.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
                new_contour = new_contour.transpose(Image.FLIP_LEFT_RIGHT)

            if self.task == "RGB":
                new_img = self.transform_RGB(new_img)
                new_depth = self.transform_depth(new_depth)
            elif self.task == "RGBD":
                new_img = self.transformdepth_RGB(new_img)
                new_depth = self.transform_depth(new_depth)
            elif self.task == "RGBT":
                new_img = self.transformthermal_RGB(new_img)
                new_depth = self.transform_thermal(new_depth)
            elif self.task == "CODRGB":
                new_img = self.transformCOD_RGB(new_img)
                new_depth = self.transform_depth(new_depth)
            elif self.task == "CODRGBV":
                new_img = self.transformCOD_RGBV(new_img)
                new_depth = self.transformCOD_Video(new_depth)
            else:
                new_img = self.transformvideo_RGB(new_img)
                new_depth = self.transform_video(new_depth)

            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)

            contour_14 = self.label_14_transform(new_contour)
            contour_28 = self.label_28_transform(new_contour)
            contour_56 = self.label_56_transform(new_contour)
            contour_112 = self.label_112_transform(new_contour)
            contour_224 = self.t_transform(new_contour)

            return new_img, new_depth, label_224, label_14, label_28, label_56, label_112, \
                   contour_224, contour_14, contour_28, contour_56, contour_112

        else:

            image = self.transform(image)
            depth = self.depth_transform(depth)

            return image, depth, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train', task=None):

    if mode == 'train':
        transform_RGB = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.492, 0.463, 0.298], [0.222, 0.219, 0.225]),
        ])
        transformCOD_RGB = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.447, 0.443, 0.364], [0.192, 0.185, 0.180]),
        ])
        transformCOD_RGBV = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.437, 0.476, 0.427], [0.159, 0.157, 0.162]),
        ])
        transformCOD_Video = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.763, 0.763, 0.805], [0.114, 0.102, 0.099]),
        ])

        transformdepth_RGB = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.445, 0.425, 0.397], [0.209, 0.205, 0.204]),])

        transform_depth = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.507, 0.507, 0.507], [0.232, 0.232, 0.232]),
        ])
        
        transformtherml_RGB = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.532, 0.619, 0.552], [0.187, 0.170, 0.181]),
        ])

        transform_thermal = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.754, 0.361, 0.342], [0.180, 0.210, 0.171]),
        ])
        
        transformvideo_RGB = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.432, 0.404, 0.366], [0.241, 0.230, 0.230]),
        ])

        transform_video = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.866, 0.832, 0.817], [0.100, 0.102, 0.113]),
        ])

        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])
        label_14_transform = trans.Compose([
            trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        scale_size = 384
    else:
        transform_RGB = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        transformdepth_RGB = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.445, 0.425, 0.397], [0.209, 0.205, 0.204]),
        ])

        transform_depth = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.507, 0.507, 0.507], [0.232, 0.232, 0.232]),
        ])
        
        transformtherml_RGB = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.532, 0.619, 0.552], [0.187, 0.170, 0.181]),
        ])

        transform_thermal = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.754, 0.361, 0.342], [0.180, 0.210, 0.171]),
        ])
        
        transformvideo_RGB = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.449, 0.456, 0.427], [0.203, 0.180, 0.196]),
        ])

        transform_video = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.706, 0.748, 0.826], [0.113, 0.094, 0.088]),
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform_RGB, transformCOD_RGB, transformCOD_RGBV, transformCOD_Video, transformdepth_RGB, transform_depth, transformtherml_RGB, transform_thermal, transformvideo_RGB, transform_video, mode, task, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform_RGB, transformCOD_RGB, transformdepth_RGB, transform_depth, transformtherml_RGB, transform_thermal, transformvideo_RGB, transform_video, mode, task)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset