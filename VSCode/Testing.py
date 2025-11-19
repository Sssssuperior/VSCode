import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset_test import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models_Test.ImageDepthNet import ImageDepthNet
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from torch.utils import data
import numpy as np
import os

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 3, 352, 352).cuda()
    x2 = torch.FloatTensor(1, 3, 352, 352).cuda()
    return  dict(image_Input=x1, depth_Input=x2, task = 'RGBD')
    
def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(model_path))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(model_path))

    test_paths_total = [args.RGBDtest_paths, args.RGBTtest_paths, args.RGBVtest_paths, args.CODRGBtest_paths, args.CODRGBVtest_paths]
    task_total = ['RGBD', 'RGBT', 'RGBV', 'CODRGB', 'CODRGBV']
    data_root = [args.RGBDdata_root, args.RGBTdata_root, args.RGBVdata_root, args.CODRGBdata_root, args.CODRGBVdata_root]
    
    for k in range(len(test_paths_total)):
        test_paths = test_paths_total[k].split('+')
        task = task_total[k]
        for test_dir_img in test_paths:

            test_dataset = get_loader(test_dir_img, data_root[k], args.img_size, mode='test', task=task)

            test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
            print('''
                               Starting testing:
                                   dataset: {}
                                   Testing size: {}
                               '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

            time_list = []
            for i, data_batch in enumerate(test_loader):
                images, depths, image_w, image_h, image_path = data_batch
                images, depths = Variable(images.cuda()), Variable(depths.cuda())

                starts = time.time()
                #flops, params = get_model_complexity_info(net, ((1,3,352,352),(1,3,352,352),(1)),input_constructor=prepare_input,as_strings=True,print_per_layer_stat=True,verbose=True)
                #print(params)
                outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s = net(images, depths, task)
                ends = time.time()
                time_use = ends - starts
                time_list.append(time_use)

                mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
                mask_1_16_s, mask_1_8_s, mask_1_4_s, mask_1_1_s = outputs_saliency_s

                image_w, image_h = int(image_w[0]), int(image_h[0])

                output_s = F.sigmoid(mask_1_1)
                output_s_s = F.sigmoid(mask_1_1_s)

                output_s = output_s.data.cpu().squeeze(0)
                output_s_s = output_s_s.data.cpu().squeeze(0)

                transform = trans.Compose([
                    transforms.ToPILImage(),
                    trans.Scale((image_w, image_h))
                ])
                output_s = transform(output_s)
                output_s_s = transform(output_s_s)

                dataset = test_dir_img.split('/')[0]
                file = image_path[0].split('/')[-3]
                filename = image_path[0].split('/')[-1].split('.')[0]

                if task == "RGBV" or task == "CODRGBV":
                    save_test_path2 = args.save_test_path_root + task + "/" + dataset + '/RGB_VST_yuanshi/' + file
                else:
                    save_test_path2 = args.save_test_path_root + task + "/" + dataset + '/RGB_VST_yuanshi/'

                if not os.path.exists(save_test_path2):
                    os.makedirs(save_test_path2)
                output_s_s.save(os.path.join(save_test_path2, filename + '.png'))

            print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list) * 1000))





