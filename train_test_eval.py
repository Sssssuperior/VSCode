import os
import torch
import Training
import Testing
from Evaluation import main
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33115', type=str, help='init_method')
    parser.add_argument('--RGBdata_root', default='./Data/RGB/', type=str, help='rgbdata path')
    parser.add_argument('--RGBDdata_root', default='./Data/RGBD/', type=str, help='rgbddata path')
    parser.add_argument('--RGBTdata_root', default='./Data/RGBT/',type=str, help='rgbtdata path')
    parser.add_argument('--RGBVdata_root', default='./Data/Video/Train/',type=str, help='rgbvdata path')
    parser.add_argument('--CODRGBdata_root', default='./Data/COD/rgb/Train/', type=str, help='codrgbdata path')
    parser.add_argument('--CODRGBDdata_root', default='./Data/COD/rgb/Train/', type=str, help='codrgbdata path')
    parser.add_argument('--CODRGBVdata_root', default='./Data/COD/rgbv/Train/', type=str, help='codrgbvdata path')
    parser.add_argument('--train_steps', default=150000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/swin_tiny_patch4_window7_224.pth', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=12, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=75000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=112500, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--RGBtrainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--RGBDtrainset', default='NJUD+NLPR+DUTLF-Depth', type=str, help='Trainging set')
    parser.add_argument('--RGBTtrainset', default='VT5000', type=str, help='Trainging set')
    parser.add_argument('--RGBVtrainset', default='DAVIS+DAVSOD', type=str, help='Trainging set')
    parser.add_argument('--CODRGBtrainset', default='COD10K+CAMO', type=str, help='Trainging set')
    parser.add_argument('--CODRGBVtrainset', default='MoCA_Mask', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
    parser.add_argument('--dim', default=64, type=int, help='dim')
    parser.add_argument('--encoder_dim', default=[96,192,384,768], type=int, help='dim of each encoder layer')

    parser.add_argument('--domain_num', default=[1,1,1,1], type=list, help='the number of domain prompt')
    parser.add_argument('--task_num', default=[1,1,5,10], type=list, help='the number of task prompt in encoder')
    parser.add_argument('--task_deco_num', default=10, type=list, help='the number of task prompt in decoder')

    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--RGBtest_paths', type=str, default='DUTS/DUTS-TE+ECSSD+HKU-IS+PASCAL-S+DUT-O+BSD')
    parser.add_argument('--RGBDtest_paths', type=str, default='NJUD+NLPR+DUTLF-Depth+STERE+SIP+ReDWeb-S')
    parser.add_argument('--RGBTtest_paths', type=str,default='VT821+VT1000+VT5000')
    parser.add_argument('--RGBVtest_paths', type=str, default='ViSal+DAVSOD-easy-30+DAVSOD-Normal-25+DAVSOD-Difficult-20+DAVIS16+FBMS+SegV2')
    parser.add_argument('--CODRGBtest_paths', type=str, default='COD10K+CAMO+NC4K')
    parser.add_argument('--CODRGBVtest_paths', type=str, default='MoCA_Mask+CAD')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods1', type=str, default='RGB_VST_yuanshi', help='evaluated method name')
    parser.add_argument('--methods2', type=str, default='RGB_VST_yuanshi', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        main.evaluate(args)