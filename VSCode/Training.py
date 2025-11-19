import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset_train import get_loader
import math
from Models_Train.ImageDepthNet import ImageDepthNet
import os

import torch.nn.functional as F

def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss,epoch_loss_s, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    epoch_loss_s = str(epoch_loss_s)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss_s' + epoch_loss_s + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()

def IOU_loss(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU/b

def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

def main(local_rank, num_gpus, args):
    cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = ImageDepthNet(args)
    net.train()
    net.cuda()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])

    train_datasetR = get_loader(args.RGBtrainset, args.RGBdata_root, args.img_size, mode='train', task='RGB')
    train_datasetD = get_loader(args.RGBDtrainset, args.RGBDdata_root, args.img_size, mode='train', task ='RGBD')
    train_datasetT = get_loader(args.RGBTtrainset, args.RGBTdata_root, args.img_size, mode='train', task = 'RGBT')
    train_datasetV = get_loader(args.RGBVtrainset, args.RGBVdata_root, args.img_size, mode='train', task ='RGBV')
    train_datasetCR = get_loader(args.CODRGBtrainset, args.CODRGBdata_root, args.img_size, mode='train', task='CODRGB')
    train_datasetCV = get_loader(args.CODRGBVtrainset, args.CODRGBVdata_root, args.img_size, mode='train', task='CODRGBV')

    samplerR = torch.utils.data.distributed.DistributedSampler(
        train_datasetR,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    samplerD = torch.utils.data.distributed.DistributedSampler(
        train_datasetD,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    samplerT = torch.utils.data.distributed.DistributedSampler(
        train_datasetT,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    samplerV = torch.utils.data.distributed.DistributedSampler(
        train_datasetV,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    samplerCR = torch.utils.data.distributed.DistributedSampler(
        train_datasetCR,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    samplerCV = torch.utils.data.distributed.DistributedSampler(
        train_datasetCV,
        num_replicas=num_gpus,
        rank=local_rank,
    )

    train_loaderR = torch.utils.data.DataLoader(train_datasetR, batch_size=args.batch_size // 6, num_workers=6,
                                                pin_memory=True,
                                                sampler=samplerR,
                                                drop_last=True,
                                                )
    train_loaderD = torch.utils.data.DataLoader(train_datasetD, batch_size=args.batch_size // 6, num_workers=6,
                                               pin_memory=True,
                                               sampler=samplerD,
                                               drop_last=True,
                                               )
    train_loaderT = torch.utils.data.DataLoader(train_datasetT, batch_size=args.batch_size // 6, num_workers=6,
                                                pin_memory=True,
                                                sampler=samplerT,
                                                drop_last=True,
                                                )
    train_loaderV = torch.utils.data.DataLoader(train_datasetV, batch_size=args.batch_size // 6, num_workers=6,
                                                pin_memory=True,
                                                sampler=samplerV,
                                                drop_last=True,
                                                )
    train_loaderCR = torch.utils.data.DataLoader(train_datasetCR, batch_size=args.batch_size // 6, num_workers=6,
                                                pin_memory=True,
                                                sampler=samplerCR,
                                                drop_last=True,
                                                )
    train_loaderCV = torch.utils.data.DataLoader(train_datasetCV, batch_size=args.batch_size // 6, num_workers=6,
                                                 pin_memory=True,
                                                 sampler=samplerCV,
                                                 drop_last=True,
                                                 )

    train_loaderR_length = len(train_loaderR)
    train_loaderD_length = len(train_loaderD)
    train_loaderT_length = len(train_loaderT)
    train_loaderV_length = len(train_loaderV)
    train_loaderCR_length = len(train_loaderCR)
    train_loaderCV_length = len(train_loaderCV)

    batch_size_div = args.batch_size // 6
            
    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, (train_loaderD_length+train_loaderT_length+train_loaderV_length+train_loaderR_length+train_loaderCR_length+train_loaderCV_length)*2))

    N_train = train_loaderD_length * batch_size_div + train_loaderT_length * batch_size_div + train_loaderV_length * batch_size_div + train_loaderR_length * batch_size_div +train_loaderCR_length *batch_size_div +train_loaderCV_length *batch_size_div
    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    whole_iter_num = 0

    iter_num = math.ceil(max(train_loaderR_length, train_loaderD_length, train_loaderT_length, train_loaderV_length, train_loaderCR_length))
    print(iter_num)

    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0
        epoch_loss_s = 0
        
        train_loaderD_iter = iter(train_loaderD)
        train_loaderT_iter = iter(train_loaderT)
        train_loaderR_iter = iter(train_loaderR)
        train_loaderCV_iter = iter(train_loaderCV)
        train_loaderCR_iter = iter(train_loaderCR)

        for i, data_batchV in enumerate(train_loaderV):
            if (i + 1) > iter_num: break
            
            try:
                data_batchD = next(train_loaderD_iter)
            except StopIteration:
                train_loaderD_iter = iter(train_loaderD)
                data_batchD = next(train_loaderD_iter)
            try:
                data_batchT = next(train_loaderT_iter)
            except StopIteration:
                train_loaderT_iter = iter(train_loaderT)
                data_batchT = next(train_loaderT_iter)
            try:  
                data_batchR = next(train_loaderR_iter)
            except StopIteration:
                train_loaderR_iter = iter(train_loaderR)
                data_batchR = next(train_loaderR_iter)
            try:  
                data_batchCV = next(train_loaderCV_iter)
            except StopIteration:
                train_loaderCV_iter = iter(train_loaderCV)
                data_batchCV = next(train_loaderCV_iter)
            try: 
                data_batchCR = next(train_loaderCR_iter)
            except StopIteration:
                train_loaderCR_iter = iter(train_loaderCR)
                data_batchCR = next(train_loaderCR_iter)
            

            data_batch = [torch.cat((t1, t2, t3, t4, t5, t6), dim=0) for t1, t2, t3, t4, t5, t6 in zip(data_batchD, data_batchT, data_batchV, data_batchCV, data_batchR, data_batchCR)]

            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                                     Variable(depths.cuda(local_rank, non_blocking=True)), \
                                                     Variable(label_224.cuda(local_rank, non_blocking=True)), \
                                                     Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()), \
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                              Variable(contour_28.cuda()), \
                                                              Variable(contour_56.cuda()), Variable(contour_112.cuda())

            outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s, domain_prompt_total, task_prompt_total2, task_prompt_total1 = net(images, depths)

            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour
            mask_1_16_s, mask_1_8_s, mask_1_4_s, mask_1_1_s = outputs_saliency_s
            cont_1_16_s, cont_1_8_s, cont_1_4_s, cont_1_1_s = outputs_contour_s
            
            # saliency loss
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)
            loss1_i = IOU_loss(sigmoid(mask_1_1),label_224)

            # contour loss
            c_loss5 = criterion(cont_1_16, contour_14)
            c_loss4 = criterion(cont_1_8, contour_28)
            c_loss3 = criterion(cont_1_4, contour_56)
            c_loss1 = criterion(cont_1_1, contour_224)
            
            loss5_s = criterion(mask_1_16_s, label_14)
            loss4_s = criterion(mask_1_8_s, label_28)
            loss3_s = criterion(mask_1_4_s, label_56)
            loss1_s = criterion(mask_1_1_s, label_224)
            loss1_s_i = IOU_loss(sigmoid(mask_1_1_s),label_224)

            # contour loss
            c_loss5_s = criterion(cont_1_16_s, contour_14)
            c_loss4_s = criterion(cont_1_8_s, contour_28)
            c_loss3_s = criterion(cont_1_4_s, contour_56)
            c_loss1_s = criterion(cont_1_1_s, contour_224)
            
            # discrimination loss
            r_rd = torch.log(torch.abs(F.cosine_similarity(domain_prompt_total[0].view(-1), domain_prompt_total[1].view(-1), dim=0))+1)
            r_rt = torch.log(torch.abs(F.cosine_similarity(domain_prompt_total[0].view(-1), domain_prompt_total[2].view(-1), dim=0))+1)
            r_rv = torch.log(torch.abs(F.cosine_similarity(domain_prompt_total[0].view(-1), domain_prompt_total[3].view(-1), dim=0))+1)
            r_dt = torch.log(torch.abs(F.cosine_similarity(domain_prompt_total[1].view(-1), domain_prompt_total[2].view(-1), dim=0))+1)
            r_dv = torch.log(torch.abs(F.cosine_similarity(domain_prompt_total[1].view(-1), domain_prompt_total[3].view(-1), dim=0))+1)
            r_tv = torch.log(torch.abs(F.cosine_similarity(domain_prompt_total[2].view(-1), domain_prompt_total[3].view(-1), dim=0))+1)

            r_sc_en = torch.log(torch.abs(F.cosine_similarity(task_prompt_total2[0].view(-1), task_prompt_total2[1].view(-1), dim=0))+1)
            r_sc_de = torch.log(torch.abs(F.cosine_similarity(task_prompt_total1[0].view(-1), task_prompt_total1[1].view(-1), dim=0))+1)
            
            cosin_loss = (r_rd + r_rt + r_rv + r_dt + r_dv + r_tv + r_sc_en + r_sc_de) / 8

            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5
            contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + loss_weights[4] * c_loss5
            img_total_loss_s = loss_weights[0] * loss1_s  + loss_weights[2] * loss3_s + loss_weights[3] * loss4_s + loss_weights[4] * loss5_s
            contour_total_loss_s = loss_weights[0] * c_loss1_s + loss_weights[2] * c_loss3_s+ loss_weights[3] * c_loss4_s + loss_weights[4] * c_loss5_s
            
            img_total_loss_i = loss_weights[0] * loss1_i
            img_total_loss_s_i = loss_weights[0] * loss1_s_i

            total_loss = img_total_loss + contour_total_loss + img_total_loss_s + contour_total_loss_s + img_total_loss_i + img_total_loss_s_i + cosin_loss
            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()
            epoch_loss_s += loss1_s.cpu().data.item()

            accum_steps = 1
            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f}--- saliency loss2: {4:.6f}'.format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item(), cosin_loss.item()))

            total_loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loaderV):
                optimizer.step()
                optimizer.zero_grad()
                whole_iter_num += 1

                if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                    optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                    save_dir = './loss.txt'
                    save_lr(save_dir, optimizer)
                    print('have updated lr!!')

                if (local_rank == 0) and (whole_iter_num == args.train_steps):
                    torch.save(net.state_dict(), args.save_model_dir + 'RGB_VST.pth')

                if whole_iter_num == args.train_steps:
                    return 0

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num,epoch_loss_s/iter_num, epoch+1)






