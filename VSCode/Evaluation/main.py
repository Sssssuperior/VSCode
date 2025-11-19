import os.path as osp
import os
from .evaluator import Eval_thread
from .dataloader import EvalDataset


def evaluate(args):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir

    method_names2 = args.methods2.split('+')

    threads = []
    gt_dir_all = []

    test_paths_total = [args.RGBtest_paths, args.RGBDtest_paths, args.RGBTtest_paths, args.RGBVtest_paths, args.CODRGBtest_paths, args.CODRGBVtest_paths]
    task_total = ['RGB', 'RGBD', 'RGBT', 'RGBV', 'CODRGB', 'CODRGBV']
    data_root = [args.RGBdata_root, args.RGBDdata_root, args.RGBTdata_root, args.RGBVdata_root, args.CODRGBdata_root, args.CODRGBVdata_root]

    for k in range(len(test_paths_total)):
        test_paths = test_paths_total[k].split('+')
        task = task_total[k]
        gt_dir = data_root[k]
        for dataset_setname in test_paths:

            dataset_name = dataset_setname.split('/')[0]

            for method2 in method_names2:
                if task == "RGBV" or task == "CODRGBV":
                    pred_dir_all = osp.join(pred_dir, task, dataset_name, method2)
                    gt_dir_all = osp.join(gt_dir, dataset_setname)

                else:
                    if dataset_name in ['NJUD', 'NLPR', 'DUTLF-Depth', 'ReDWeb-S']:
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname), 'testset/GT')
                    elif dataset_name in ['VT5000']:
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname), 'Test/GT')
                    elif dataset_name == 'DUTS':
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/DUTS-TE-Mask'
                    else:
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/GT'
                    pred_dir_all = osp.join(pred_dir, task, dataset_name, method2)

            loader = EvalDataset(pred_dir_all, gt_dir_all, task)
            thread = Eval_thread(loader, method2, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
    for thread in threads:
        print(thread.run())