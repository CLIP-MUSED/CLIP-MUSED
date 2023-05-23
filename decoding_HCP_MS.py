import numpy as np
import os
import glob
import argparse
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import time
import sys
sys.path.append('/data/home/multi_subject/base_exp')
from dataloader_HCP_brainformer import Dataset_individual, Dataset_individual_test, Dataset_shared
from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss
import scipy.io as sio
import collections
from BrainFormer_zqy import BrainFormer_MS
from torch import cosine_similarity
from losses import *
import pdb


def trainer(args, root_path):
    """
    Args:
    parser.parse_args()
    root_path: where to save exp results (results, model weights and tb logs)
    Returns:
    None
    """
    time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    suffix = '_'.join([time_stamp, 'edim'+str(args.embed_dim), 
                'depth'+str(args.depth), 
                str(args.coe_clf), str(args.coe_orth), str(args.coe_rdm_hlv), str(args.coe_rdm_llv)])
    log_dir = os.path.join(root_path, 'logs', suffix)
    model_dir = os.path.join(root_path, 'models', suffix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # save argparser
    args_dicts = vars(args)
    sio.savemat(os.path.join(model_dir, 'args.mat'), args_dicts)

    list_IDs = {'train': np.genfromtxt(os.path.join(args.list_IDs, 'ListIDs_mean_train_MS.txt'), dtype = 'str'),
                 'val': np.genfromtxt(os.path.join(args.list_IDs, 'ListIDs_mean_val_MS.txt'), dtype = 'str')}
    padding = True
    data_loader = {split: DataLoader(
                Dataset_shared(list_IDs[split],  args.data_root, 
                            num_subs=args.subject_num, volume = (111, 127, 111), 
                            delay = args.delay, padding=padding, 
                            fea_root=args.fea_root, 
                            hlv_fea_model=args.hlv_fea_model, llv_fea_model=args.llv_fea_model, 
                            hlv_dim_rd=args.hlv_dim_rd, llv_dim_rd=args.llv_dim_rd, 
                            llv_layer=args.llv_layer, hlv_layer=args.hlv_layer, split=split,
                            sel_label=args.sel_label, sel_label_path=args.sel_label_path,
                            fea_postproc=args.fea_postproc, threshold=args.threshold), 
                    batch_size=args.batch_size,
                    shuffle=(split=='train'), drop_last=True, num_workers=4) for split in ('train', 'val')}
    dataset_tmp = Dataset_shared(list_IDs['val'],  args.data_root, 
                            num_subs=args.subject_num, volume = (111, 127, 111), 
                            delay = args.delay, padding=padding, 
                            fea_root=args.fea_root, 
                            hlv_fea_model=args.hlv_fea_model, llv_fea_model=args.llv_fea_model, 
                            hlv_dim_rd=args.hlv_dim_rd, llv_dim_rd=args.llv_dim_rd, 
                            llv_layer=args.llv_layer, hlv_layer=args.hlv_layer, split='val',
                            sel_label=args.sel_label, sel_label_path=args.sel_label_path,
                            fea_postproc=args.fea_postproc, threshold=args.threshold)
    class_num = dataset_tmp.class_num
    hlv_fea_dim = dataset_tmp.hfdim
    sub_lists = open(os.path.join(args.list_IDs, 'subject_IDs_train.txt'), 'r').readlines()
    sub_lists = [ele.strip('\n') for ele in sub_lists]
    # print(sub_lists)

    print('The number of category', class_num)
    # build model
    model, param = BrainFormer_MS(depth=args.depth, num_classes=class_num, 
            use_cls_token=args.use_cls_token,
            num_subs=args.subject_num)
    model.cuda()
    optimizer = optim.Adam(param, lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    summary = SummaryWriter(logdir=log_dir)

    met_list = ['average_precision_score', 'roc_auc_score', 'hamming_loss']
    for epoch in range(args.epochs):

        lr = args.learning_rate * (args.gamma ** (epoch // args.step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)	
        print('-' * 10)
        print('Iteration', epoch+1)
        loss_list = ['tot_loss', 'loss_clf', 
                        'loss_orth', 'loss_rdm_hlv', 'loss_rdm_llv']
        loss_dict = {key: {'train':0, 'val':0} for key in loss_list}

        for split in ('train', 'val'):
            print(split)
            if split == 'train':
                model.train()
            else:
                model.eval()
            for i,(sub_id, fmri, fea, label) in enumerate(data_loader[split]):
                fmri = fmri.float().cuda()
                fea = fea.float().cuda()
                label = label.float().cuda()
                sub_id_int = [sub_lists.index(ele) for ele in sub_id]
                if split == 'val':
                    with torch.no_grad():
                        pred_logit, _, hlv_token, llv_token = model(fmri, sub_id_int)
                elif split == 'train':
                    pred_logit, _, hlv_token, llv_token = model(fmri, sub_id_int)
                loss_clf = criterion(pred_logit, label)
                # cls token orthogonal loss
                loss_orth = eval('calculate_orthogonal_regularization_%s'%args.orth_type)(
                            hlv_token, llv_token)
                # rsa loss
                # # high level
                rdm_hlv_fea = cal_rdm(fea[:, :hlv_fea_dim], fea[:, :hlv_fea_dim])
                rdm_hlv_fmri = cal_rdm(hlv_token, hlv_token)
                # print(rdm_hlv_fea.size(), rdm_hlv_fmri.size())
                loss_rdm_hlv = regularization_F(rdm_hlv_fea-rdm_hlv_fmri)
                # # low level
                rdm_llv_fea = cal_rdm(fea[:, hlv_fea_dim:], fea[:, hlv_fea_dim:])
                rdm_llv_fmri = cal_rdm(llv_token, llv_token)
                # print(rdm_llv_fea.size(), rdm_llv_fmri.size())
                loss_rdm_llv = regularization_F(rdm_llv_fea-rdm_llv_fmri)
                # evaluation
                pred_prob = torch.sigmoid(pred_logit)
                pred_label = pred_prob.ge(0.5)
                tot_loss = loss_clf * args.coe_clf + \
                            loss_orth * args.coe_orth + \
                            loss_rdm_hlv * args.coe_rdm_hlv + \
                            loss_rdm_llv * args.coe_rdm_llv

                for loss_name in loss_list:
                    loss_dict[loss_name][split] += eval(loss_name)

                if split == 'train':
                    model.zero_grad()
                    tot_loss.backward()
                    optimizer.step()

            for key in loss_list:
                summary.add_scalar(split+'/'+key, loss_dict[key][split]/(i+1), epoch+1)

    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    summary.close()
    score_final = test(args, model, data_loader['val'], met_list, 'val')
    sio.savemat(os.path.join(model_dir, 'score_final_val.mat'), score_final)
    return model, suffix

def test(args, model, data_loader, met_list, mode):
    # test and record
    sub_lists = open(os.path.join(args.list_IDs, 'subject_IDs_train.txt'), 'r').readlines()
    sub_lists = [ele.strip('\n') for ele in sub_lists]
    print(sub_lists)
    score_final = collections.defaultdict()
    labels = []
    pred_probs = []
    pred_labels = []
    with torch.no_grad():
        for i, (sub_id, fmri, fea, label) in enumerate(data_loader):
            fmri = fmri.float().cuda()
            fea = fea.float().cuda()
            label = label.float().cuda()
            sub_id_int = [sub_lists.index(ele) for ele in sub_id]
            pred_logit = model(fmri, sub_id_int)[0]
            # evaluation
            pred_prob = torch.sigmoid(pred_logit)
            pred_label = pred_prob.ge(0.5)
            labels.append(label.detach().cpu().numpy())
            pred_probs.append(pred_prob.detach().cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())
    labels, pred_probs, pred_labels = np.vstack(labels), np.vstack(pred_probs), np.vstack(pred_labels)
    for met in met_list:
        if met in ['average_precision_score']:
            score_final[met] = eval(met)(y_true=labels, y_score=pred_probs, average=None)
        elif met in ['roc_auc_score']:
            auc_tmp = np.ones_like((labels[0]))
            for i in range(auc_tmp.shape[0]):
                try:
                    auc_tmp[i] = eval(met)(y_true=labels[:, i], y_score=pred_probs[:, i], average='macro')
                except:
                    ValueError
            score_final[met] = auc_tmp
        elif met == 'hamming_loss':
            score_final[met] = eval(met)(y_true=labels, y_pred=pred_labels)
    return score_final

def test_wofea(args, model, data_loader, met_list):
    # test and record
    sub_lists = open(os.path.join(args.list_IDs, 'subject_IDs_train.txt'), 'r').readlines()
    sub_lists = [ele.strip('\n') for ele in sub_lists]
    print(sub_lists)
    score_final = collections.defaultdict()
    score_avg_final = collections.defaultdict(int)
    labels = []
    pred_probs = []
    pred_labels = []
    with torch.no_grad():
        for i, (sub_id, fmri, label) in enumerate(data_loader):
            fmri = fmri.float().cuda()
            label = label.float().cuda()
            sub_id_int = [sub_lists.index(ele) for ele in sub_id]
            pred_logit = model(fmri, sub_id_int)[0]
            # evaluation
            pred_prob = torch.sigmoid(pred_logit)
            pred_label = pred_prob.ge(0.5)
            labels.append(label.detach().cpu().numpy())
            pred_probs.append(pred_prob.detach().cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())
    labels, pred_probs, pred_labels = np.vstack(labels), np.vstack(pred_probs), np.vstack(pred_labels)
    for met in met_list:
        if met in ['average_precision_score']:
            score_final[met] = eval(met)(y_true=labels, y_score=pred_probs, average=None)
            score_avg_final[met] = round(score_final[met].mean(), 4)
        elif met in ['roc_auc_score']:
            auc_tmp = np.ones_like((labels[0]))
            for i in range(auc_tmp.shape[0]):
                try:
                    auc_tmp[i] = eval(met)(y_true=labels[:, i], y_score=pred_probs[:, i], average='macro')
                except:
                    ValueError
            score_final[met] = auc_tmp
            score_avg_final[met] = round(score_final[met].mean(), 4)
        elif met == 'hamming_loss':
            score_final[met] = eval(met)(y_true=labels, y_pred=pred_labels)
            score_avg_final[met] = round(score_final[met], 4)
    return score_final


def load_test_data(args, subject):
    padding = True
    dataset = Dataset_individual_test(args.data_root, movie_idx='4',
                            subject=subject,  volume = (111, 127, 111), 
                            delay = args.delay, padding=padding, 
                            split='test', sel_label=args.sel_label, sel_label_path=args.sel_label_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    class_num = dataset.class_num
    return data_loader, class_num


def main():

    parser = argparse.ArgumentParser(description='CLIP-MUSED model on HCP datasets.')
    parser.add_argument('-m','--mode', default = 'train', choices=['train', 'test'], type=str)
    parser.add_argument('-rp','--root_path', help='root path', 
                        default='/data/home/multi_subject/base_exp/', type=str)
    parser.add_argument('-rd','--result_dir', help='save model weights and tensorboard logs',
                        default = 'results', type=str)
    parser.add_argument('-dr', '--data_root', type=str, help='root path of HCP 7T movie data',
            default="/nfs/diskstation/DataStation/public_dataset/HCP_7T_Movie")
    parser.add_argument('-lIDs', '--list_IDs', type=str, help='root path of ID list files',
            default='/data/home/multi_subject/dataset/HCP')
    parser.add_argument('-cn', '--csv_name',help='name of csv file', default = 'summary', type=str)
    parser.add_argument('-delay', '--delay', default = 0, type = int, help = 'HR')
    parser.add_argument('-b', '--batch_size',help='batch size of dnn training', default = 50, type=int)
    parser.add_argument('-stepsz', '--step_size',help='step size', default = 25, type=int)
    parser.add_argument('-lr', '--learning_rate',help='learning rate', default = 3e-5, type=float)
    parser.add_argument('-gamma', '--gamma',help='gamma', default = 0.7, type=float)
    parser.add_argument('-e', '--epochs',help='epochs of dnn training', default = 1, type=int)
    parser.add_argument('-si', '--subject',help='Subject ID (Only for test mode)', default = '233326', type=str)
    parser.add_argument('-subn', '--subject_num',help='Subject number', default = 9, type=int)
    parser.add_argument('-suffix', '--suffix', help='only need in test mode', default=' ', type=str)
    parser.add_argument('-depth', '--depth', help='depth of vit backbone', default=4, type=int)
    parser.add_argument('-usecls', '--use_cls_token', help='use cls token or use the average of patch tokens', 
                        action='store_true')

    parser.add_argument('-fr', '--fea_root',help='root path of stimuli features', 
                        default = '/data/home/multi_subject/dataset/HCP/features/', type=str)
    parser.add_argument('-hfmodel', '--hlv_fea_model',help='model name of high-level feature extractor', default = '', type=str)
    parser.add_argument('-lfmodel', '--llv_fea_model',help='model name of low-level feature extractor', default = '', type=str)
    parser.add_argument('-hdim_rd', '--hlv_dim_rd',help='dimension reduction strategy of high-level feature', default = 'not_rd', type=str)
    parser.add_argument('-ldim_rd', '--llv_dim_rd',help='dimension reduction strategy of low-level feature', default = 'not_rd', type=str)
    parser.add_argument('-hlayer', '--hlv_layer',help='high-level feature layer', default = '-1', type=str)
    parser.add_argument('-llayer', '--llv_layer',help='low-level feature layer', default = '1', type=str)
    parser.add_argument('-met', '--metric', help='metric of RDM', default='cosine_similarity', type=str)
    parser.add_argument('-otype', '--orth_type', help='type of orthogonal regularization', 
                        default='F', choices=['L1', 'L2', 'F'], type=str)
    parser.add_argument('-coe_clf', '--coe_clf', help='coefficient of classification loss', default=1., type=float)
    parser.add_argument('-coe_orth', '--coe_orth', help='coefficient of orthogonal loss', default=1., type=float)
    parser.add_argument('-coe_rdm_hlv', '--coe_rdm_hlv', help='coefficient of hlv-rdm loss', default=1., type=float)
    parser.add_argument('-coe_rdm_llv', '--coe_rdm_llv', help='coefficient of llv-rdm loss', default=1., type=float)
    parser.add_argument('-sel_label', '--sel_label', help='only classify the high-frequency labels', 
                        action='store_true')
    parser.add_argument('-sel_label_path', '--sel_label_path', type=str, help='root path of sel_label_idx.npy',
            default='/data/home/multi_subject/dataset/HCP/sel_label_idx_0p1.npy')
    parser.add_argument('-fea_pp', '--fea_postproc', help='post-processing of image features (threshold and l2 norm)', 
                        action='store_true')
    parser.add_argument('-thres', '--threshold', help='threshold used in the post-processing', default=0.0, type=float)
    parser.add_argument('-me', '--model_epoch', help='epoch of the saved model', default=-1, type=int)
    parser.add_argument('-ti', '--test_interval', help='test interval', default=5, type=int)
    parser.add_argument('-abkind', '--ablation_kind', help='ablation_kind', default=' ', type=str)
    
    args = parser.parse_args()

    # hyper_suffix = ''
    if args.ablation_kind != ' ':
        root_path = os.path.join(args.root_path, args.result_dir, 
                            'HCP', 'multisub_%s'%args.ablation_kind)
    else:
        root_path = os.path.join(args.root_path, args.result_dir, 
                            'HCP', 'multisub')

    if args.mode == 'train':
        model, suffix = trainer(args, root_path)

    elif args.mode == 'test':
        # load model weights and test
        data_loader, class_num = load_test_data(args, args.subject)
        if args.suffix == ' ':
            files = os.listdir(os.path.join(root_path, 'models'))
            candidates = sorted(files, key=lambda x: x.split('_')[0], reverse=True)   # if not specfic, use the latest one.
            suffix = candidates[0]
        else:
            suffix = args.suffix
        model_dir = glob.glob(os.path.join(root_path, 'models', suffix))[0]
        print('Load pre-trained model from', model_dir)
        print('The number of category', class_num)

        # build model
        model, param = BrainFormer_MS(depth=args.depth, num_classes=class_num,
                use_cls_token=args.use_cls_token,
                num_subs=args.subject_num)

        if args.model_epoch == -1:
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model_%03d.pth'%args.model_epoch)))
        model.cuda()
        model.eval()

        met_list = ['average_precision_score', 'roc_auc_score', 'hamming_loss']
        score_final = test_wofea(args, model, data_loader, met_list=met_list)
        sio.savemat(os.path.join(model_dir, 
                'score_final_%s_%s.mat'%(args.mode, args.subject)), score_final)

if __name__ == "__main__":
    main()
