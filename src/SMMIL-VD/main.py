import argparse
import logging
import math
import os
import random
import shutil
import time
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
from libml.Echo_data import EchoDataset_traincombined, EchoDataset, CustomizedSampler

from libml.utils import save_pickle
from libml.utils import train_one_epoch, eval_model, eval_unlabeledset

from libml.utils import EarlyStopping


from libml.models.ema import ModelEMA
from libml.randaugment import RandAugmentMC


logger = logging.getLogger(__name__)


# Training settings
parser = argparse.ArgumentParser()

# parser.add_argument('--patience', default=200, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--resume', default='last_checkpoint.pth.tar', type=str,
                    help='name of the checkpoint (default: last_checkpoint.pth.tar)')
# parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
#                     help='fullpath of the checkpoint to resume from(default: none)') #automatically set
parser.add_argument('--early_stopping_warmup', default=0, type=int)


parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_classes', default=3, type=int)
parser.add_argument('--train_epoch', default=7200, type=int, help='total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--eval_every_Xepoch', default=360, type=int)


#data setting
parser.add_argument('--dataset_name', default='echo', type=str, choices=['echo'], help='dataset name')#没用
parser.add_argument('--resized_shape', default=112, type=int)
parser.add_argument('--data_seed', default=0, type=int, help='which predefined split of TMED2')
parser.add_argument('--training_seed', default=0, type=int, help='random seed for training procedure')
parser.add_argument('--development_size', default='DEV479', help='DEV479, DEV165, DEV56')
parser.add_argument('--use_data_normalization', default='False', type=str,
                    help='whether to normalize using train set mean and std')
parser.add_argument('--train_dir')
parser.add_argument('--data_dir')
parser.add_argument('--train_PatientStudy_list_path', type=str)
parser.add_argument('--val_PatientStudy_list_path', type=str)
parser.add_argument('--test_PatientStudy_list_path', type=str)
parser.add_argument('--unlabeled_PatientStudy_list_path', type=str)

parser.add_argument('--unlabeled_IrrelevantPatientStudy_list_path', type=str)
parser.add_argument('--unlabeled_NoDopplerPatientStudy_list_path', type=str)


#Method setting:
parser.add_argument('--augmentation', default='standard', type=str,
                    help='either standar or RandAug')
parser.add_argument('--use_class_weights', default='True', type=str,
                    help='if use_class_weights is True, set class weights to be tie to combo of development_size and data_seed') 



parser.add_argument('--Pretrained', default='Whole', type=str, help='Whole, FeatureExtractor1, NoPretrain')
parser.add_argument('--freeze_f1_weights', default='True', type=str)
parser.add_argument('--freeze_f1_runningstats', default='True', type=str)

parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'Adam', 'AdamW'], type=str) 
parser.add_argument('--lr_warmup_epochs', default=0, type=float, help='warmup epoch for learning rate schedule') #following MixMatch and FixMatch repo
parser.add_argument('--lr_schedule_type', default='CosineLR', choices=['CosineLR', 'FixedLR'], type=str) 
parser.add_argument('--lr_cycle_epochs', default=50, type=int, help='epoch') 


parser.add_argument('--view_checkpoint_path', default='', type=str,
                    help='checkpoint for the pretrained view model')

parser.add_argument('--ViewRegularization_warmup_schedule_type', default='NoWarmup', choices=['NoWarmup', 'Linear', 'Sigmoid', ], type=str) 

parser.add_argument('--ViewRegularization_warmup_pos', default=0.4, type=float, help='position at which view regularization loss warmup ends') #following MixMatch and FixMatch repo

parser.add_argument('--lambda_ViewRegularization', default=1.0, type=float, help='coefficient of ViewRegularizationLoss')

parser.add_argument('--T', default=0.5, type=float, help='distribution sharpening temperature')#attention relevance alignment

parser.add_argument('--lr', default=0.0005,type=float, help='learning rate (default: 0.0005)')
parser.add_argument('--smaller_backbone_lr', default='False',type=str)

parser.add_argument('--wd', default=10e-5, type=float, help='weight decay')
parser.add_argument('--percentile_increment', default=20, type=int)
parser.add_argument('--inference_model_unlabeledset', default='raw', type=str)
parser.add_argument('--update_class_weights', default='True', type=str)


#default hypers not to search for now
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')

parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')


def prRed(prt): print("\033[91m{}\033[0m" .format(prt))
def prGreen(prt): print("\033[92m{}\033[0m" .format(prt))
def prYellow(prt): print("\033[93m{}\033[0m" .format(prt))
def prLightPurple(prt): print("\033[94m{}\033[0m" .format(prt))
def prPurple(prt): print("\033[95m{}\033[0m" .format(prt))
def prCyan(prt): print("\033[96m{}\033[0m" .format(prt))
def prRedWhite(prt): print("\033[41m{}\033[0m" .format(prt))
def prWhiteBlack(prt): print("\033[7m{}\033[0m" .format(prt))
    

def recalculate_class_weights(survived_predicted_classes):
    #all splits have the same number of no/early/sig AS studies
    trainlabeled_noAS = 76
    trainlabeled_earlyAS = 103
    trainlabeled_sigAS = 181
    
    survived_unlabeled_noAS = np.sum(survived_predicted_classes==0)
    survived_unlabeled_earlyAS = np.sum(survived_predicted_classes==1)
    survived_unlabeled_sigAS = np.sum(survived_predicted_classes==2)
    assert survived_unlabeled_noAS + survived_unlabeled_earlyAS + survived_unlabeled_sigAS == len(survived_predicted_classes)
    
    N_combined_noAS = trainlabeled_noAS + survived_unlabeled_noAS
    N_combined_earlyAS = trainlabeled_earlyAS + survived_unlabeled_earlyAS
    N_combined_significantAS = trainlabeled_sigAS + survived_unlabeled_sigAS
    
    product_without_N_combined_noAS =  N_combined_earlyAS * N_combined_significantAS 
#     print(product_without_N_combined_noAS)
    
    product_without_N_combined_earlyAS = N_combined_noAS * N_combined_significantAS 
#     print(product_without_N_combined_earlyAS)
    
    product_without_N_combined_significantAS = N_combined_noAS * N_combined_earlyAS 
#     print(product_without_N_combined_significantAS)
        
    
    denominator = product_without_N_combined_noAS + product_without_N_combined_earlyAS + product_without_N_combined_significantAS
    
    
    W_combined_noAS = round(product_without_N_combined_noAS / denominator,3)
    
    W_combined_earlyAS = round(product_without_N_combined_earlyAS / denominator,3)
    
    W_combined_significantAS = round(product_without_N_combined_significantAS / denominator,3)
    
    
    return torch.Tensor([W_combined_noAS, W_combined_earlyAS, W_combined_significantAS])

    
def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
        
        
        
#checked
def save_checkpoint(state, checkpoint_dir, filename='last_checkpoint.pth.tar'):
    '''last_checkpoint.pth.tar or xxx_model_best.pth.tar'''
    
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
        
#checked
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
#learning rate schedule   
def get_cosine_schedule_with_warmup(optimizer,
                                    lr_warmup_epochs,
                                    lr_cycle_epochs, #total train epochs
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
#         no_progress = float(current_epoch - lr_warmup_epochs) / \
#             float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))

        #see if using restart
        ###############################################################
        if current_epoch%lr_cycle_epochs==0: 
            current_cycle_epoch=lr_cycle_epochs
        else:
            current_cycle_epoch = current_epoch%lr_cycle_epochs
        
        no_progress = float(current_cycle_epoch - lr_warmup_epochs) / \
            float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))
        #################################################################
        
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)     


def get_fixed_lr(optimizer,
                lr_warmup_epochs,
                lr_cycle_epochs, #total train iterations
                num_cycles=7./16.,
                last_epoch=-1):
    def _lr_lambda(current_epoch):
        
        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    


def create_view_model(args):
    
    import libml.models.view_classifier as view_models
    
    view_model = view_models.build_wideresnet(depth=28,
                                        widen_factor=2,
                                        dropout=0.0,
                                        num_classes=3)
    
    logger.info("Total params for View Model: {:.2f}M".format(
        sum(p.numel() for p in view_model.parameters())/1e6))
    
    
        
    view_checkpoint = torch.load(args.view_checkpoint_path)

    view_model.load_state_dict(view_checkpoint['ema_state_dict'])
    
    view_model.eval()
    
    return view_model


def create_model(args):
    
    from libml.models.model import MultimodalNetwork
    
    model = MultimodalNetwork(num_classes=args.num_classes)
                    
        
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))
    
    return model


def get_model_optimizer_scheduler(args):
    
    #create model
    model = create_model(args)
    model.to(args.device)
          
    
    #optimizer_type choice
    no_decay = ['bias', 'bn']
    
    if args.smaller_backbone_lr:
        lr_head = args.lr
        lr_backbone = 0.1 * lr_head
        
        backbone_params = [
            {'params': [p for n, p in model.encoder_2D.feature_extractor_part1.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': lr_backbone, 'weight_decay': args.wd},
            {'params': [p for n, p in model.encoder_2D.feature_extractor_part1.named_parameters() if any(nd in n for nd in no_decay)], 'lr': lr_backbone, 'weight_decay': 0.0}
        ]
        
        other_params = [
            {'params': [p for n, p in model.named_parameters() if "encoder_2D.feature_extractor_part1" not in n and not any(nd in n for nd in no_decay)], 'lr': lr_head, 'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters() if "encoder_2D.feature_extractor_part1" not in n and any(nd in n for nd in no_decay)], 'lr': lr_head, 'weight_decay': 0.0}
        ]

        grouped_parameters = backbone_params + other_params
    
    else:
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        
    if args.optimizer_type == 'SGD':
        
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        
    elif args.optimizer_type == 'Adam':
        optimizer = optim.Adam(grouped_parameters, lr=args.lr)
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    elif args.optimizer_type == 'AdamW':
        optimizer = optim.AdamW(grouped_parameters, lr=args.lr)
        
    else:
        raise NameError('Not supported optimizer setting')
          
    
    #lr_schedule_type choice
    if args.lr_schedule_type == 'CosineLR':
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    
    elif args.lr_schedule_type == 'FixedLR':
        scheduler = get_fixed_lr(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    
    else:
        raise NameError('Not supported lr scheduler setting')
    
    
    #instantiate the ema_model object
    ema_model = ModelEMA(args, model, args.ema_decay)
    
    
    return model, ema_model, optimizer, scheduler



def get_transforms(args):
    
    echo_mean = [0.059, 0.059, 0.059]
    echo_std = [0.138, 0.138, 0.138]
    
    if args.use_data_normalization:
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=echo_mean, std=echo_std)
        ])
        
        if args.augmentation == 'standard':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=args.resized_shape,
                                         padding=int(args.resized_shape*0.125),
                                         padding_mode='reflect'),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
            
        elif args.augmentation == 'RandAug':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=args.resized_shape,
                                         padding=int(args.resized_shape*0.125),
                                         padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
            
        else:
            raise NameError('Not implemented augmentation')
            
            

    else:
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
        #         transforms.Normalize(mean=echo_mean, std=echo_std)
        ])
        
        if args.augmentation == 'standard':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=args.resized_shape,
                                         padding=int(args.resized_shape*0.125),
                                         padding_mode='reflect'),
                    transforms.ToTensor(),
            #         transforms.Normalize(mean=echo_mean, std=echo_std)
                ])

        elif args.augmentation == 'RandAug':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=args.resized_shape,
                                         padding=int(args.resized_shape*0.125),
                                         padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
#                     transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
        else:
            raise NameError('Not implemented augmentation')
            
    
    return transform_labeledtrain, transform_eval



def main(args, brief_summary):
    
    iteration_dependent_patience = [70, 50, 50, 50, 30, 30]
    
    args.use_data_normalization = str2bool(args.use_data_normalization)
    args.smaller_backbone_lr = str2bool(args.smaller_backbone_lr)
    
#     TMED2SummaryTable = pd.read_csv(os.path.join(args.data_info_dir, 'TMED2SummaryTable.csv'))
    TMED2SummaryTable = pd.read_csv('../data_info/SimplifiedTMED2SummaryTable.csv')

    #get the transformation
    transform_labeledtrain, transform_eval = get_transforms(args)
    
    train_PatientStudy_list = pd.read_csv(args.train_PatientStudy_list_path)
    train_PatientStudy_list = train_PatientStudy_list['study'].values
    
    val_PatientStudy_list = pd.read_csv(args.val_PatientStudy_list_path)
    val_PatientStudy_list = val_PatientStudy_list['study'].values

    test_PatientStudy_list = pd.read_csv(args.test_PatientStudy_list_path)
    test_PatientStudy_list = test_PatientStudy_list['study'].values
    
    unlabeled_PatientStudy_list = pd.read_csv(args.unlabeled_PatientStudy_list_path)
    unlabeled_PatientStudy_list = unlabeled_PatientStudy_list['study'].values
    
    
    unlabeled_IrrelevantPatientStudy_list = pd.read_csv(args.unlabeled_IrrelevantPatientStudy_list_path)
    unlabeled_IrrelevantPatientStudy_list = unlabeled_IrrelevantPatientStudy_list['study'].values

    unlabeled_NoDopplerPatientStudy_list = pd.read_csv(args.unlabeled_NoDopplerPatientStudy_list_path)
    unlabeled_NoDopplerPatientStudy_list = unlabeled_NoDopplerPatientStudy_list['study'].values

    unlabeled_ToRemovePatientStudy_list = np.concatenate([unlabeled_IrrelevantPatientStudy_list, unlabeled_NoDopplerPatientStudy_list])
    unlabeled_ToRemovePatientStudy_list = np.array(list(set(unlabeled_ToRemovePatientStudy_list)))
    
    unlabeled_PatientStudy_list = unlabeled_PatientStudy_list[~np.isin(unlabeled_PatientStudy_list, unlabeled_ToRemovePatientStudy_list)]
    
    
    #Create dataset only once
    prGreen('===============Create Dataset===============')
    #traincombined_dataset (只有traincombined_dataset 用loaderV2)
    traincombined_dataset = EchoDataset_traincombined(train_PatientStudy_list, unlabeled_PatientStudy_list, TMED2SummaryTable, args.data_dir, training_seed=args.training_seed, transform_fn_strong=transform_labeledtrain, transform_fn_weak=transform_eval)
    
    #THIS IS JUST FOR SAINTY CHECK: to confirm the index of each part: trainlabeled set, unlabeled set
    trainlabeled_index, unlabeled_index, all_index = traincombined_dataset.get_fixed_ordered_index()
    print('Inside main.py, trainlabeled_index: {}, unlabeled_index: {}, all_index: {}'.format(trainlabeled_index, unlabeled_index, all_index))
    
#     trainmemory_dataset = EchoDataset(train_PatientStudy_list, TMED2SummaryTable, args.data_dir, training_seed=args.training_seed, transform_fn=transform_eval, NumFrames=args.NumFrames)

    val_dataset = EchoDataset(val_PatientStudy_list, TMED2SummaryTable, args.data_dir, training_seed=args.training_seed, transform_fn=transform_eval)
    
    test_dataset = EchoDataset(test_PatientStudy_list, TMED2SummaryTable, args.data_dir, training_seed=args.training_seed, transform_fn=transform_eval)
    
#     testSkip_dataset = EchoDatasetSkip(test_PatientStudy_list, TMED2SummaryTable, args.data_dir, training_seed=args.training_seed, transform_fn=transform_eval, NumFrames=args.NumFrames)
    
    prGreen('===============Create Samplers & Dataloaders===============')
    #get sampler for the traincombined set and the inference_unlabeled set
    traincombined_sampler = CustomizedSampler(trainlabeled_index, shuffle=True)#in the traincombined_dataset, indices 0-359 correspond to the trainlabeled set; will change from iteration to iteration, by including the indices of the unlabeled samples to use for this iteration; 1st iteration use only the trainlabeled set

    
    inference_unlabeled_sampler = CustomizedSampler(unlabeled_index) #in the traincombined_dataset, indices 360-5802 correspond to the unlabeled set; will not change for different iteration. 
    
    traincombined_loader = DataLoader(traincombined_dataset, batch_size=args.batch_size, sampler=traincombined_sampler, num_workers=args.num_workers, pin_memory=False)#will change from iteration to iteration, via the update of traincombined_sampler and the update of unlabeledset labels using update_unlabeledset_labels fct
    
    unlabeled_loader = DataLoader(traincombined_dataset, batch_size=args.batch_size, sampler=inference_unlabeled_sampler, num_workers=args.num_workers, pin_memory=False) #will not change for different iterations
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)#will not change for different iteration
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)#will not change for different iteration



    weights = args.class_weights
    weights = [float(i) for i in weights.split(',')]
    weights = torch.Tensor(weights)
#     print('weights used is {}'.format(weights))
    weights = weights.to(args.device)
    
    #load the view model, the output is unnormalized logits, need to use softmax on the output 
    view_model = create_view_model(args)
    view_model.to(args.device)
    
    #######################################################################################################################
    ############################################First Train only on Labeled Data###########################################
    
    prGreen('===============create model, ema_model, optimizer, scheduler===============')
    model, ema_model, optimizer, scheduler = get_model_optimizer_scheduler(args)   
    args.patience = iteration_dependent_patience[0]
    
    prPurple('====================FullySupervised Training====================')
    prPurple('use class weights: {}'.format(weights))
    prPurple("traincombined dataset: {} loader: {}; unlabeled loader: {}; val dataset: {} loader: {}; test dataset: {} loader: {}".format(len(traincombined_dataset), len(traincombined_loader), len(unlabeled_loader), len(val_dataset), len(val_loader), len(test_dataset), len(test_loader)))
    
    #set up folder for this iteration
    experiment_name = "Augmentation-{}_ViewRegularization_warmup-{}_Optimizer-{}_lr-{}_wd-{}_T-{}_lambdaViewRegularization-{}/FullySupervised_0".format(args.augmentation, args.ViewRegularization_warmup_schedule_type, args.optimizer_type, args.lr, args.wd, args.T, args.lambda_ViewRegularization)
    
    args.experiment_dir = os.path.join(args.train_dir, experiment_name)

    os.makedirs(args.experiment_dir, exist_ok=True)
    args.writer = SummaryWriter(args.experiment_dir)

    #train for this iteration:
    prPurple('start training for FullySupervised')
#         train_one_iteration(args, weights, iteration, traincombined_loader, trainmemory_loader, val_loader, test_loader, testSkip_loader, model, ema_model, view_model, optimizer, scheduler)
    train_one_iteration(args, weights, traincombined_loader, val_loader, test_loader, model, ema_model, view_model, optimizer, scheduler, 0, 0, 0)

    prPurple('done training for FullySupervised')

    #######################################################################################################################
    #######################################################################################################################

    iteration = 1
    while True:
        
        prPurple('====================CL iteration: {}, using percentile: {}===================='.format(iteration, 100 - args.percentile_increment * iteration))
        args.patience = iteration_dependent_patience[iteration]
        
        
        if args.inference_model_unlabeledset == 'ema':
            best_model_checkpoint_path = os.path.join(args.experiment_dir, 'val_progression_view/best_predictions_at_ema_val/best_model.pth.tar')
        elif args.inference_model_unlabeledset == 'raw':
            best_model_checkpoint_path = os.path.join(args.experiment_dir, 'val_progression_view/best_predictions_at_raw_val/best_model.pth.tar')
        else:
            raise NameError('Invalid')
            
        
        prPurple('evaluation of unlabeledset for iteration: {}'.format(iteration))
        survived_original_indexes, survived_predicted_classes = eval_unlabeledset(args, unlabeled_loader, best_model_checkpoint_path, model, iteration)
        
        n_survived_unlabeled_noAS = int(np.sum(survived_predicted_classes==0))
        n_survived_unlabeled_earlyAS = int(np.sum(survived_predicted_classes==1))
        n_survived_unlabeled_sigAS = int(np.sum(survived_predicted_classes==2))
    
        prPurple('n_survived_unlabeled_noAS: {}, n_survived_unlabeled_earlyAS: {}, n_survived_unlabeled_sigAS: {}'.format(n_survived_unlabeled_noAS, n_survived_unlabeled_earlyAS, n_survived_unlabeled_sigAS))
        
        del model, ema_model
        torch.cuda.empty_cache()
        
        
        if args.update_class_weights:
            weights = recalculate_class_weights(survived_predicted_classes)
            weights = weights.to(args.device)
    
        prPurple('use class weights: {}'.format(weights))
        
        #####################################reset model, optimizer, sampler#############################
        prPurple('===============reset model, ema_model, optimizer, scheduler===============')
        model, ema_model, optimizer, scheduler = get_model_optimizer_scheduler(args)        
        swa_model = AveragedModel(model)
        
        
        #reset traincombined sampler
        prPurple('===============reset traincombined_sampler===============')
        traincombined_sampler = CustomizedSampler(trainlabeled_index + list(survived_original_indexes), shuffle=True)
        
        #update traincombined_dataset
        prPurple('===============update traincombined_dataset with the survived indexes and predicted classes===============')
        traincombined_dataset.update_unlabeledset_labels(survived_original_indexes, survived_predicted_classes)
        
        #update traincombined_loader
        traincombined_loader = DataLoader(traincombined_dataset, batch_size=args.batch_size, sampler=traincombined_sampler, num_workers=args.num_workers, pin_memory=False)#will change from iteration to iteration, via the update of traincombined_sampler and the update of unlabeledset labels using update_unlabeledset_labels fct
        #########################################################################################################


        prPurple("traincombined dataset: {} loader: {}; unlabeled loader: {}; val dataset: {} loader: {}; test dataset: {} loader: {}".format(len(traincombined_dataset), len(traincombined_loader), len(unlabeled_loader), len(val_dataset), len(val_loader), len(test_dataset), len(test_loader)))
        
        #set up folder for this iteration
        experiment_name = "Augmentation-{}_ViewRegularization_warmup-{}_Optimizer-{}_lr-{}_wd-{}_T-{}_lambdaViewRegularization-{}/iteration-{}".format(args.augmentation, args.ViewRegularization_warmup_schedule_type, args.optimizer_type, args.lr, args.wd, args.T, args.lambda_ViewRegularization, iteration)
        
        args.experiment_dir = os.path.join(args.train_dir, experiment_name)

        os.makedirs(args.experiment_dir, exist_ok=True)
        args.writer = SummaryWriter(args.experiment_dir)
    
        #train for this iteration:
        prPurple('start training for iteration: {}'.format(iteration))
#         train_one_iteration(args, weights, iteration, traincombined_loader, trainmemory_loader, val_loader, test_loader, testSkip_loader, model, ema_model, view_model, optimizer, scheduler)
        train_one_iteration(args, weights, traincombined_loader, val_loader, test_loader, model, ema_model, view_model, optimizer, scheduler, n_survived_unlabeled_noAS, n_survived_unlabeled_earlyAS, n_survived_unlabeled_sigAS)

        prPurple('done training for iteration: {}'.format(iteration))
        
        if args.percentile_increment * iteration >= 100:
            print('Process finished')
            break
        
        iteration +=1 
        

# def train_one_iteration(args, weights, iteration, traincombined_loader, trainmemory_loader, val_loader, test_loader, testSkip_loader, model, ema_model, view_model, optimizer, scheduler):
def train_one_iteration(args, weights, traincombined_loader, val_loader, test_loader, model, ema_model, view_model, optimizer, scheduler, n_survived_unlabeled_noAS, n_survived_unlabeled_earlyAS, n_survived_unlabeled_sigAS):
    
    args.start_epoch = 0
    
    
    #val progression view: tracking as best val performance progress, the corresponding test performance.
    #regular val
    best_val_ema_Bacc = 0
    best_test_ema_Bacc_at_val = 0
    
    best_val_raw_Bacc = 0
    best_test_raw_Bacc_at_val = 0
    
    
    current_count=0 #for early stopping, when continue training
    #if continued from a checkpoint, overwrite the best_val_ema_Bacc, best_test_ema_Bacc_at_val, 
    #                                              best_val_raw_Bacc, best_test_raw_Bacc_at_val,
    #                                              start_epoch,
    #                                              model weights, ema model weights
    #                                              optimizer state dict
    #                                              scheduler state dict 
    
#     train_loss_dict = dict()
#     train_loss_dict['Totalloss'] = []
#     train_loss_dict['LabeledCEloss'] = []
#     train_loss_dict['ViewRegularizationLoss'] = []
        
    early_stopping = EarlyStopping(patience=args.patience, initial_count=current_count)
    early_stopping_warmup = args.early_stopping_warmup

    for epoch in tqdm(range(args.start_epoch, args.train_epoch)):
        val_predictions_save_dict = dict()
        test_predictions_save_dict = dict()
#         train_predictions_save_dict = dict()

        print('!!!!!!!!!Train!!!!!!!!!')
        TotalLoss_list, LabeledCEloss_list, ViewRegularizationLoss_list, scaled_ViewRegularizationLoss_list = train_one_epoch(args, weights, traincombined_loader, model, ema_model, view_model, optimizer, scheduler, epoch)
#         train_loss_dict['Totalloss'].extend(TotalLoss_list)
#         train_loss_dict['LabeledCEloss'].extend(LabeledCEloss_list)
#         train_loss_dict['ViewRegularizationLoss'].extend(ViewRegularizationLoss_list)
#         save_pickle(os.path.join(args.experiment_dir, 'losses'), 'train_losses_dict.pkl', train_loss_dict)


        if epoch % args.eval_every_Xepoch == 0:
            print('!!!!!!!!!Val!!!!!!!!!')
#             val_raw_Bacc, val_raw_class1rec, val_raw_class2rec, val_raw_class3rec, val_ema_Bacc, val_ema_class1rec, val_ema_class2rec, val_ema_class3rec, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, val_loader, model, ema_model.ema, epoch)
#             val_predictions_save_dict['raw_Bacc'] = val_raw_Bacc
#             val_predictions_save_dict['ema_Bacc'] = val_ema_Bacc
#             val_predictions_save_dict['true_labels'] = val_true_labels
#             val_predictions_save_dict['raw_predictions'] = val_raw_predictions
#             val_predictions_save_dict['ema_predictions'] = val_ema_predictions
#             val_predictions_save_dict['this_epoch'] = epoch
            val_MRN, val_raw_Bacc, val_raw_class1rec, val_raw_class2rec, val_raw_class3rec, val_ema_Bacc, val_ema_class1rec, val_ema_class2rec, val_ema_class3rec, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, val_loader, model, ema_model.ema, epoch)
            val_predictions_save_dict['val_MRN'] = val_MRN
            val_predictions_save_dict['raw_Bacc'] = val_raw_Bacc
            val_predictions_save_dict['ema_Bacc'] = val_ema_Bacc
            val_predictions_save_dict['true_labels'] = val_true_labels
            val_predictions_save_dict['raw_predictions'] = val_raw_predictions
            val_predictions_save_dict['ema_predictions'] = val_ema_predictions
            val_predictions_save_dict['this_epoch'] = epoch
            
            
            save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'val_epoch_{}_predictions.pkl'.format(str(epoch)), val_predictions_save_dict)
            
            print('!!!!!!!!!Test!!!!!!!!!')
#             test_raw_Bacc, test_raw_class1rec, test_raw_class2rec, test_raw_class3rec, test_ema_Bacc, test_ema_class1rec, test_ema_class2rec, test_ema_class3rec, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, test_loader, model, ema_model.ema, epoch)
        
#             test_predictions_save_dict['raw_Bacc'] = test_raw_Bacc
#             test_predictions_save_dict['ema_Bacc'] = test_ema_Bacc
#             test_predictions_save_dict['true_labels'] = test_true_labels
#             test_predictions_save_dict['raw_predictions'] = test_raw_predictions
#             test_predictions_save_dict['ema_predictions'] = test_ema_predictions
#             test_predictions_save_dict['this_epoch'] = epoch
            test_MRN, test_raw_Bacc, test_raw_class1rec, test_raw_class2rec, test_raw_class3rec, test_ema_Bacc, test_ema_class1rec, test_ema_class2rec, test_ema_class3rec, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, test_loader, model, ema_model.ema, epoch)
            test_predictions_save_dict['test_MRN'] = test_MRN
            test_predictions_save_dict['raw_Bacc'] = test_raw_Bacc
            test_predictions_save_dict['ema_Bacc'] = test_ema_Bacc
            test_predictions_save_dict['true_labels'] = test_true_labels
            test_predictions_save_dict['raw_predictions'] = test_raw_predictions
            test_predictions_save_dict['ema_predictions'] = test_ema_predictions
            test_predictions_save_dict['this_epoch'] = epoch
            
            save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'test_epoch_{}_predictions.pkl'.format(str(epoch)), test_predictions_save_dict)





            #val progression view
            #regular Val
            if val_raw_Bacc > best_val_raw_Bacc:

                best_val_raw_Bacc = val_raw_Bacc
                best_test_raw_Bacc_at_val = test_raw_Bacc
#                 best_train_raw_Bacc_at_val = train_raw_Bacc

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_raw_val'), 'val_predictions.pkl', val_predictions_save_dict)

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_raw_val'), 'test_predictions.pkl', test_predictions_save_dict)
                
                
                
                save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'current_count':current_count,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    
                'val_progression_view': 
                    {'epoch': epoch+1,
                    #regular val
                    'best_val_ema_Bacc': best_val_ema_Bacc,
                    'best_val_raw_Bacc': best_val_raw_Bacc,
                    'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
                    'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
#                     'best_train_ema_Bacc_at_val': best_train_ema_Bacc_at_val,
#                     'best_train_raw_Bacc_at_val': best_train_raw_Bacc_at_val,
                     
                     }, 
               
                }, args.experiment_dir, filename='val_progression_view/best_predictions_at_raw_val/best_model.pth.tar')
                
                

            if val_ema_Bacc > best_val_ema_Bacc:
                
                best_val_ema_Bacc = val_ema_Bacc
                best_test_ema_Bacc_at_val = test_ema_Bacc
#                 best_train_ema_Bacc_at_val = train_ema_Bacc

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_ema_val'), 'val_predictions.pkl', val_predictions_save_dict)

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_ema_val'), 'test_predictions.pkl', test_predictions_save_dict)
                                                
#                 save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_ema_val'), 'train_predictions.pkl', train_predictions_save_dict)
                
                save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'current_count':current_count,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    
                'val_progression_view': 
                    {'epoch': epoch+1,
                    #regular val
                    'best_val_ema_Bacc': best_val_ema_Bacc,
                    'best_val_raw_Bacc': best_val_raw_Bacc,
                    'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
                    'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
#                     'best_train_ema_Bacc_at_val': best_train_ema_Bacc_at_val,
#                     'best_train_raw_Bacc_at_val': best_train_raw_Bacc_at_val,                     
                     }, 
               
                }, args.experiment_dir, filename='val_progression_view/best_predictions_at_ema_val/best_model.pth.tar')
                
                
            
            #val progression view
            logger.info('val progression view:')
            #regular val
            logger.info('At RAW Best val, validation/test/ %.2f %.2f' % (best_val_raw_Bacc, best_test_raw_Bacc_at_val))
            logger.info('At EMA Best val, validation/test/ %.2f %.2f' % (best_val_ema_Bacc, best_test_ema_Bacc_at_val))
            
            
            
            #only record the train loss, val_raw_Bacc, val_ema_Bacc, test_raw_Bacc, test_ema_Bacc every eval_every_Xepoch.
#             args.writer.add_scalar('train/1.train_raw_Bacc', train_raw_Bacc, epoch)
#             args.writer.add_scalar('train/1.train_ema_Bacc', train_ema_Bacc, epoch)
            args.writer.add_scalar('3.train/1.Totalloss', np.mean(TotalLoss_list), epoch)
            args.writer.add_scalar('3.train/2.LabeledCEloss', np.mean(LabeledCEloss_list), epoch)
            args.writer.add_scalar('3.train/3.ViewRegularizationLoss', np.mean(ViewRegularizationLoss_list), epoch)
            args.writer.add_scalar('3.train/4.scaled_ViewRegularizationLoss', np.mean(scaled_ViewRegularizationLoss_list), epoch)

            #regular val
            args.writer.add_scalar('2.val/1.val_raw_Bacc', val_raw_Bacc, epoch)
            args.writer.add_scalar('2.val/2.val_raw_class1rec', val_raw_class1rec, epoch)
            args.writer.add_scalar('2.val/3.val_raw_class2rec', val_raw_class2rec, epoch)
            args.writer.add_scalar('2.val/4.val_raw_class3rec', val_raw_class3rec, epoch)
            
            
            args.writer.add_scalar('2.val/5.val_ema_Bacc', val_ema_Bacc, epoch)
            args.writer.add_scalar('2.val/6.val_ema_class1rec', val_ema_class1rec, epoch)
            args.writer.add_scalar('2.val/7.val_ema_class2rec', val_ema_class2rec, epoch)
            args.writer.add_scalar('2.val/8.val_ema_class3rec', val_ema_class3rec, epoch)
    
           
            args.writer.add_scalar('1.test/1.test_raw_Bacc', test_raw_Bacc, epoch)
            args.writer.add_scalar('1.test/2.test_raw_class1rec', test_raw_class1rec, epoch)
            args.writer.add_scalar('1.test/3.test_raw_class2rec', test_raw_class2rec, epoch)
            args.writer.add_scalar('1.test/4.test_raw_class3rec', test_raw_class3rec, epoch)
            
            args.writer.add_scalar('1.test/5.test_ema_Bacc', test_ema_Bacc, epoch)
            args.writer.add_scalar('1.test/6.test_ema_class1rec', test_ema_class1rec, epoch)
            args.writer.add_scalar('1.test/7.test_ema_class2rec', test_ema_class2rec, epoch)
            args.writer.add_scalar('1.test/8.test_ema_class3rec', test_ema_class3rec, epoch)
            
            #val progression view
            #regular val
            brief_summary['val_progression_view']['best_val_ema_Bacc'] = best_val_ema_Bacc
            brief_summary['val_progression_view']['best_val_raw_Bacc'] = best_val_raw_Bacc
            brief_summary['val_progression_view']['best_test_ema_Bacc_at_val'] = best_test_ema_Bacc_at_val 
            brief_summary['val_progression_view']['best_test_raw_Bacc_at_val'] = best_test_raw_Bacc_at_val
#             brief_summary['val_progression_view']['best_train_ema_Bacc_at_val'] = best_train_ema_Bacc_at_val 
#             brief_summary['val_progression_view']['best_train_raw_Bacc_at_val'] = best_train_raw_Bacc_at_val
            
            
            
            with open(os.path.join(args.experiment_dir, "brief_summary.json"), "w") as f:
                json.dump(brief_summary, f)
                
            
            #early stopping counting:
            if epoch > early_stopping_warmup:
                current_count = early_stopping(val_ema_Bacc)
            
            save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'current_count':current_count,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    
                'val_progression_view': 
                    {'epoch': epoch+1,
                    #regular val
                    'best_val_ema_Bacc': best_val_ema_Bacc,
                    'best_val_raw_Bacc': best_val_raw_Bacc,
                    'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
                    'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
#                     'best_train_ema_Bacc_at_val': best_train_ema_Bacc_at_val,
#                     'best_train_raw_Bacc_at_val': best_train_raw_Bacc_at_val,                     
                     }, 
                    
               
                }, args.experiment_dir, filename='last_checkpoint.pth.tar')
        
            
            if early_stopping.early_stop:
                break

            
    #At the end of each iteration, evaluate using the swa_model            
    logger.info('FINAL raw_val:{}, ema_val:{}'.format(best_val_raw_Bacc, best_val_ema_Bacc))
    logger.info('FINAL raw_test:{}, ema_test:{}'.format(best_test_raw_Bacc_at_val, best_test_ema_Bacc_at_val))
    
    #val progression view
    #regular val
    
    brief_summary['val_progression_view']['best_val_ema_Bacc'] = best_val_ema_Bacc
    brief_summary['val_progression_view']['best_val_raw_Bacc'] = best_val_raw_Bacc
    brief_summary['val_progression_view']['best_test_ema_Bacc_at_val'] = best_test_ema_Bacc_at_val 
    brief_summary['val_progression_view']['best_test_raw_Bacc_at_val'] = best_test_raw_Bacc_at_val

    prPurple('n_survived_unlabeled_noAS: {}, n_survived_unlabeled_earlyAS: {}, n_survived_unlabeled_sigAS: {}'.format(n_survived_unlabeled_noAS, n_survived_unlabeled_earlyAS, n_survived_unlabeled_sigAS))
    
    brief_summary['val_progression_view']['added_unlabeled_noAS'] = n_survived_unlabeled_noAS
    brief_summary['val_progression_view']['added_unlabeled_earlyAS'] = n_survived_unlabeled_earlyAS
    brief_summary['val_progression_view']['added_unlabeled_sigAS'] = n_survived_unlabeled_sigAS
    
#     brief_summary['val_progression_view']['best_train_ema_Bacc_at_val'] = best_train_ema_Bacc_at_val 
#     brief_summary['val_progression_view']['best_train_raw_Bacc_at_val'] = best_train_raw_Bacc_at_val

    args.writer.close()

    with open(os.path.join(args.experiment_dir, "brief_summary.json"), "w") as f:
        json.dump(brief_summary, f)
        
        
    
        
    
    
    
            
if __name__=='__main__':
    
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError('Not Using GPU?')
        
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

  
    logger.info(dict(args._get_kwargs()))

    if args.training_seed is not None:
        print('setting training seed{}'.format(args.training_seed), flush=True)
        set_seed(args.training_seed)
    
        
    
    if args.Pretrained == 'Whole':
        
        raise NameError('NOT USED')
        
    elif args.Pretrained == 'FeatureExtractor1':
        raise NameError('NOT USED')
       
    elif args.Pretrained == 'NoPretrain':
        print('Not using hz manual pretraining')
    
    else:
        raise NameError('invalid pretrain option')
    
    
    
    ################################################Determining class weights################################################
    
    if args.use_class_weights == 'True':
        print('!!!!!!!!Using pre-calculated class weights!!!!!!!!')
        args.class_weights = '0.463,0.342,0.195'
        #indeed, every split should have the same class weight for diagnosis by our dataset construction

    else:
        args.class_weights = '1.0,1.0,1.0'
        print('?????????Not using pre-calculated class weights?????????')
        
        
        
    
    #brief summary:
    brief_summary = {}
    brief_summary['val_progression_view'] = {}
    
    brief_summary['dataset_name'] = args.dataset_name
    brief_summary['algorithm'] = 'Echo_MIL'
    brief_summary['hyperparameters'] = {
        'train_epoch': args.train_epoch,
        'optimizer': args.optimizer_type,
        'lr': args.lr,
        'wd': args.wd,
        'T':args.T,
        'lambda_ViewRegularization':args.lambda_ViewRegularization
    }

    main(args, brief_summary)

    
