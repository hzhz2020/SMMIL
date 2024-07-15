import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
from sklearn.metrics import confusion_matrix as sklearn_cm
import numpy as np
import os
import pickle

import torch
import torch.nn as nn



class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""
    
    def __init__(self, patience=300, initial_count=0, delta=0):
        
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            
        """
        
        self.patience = patience
        self.counter = initial_count
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        
    
    def __call__(self, val_acc):
        
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.counter = 0
            
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!counter: {}, score: {}, best_score: {}'.format(self.counter, score, self.best_score))
        
        return self.counter

            
def train_one_epoch(args, weights, train_loader, model, ema_model, view_model, optimizer, scheduler, epoch):
    
    args.writer.add_scalar('3.train/5.lr', scheduler.get_last_lr()[0], epoch)

    model.train()
    
    TotalLoss_this_epoch, LabeledCELoss_this_epoch, ViewRegularizationLoss_this_epoch, scaled_ViewRegularizationLoss_this_epoch = [], [], [], []
    
    #ViewRegularization warmup schedule choice
    if args.ViewRegularization_warmup_schedule_type == 'NoWarmup':
        current_warmup = 1
    elif args.ViewRegularization_warmup_schedule_type == 'Linear':
        current_warmup = np.clip(epoch/(float(args.ViewRegularization_warmup_pos) * args.train_epoch), 0, 1)
    elif args.ViewRegularization_warmup_schedule_type == 'Sigmoid':
        current_warmup = math.exp(-5 * (1 - min(epoch/(float(args.ViewRegularization_warmup_pos) * args.train_epoch), 1))**2)
    else:
        raise NameError('Not supported ViewRegularization warmup schedule')

    
    for batch_idx, (_, data_2D, data_Doppler, bag_label) in enumerate(tqdm(train_loader)):
        

        data_2D, data_Doppler, bag_label = data_2D.to(args.device), data_Doppler.to(args.device), bag_label.to(args.device)
                
        
        outputs, attentions = model(data_2D, data_Doppler)
        
        log_attentions = torch.log(attentions)

        
        with torch.no_grad():

            ############to reduce memory usage, the view model only process the first 4 frames of each video############
            view_predictions = view_model(data_2D.squeeze(0)[:, :4, :, :].reshape(-1, 3, 112, 112)) #first 4 frames
#             view_predictions = view_model(data.squeeze(0).view(-1, 3, 112, 112)) #all 8 frames
#             print('view_predictions shape : {}'.format(view_predictions.shape))

            ##view_predictions is now (#video * #frames: 16, output_dim)
            view_predictions = view_predictions.view(-1, 4, 3) #first 4 frames
#             view_predictions = view_predictions.view(-1, args.NumFrames, 3) #all 8 frames

#             print('view_predictions shape : {}'.format(view_predictions.shape))

            ############to reduce memory usage, the view model only process the first 4 frames of each video############        


            
            #apply softmax ove the logit of each frame
#             softmax_view_predictions = F.softmax(view_predictions, dim=2) #(#videos, 16, 3)
            view_predictions = F.softmax(view_predictions, dim=2) #(#videos, 16, 3)

#             print('softmax_view_predictions shape : {}'.format(softmax_view_predictions.shape))

            
            #average over the 16 frames of each video
#             softmax_view_predictions = softmax_view_predictions.mean(dim=1)#(#videos, 3)
            view_predictions = view_predictions.mean(dim=1)#(#videos, 3)

#             print('softmax_view_predictions shape : {}'.format(softmax_view_predictions.shape))

            
            predicted_relative_relevance = view_predictions[:, :2]
#             print('predicted_relevance shape : {}'.format(predicted_relevance.shape))

#             predicted_relevance = torch.sum(predicted_relevance, dim=1)
            predicted_relative_relevance = torch.sum(predicted_relative_relevance, dim=1)

#             print('predicted_relevance shape : {}'.format(predicted_relevance.shape))

            predicted_relative_relevance = F.softmax(predicted_relative_relevance/args.T)
            predicted_relative_relevance = predicted_relative_relevance.unsqueeze(0) 
            
        
        #element shape in F.cross_entropy: prediction torch.size([batch_size, num_classes]) and true label torch.size([batch_size])
        if args.use_class_weights == 'True':
            LabeledCELoss = F.cross_entropy(outputs, bag_label, weights, reduction='mean')
        else:
            LabeledCELoss = F.cross_entropy(outputs, bag_label, reduction='mean')
            
        
        
        
        ViewRegularizationLoss = F.kl_div(input=log_attentions, target=predicted_relative_relevance, log_target=False, reduction='batchmean')
        
        # backward pass
        total_loss = LabeledCELoss + args.lambda_ViewRegularization * ViewRegularizationLoss * current_warmup
        
        total_loss.backward()
        
        
        del view_predictions, predicted_relative_relevance
        del outputs, attentions
        del data_2D, data_Doppler
        torch.cuda.empty_cache()
        
        TotalLoss_this_epoch.append(total_loss.item())
        LabeledCELoss_this_epoch.append(LabeledCELoss.item())
        ViewRegularizationLoss_this_epoch.append(ViewRegularizationLoss.item())
        scaled_ViewRegularizationLoss_this_epoch.append(args.lambda_ViewRegularization * ViewRegularizationLoss.item() * current_warmup)

        # step
        optimizer.step()

        #update ema model
        ema_model.update(model)
        
        model.zero_grad()
    
    scheduler.step()
    
   
    
    return TotalLoss_this_epoch, LabeledCELoss_this_epoch, ViewRegularizationLoss_this_epoch, scaled_ViewRegularizationLoss_this_epoch

   

def eval_unlabeledset(args, unlabeled_loader, best_model_checkpoint_path, model_to_use, iteration):
    
    #saintycheck that the model is loaded
#     print('before loading saved model, model_to_use.classifier[0].bias: {}'.format(model_to_use.classifier[0].bias)) #in model definition, the linear layer name is 'classifier'
    
    print('loading checkpoint: {}'.format(best_model_checkpoint_path))
    inference_checkpoint = torch.load(best_model_checkpoint_path)
    
    if args.inference_model_unlabeledset == 'ema':
        model_to_use.load_state_dict(inference_checkpoint['ema_state_dict'])
    elif args.inference_model_unlabeledset == 'raw':
        model_to_use.load_state_dict(inference_checkpoint['state_dict'])
    else:
        raise NameError('Invalid')
                
    print('After loading saved model, model_to_use.classifier[0].bias: {}'.format(model_to_use.classifier[0].bias))
    percentile_to_use = 100 - args.percentile_increment * iteration
    
    model_to_use.eval()
    
    max_probabilities = []
    predicted_classes = []
    original_indexes = []
    
    with torch.no_grad():
        for batch_idx, (index, data_2D, data_Doppler, bag_label) in enumerate(tqdm(unlabeled_loader)):
#             print('Inside eval_unlabeled, batch_idx: {} sample index: {}, shape: {}'.format(batch_idx, index, index.shape))
            
            data_2D, data_Doppler, bag_label = data_2D.to(args.device), data_Doppler.to(args.device), bag_label.to(args.device)
            
            output, _ = model_to_use(data_2D, data_Doppler)
            output = F.softmax(output, dim=1)
            
            max_values, max_indices = torch.max(output, dim=1) 
#             print('output probabilities: {}, max_values: {}, max_indices: {}'.format(output, max_values, max_indices))
            
            max_probabilities.append(max_values.detach().cpu())
            predicted_classes.append(max_indices.detach().cpu())
            original_indexes.append(index)
            
        max_probabilities = np.concatenate(max_probabilities, axis=0)
        predicted_classes = np.concatenate(predicted_classes, axis=0)
        original_indexes = np.concatenate(original_indexes, axis=0)
        print('max_probabilities: {}, shape: {}'.format(max_probabilities, max_probabilities.shape))
        print('predicted_classes: {}, shape: {}'.format(predicted_classes, predicted_classes.shape))
        print('original_indexes: {}, shape: {}'.format(original_indexes, original_indexes.shape))


        threshold = np.percentile(max_probabilities, percentile_to_use)
        print('threshold at iteration {} is {}'.format(iteration, threshold))

        survived_mask = max_probabilities>=threshold

        survived_original_indexes = original_indexes[survived_mask]
        survived_predicted_classes = predicted_classes[survived_mask]
        print('total survived for iteration {} is {}, survived_original_indexes: {} survived_predicted_classes: {}'.format(iteration, len(survived_original_indexes), survived_original_indexes, survived_predicted_classes))
    
    
    del data_2D, data_Doppler, output
    torch.cuda.empty_cache()
    
    
    return survived_original_indexes, survived_predicted_classes



    
#regular eval_model
def eval_model(args, data_loader, raw_model, ema_model, epoch):
        
    raw_model.eval()
    ema_model.eval()

    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        total_ema_outputs = []
        total_MRNs = []
        
#         for batch_idx, (data_2D, data_Doppler, bag_label) in enumerate(data_loader):
        for batch_idx, (data_MRN, data_2D, data_Doppler, bag_label) in enumerate(data_loader):
            
#             print('EVAL type(data): {}, data.size: {}, require grad: {}'.format(type(data), data.size(), data.requires_grad))
#             print('EVAL type(bag_label): {}, bag_label: {}'.format(type(bag_label), bag_label))

            data_2D, data_Doppler, bag_label = data_2D.to(args.device), data_Doppler.to(args.device), bag_label.to(args.device)
            
            raw_outputs, _ = raw_model(data_2D, data_Doppler)
            ema_outputs, _ = ema_model(data_2D, data_Doppler)
#             print('target is {}, raw_outputs is: {}, ema_outputs is {}'.format(bag_label, raw_outputs, ema_outputs))

            total_targets.append(bag_label.detach().cpu())        
            total_raw_outputs.append(raw_outputs.detach().cpu())
            total_ema_outputs.append(ema_outputs.detach().cpu())
            total_MRNs.append(data_MRN)
    
        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        total_ema_outputs = np.concatenate(total_ema_outputs, axis=0)
        total_MRNs = np.concatenate(total_MRNs, axis=0)
        
#         print('RegularEval total_targets: {}'.format(total_targets))
#         print('RegularEval total_raw_outputs: {}'.format(total_raw_outputs))
#         print('RegularEval total_ema_outputs: {}'.format(total_ema_outputs))

        raw_Bacc, raw_class1rec, raw_class2rec, raw_class3rec = calculate_balanced_accuracy(total_raw_outputs, total_targets)
        ema_Bacc, ema_class1rec, ema_class2rec, ema_class3rec = calculate_balanced_accuracy(total_ema_outputs, total_targets)

#         print('raw Bacc this evaluation step: {}'.format(raw_Bacc), flush=True)
#         print('ema Bacc this evaluation step: {}'.format(ema_Bacc), flush=True)
    del data_2D, data_Doppler, raw_outputs, ema_outputs
    torch.cuda.empty_cache()


    return total_MRNs, raw_Bacc, raw_class1rec, raw_class2rec, raw_class3rec, ema_Bacc, ema_class1rec, ema_class2rec, ema_class3rec, total_targets, total_raw_outputs, total_ema_outputs

#     return raw_Bacc, raw_class1rec, raw_class2rec, raw_class3rec, ema_Bacc, ema_class1rec, ema_class2rec, ema_class3rec, total_targets, total_raw_outputs, total_ema_outputs



def calculate_balanced_accuracy(prediction, true_target, return_type = 'all'):
    
    confusion_matrix = sklearn_cm(true_target, prediction.argmax(1))
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)

    assert n_class==3
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    

    if return_type == 'all':
#         return balanced_accuracy * 100, class0_recall * 100, class1_recall * 100, class2_recall * 100
        return balanced_accuracy * 100, recalls[0], recalls[1], recalls[2]

    elif return_type == 'only balanced_accuracy':
        return balanced_accuracy * 100
    else:
        raise NameError('Unsupported return_type in this calculate_balanced_accuracy fn')

        
 #shared helper fct across different algos
def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
               