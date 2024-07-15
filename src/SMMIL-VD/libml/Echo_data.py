import numpy as np
import pandas as pd

import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torch

import pickle

def prRed(prt): print("\033[91m{}\033[0m" .format(prt))
def prGreen(prt): print("\033[92m{}\033[0m" .format(prt))
def prYellow(prt): print("\033[93m{}\033[0m" .format(prt))
def prLightPurple(prt): print("\033[94m{}\033[0m" .format(prt))
def prPurple(prt): print("\033[95m{}\033[0m" .format(prt))
def prCyan(prt): print("\033[96m{}\033[0m" .format(prt))
def prRedWhite(prt): print("\033[41m{}\033[0m" .format(prt))
def prWhiteBlack(prt): print("\033[7m{}\033[0m" .format(prt))


DiagnosisStr_to_Int_Mapping={
    'Not_Provided':-1,
    'no_AS':0,
    'mild_AS':1,
    'mildtomod_AS':1,
    'moderate_AS':2,
    'severe_AS':2
    
}


def load_pickle(path):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data

#https://github.com/perrying/realistic-ssl-evaluation-pytorch/blob/master/train.py
#https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#Sampler
    
class CustomizedSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, indices, shuffle=False):
        
        if shuffle:
            random.shuffle(indices)
            
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
    

class EchoDataset_traincombined(Dataset):
    
    def __init__(self, trainlabeled_PatientStudy_list, unlabeled_PatientStudy_list, TMED2SummaryTable, ML_DATA_dir, training_seed=0, transform_fn_strong=None, transform_fn_weak=None):
        
        
        
        self.trainlabeled_PatientStudy_list = trainlabeled_PatientStudy_list
#         self.unlabeled_PatientStudy_list = unlabeled_PatientStudy_list[:50] #for quick prototyping
        self.unlabeled_PatientStudy_list = unlabeled_PatientStudy_list
        
        self.TMED2SummaryTable = TMED2SummaryTable 
        
        self.ML_DATA_dir = ML_DATA_dir + '_dataloaderV2'

        self.training_seed=training_seed
        
        self.transform_fn_strong = transform_fn_strong
        self.transform_fn_weak = transform_fn_weak
        
        
        # trainlabeled PatientStudy
        self.bag_of_trainlabeled_PiatentStudy_images_paths, self.bag_of_trainlabeled_PatientStudy_DiagnosisLabels = self._create_bags(part='trainlabeled')
        
        # unlabeled PatientStudy
        self.bag_of_unlabeled_PiatentStudy_images_paths, self.bag_of_unlabeled_PatientStudy_DiagnosisLabels = self._create_bags(part='unlabeled')
        
        # all train PatientStudy
        self.bag_of_all_PatientStudy_images_paths = self.bag_of_trainlabeled_PiatentStudy_images_paths + self.bag_of_unlabeled_PiatentStudy_images_paths
        
        self.bag_of_all_PatientStudy_DiagnosisLabels = self.bag_of_trainlabeled_PatientStudy_DiagnosisLabels + self.bag_of_unlabeled_PatientStudy_DiagnosisLabels

        
        self.num_trainlabeled_PatientStudy = len(self.bag_of_trainlabeled_PiatentStudy_images_paths) 
        self.num_unlabeled_PatientStudy = len(self.bag_of_unlabeled_PiatentStudy_images_paths)
        self.num_all_PatientStudy = len(self.bag_of_all_PatientStudy_images_paths)
        assert self.num_all_PatientStudy == self.num_trainlabeled_PatientStudy + self.num_unlabeled_PatientStudy
        
        prRed('traincombined set. all train PatientStudy: {}, trainlabeled PatientStudy: {}, unlabeled PatientStudy: {}'.format(self.num_all_PatientStudy, self.num_trainlabeled_PatientStudy, self.num_unlabeled_PatientStudy))
        
        self.trainlabeled_index = list(np.array(range(self.num_trainlabeled_PatientStudy)))
        self.unlabeled_index = list(np.array(range(self.num_trainlabeled_PatientStudy, self.num_all_PatientStudy)))
        self.all_index = list(np.array(range(self.num_all_PatientStudy)))
        
        prRed('traincombined set. self.trainlabeled_index: {} - {}, self.unlabeled_index: {} - {}, self.all_index: {} - {}'.format(self.trainlabeled_index[0],self.trainlabeled_index[-1],self.unlabeled_index[0],self.unlabeled_index[-1], self.all_index[0],self.all_index[-1]))
        
        

    def get_fixed_ordered_index(self):
        return self.trainlabeled_index, self.unlabeled_index, self.all_index
    
    def update_unlabeledset_labels(self, survived_original_indexes, survived_predicted_classes):
        
        for index, cls in zip(survived_original_indexes, survived_predicted_classes):
#             prRed('Before updating unlabeledset labels: {}'.format(self.bag_of_all_PatientStudy_DiagnosisLabels[index]))
            self.bag_of_all_PatientStudy_DiagnosisLabels[index] = cls
#             prRed('After updating unlabeledset labels: {}'.format(self.bag_of_all_PatientStudy_DiagnosisLabels[index]))
        
        
    def __len__(self):
        #return the size of total available traincombined_set 
        return len(self.bag_of_all_PatientStudy_images_paths)
    
    
    def __getitem__(self, index):
        
        bag_image_path = self.bag_of_all_PatientStudy_images_paths[index]
        
        bag_image_2D = load_pickle(os.path.join(bag_image_path, '2DImages.pkl'))
        bag_image_2D = bag_image_2D['images'] 
        
        bag_image_Doppler = np.load(os.path.join(bag_image_path, 'dopplers.npy')) #(n_tiff, 160, 200, 3)
        
        
#         print('Inside EchoDataset_traincombined, bag_images shape: {}'.format(bag_image.shape))
        
        #the data in dataloaderV2 is of type numpy array (n_video, 8, 112, 112, 3)
        bag_image_2D_transformed = [torch.stack([self.transform_fn_strong(Image.fromarray(frame)) for frame in video]) for video in bag_image_2D]
        bag_image_2D = torch.stack(bag_image_2D_transformed)            
        
        bag_image_Doppler = torch.stack([self.transform_fn_weak(Image.fromarray(frame)) for frame in bag_image_Doppler])
        
#         print('Inside EchoDataset_traincombined, bag_image_2D shape: {}, bag_image_Doppler shape: {}'.format(bag_image_2D.shape, bag_image_Doppler.shape)) # Inside EchoDataset_traincombined, bag_image_2D shape: torch.Size([66, 8, 3, 112, 112]), bag_image_Doppler shape: torch.Size([42, 3, 160, 200])

#         bag_image = self.bag_of_all_PatientStudy_images[index]
            
#         print('Inside Echo_data_DS1Like, bag_image shape: {}'.format(bag_image.shape))
        
        DiagnosisLabel = self.bag_of_all_PatientStudy_DiagnosisLabels[index]
#         print('DiagnosisLabel: {}'.format(DiagnosisLabel))

#         print('index: {} bag label: {}'.format(index, DiagnosisLabel))

        
        return index, bag_image_2D, bag_image_Doppler, DiagnosisLabel
    
        
        
    def _create_bags(self, part='trainlabeled'):
        
        if part == 'trainlabeled':
            data_root_dir = os.path.join(self.ML_DATA_dir, 'view_and_diagnosis_labeled_set/labeled_unlabeled_combined')
            PatientStudy_list = self.trainlabeled_PatientStudy_list
        elif part == 'unlabeled':
            data_root_dir = os.path.join(self.ML_DATA_dir, 'unlabeled')
            PatientStudy_list = self.unlabeled_PatientStudy_list
        else:
            raise NameError('Invalid part')
            
        
        bag_of_PatientStudy_images_paths = []
#         bag_of_PatientStudy_image_viewrelevance = []
        bag_of_PatientStudy_DiagnosisLabels = []
#         num_cineloop_with_only_1_frame = 0
        
        
        for PatientStudy in PatientStudy_list:
#             if PatientStudy in ['2696500s1', '2961147s1']:
#                 continue
            this_PatientStudy_dir = os.path.join(data_root_dir, PatientStudy)
            print('this_PatientStudy_dir: {}'.format(this_PatientStudy_dir))
            
            #get diagnosis label for this PatientStudy
            this_PatientStudyRecords_from_TMED2SummaryTable = self.TMED2SummaryTable[self.TMED2SummaryTable['patient_study']==PatientStudy]
            assert this_PatientStudyRecords_from_TMED2SummaryTable.shape[0]!=0, 'every PatientStudy from the studylist should be found in TMED2SummaryTable'
            
            this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel = list(set(this_PatientStudyRecords_from_TMED2SummaryTable.diagnosis_label.values)) 
            assert len(this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel)==1, 'every PatientStudy should only have one diagnosis label'
            
            this_PatientStudy_DiagnosisLabel = this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel[0]
            this_PatientStudy_DiagnosisLabel = DiagnosisStr_to_Int_Mapping[this_PatientStudy_DiagnosisLabel]
            
            
            assert os.path.exists(this_PatientStudy_dir), 'every PatientStudy from the studylist should be found {}'.format(this_PatientStudy_dir)
            
            
            this_PatientStudy_2Dimages_data = load_pickle(os.path.join(this_PatientStudy_dir, '2DImages.pkl'))
#             this_PatientStudy_thumbnails_data = load_pickle(os.path.join(this_PatientStudy_dir, 'thumbnails.pkl'))
            
            if this_PatientStudy_2Dimages_data['images'].shape[0]<1:
                print('NO 2D images {}'.format(this_PatientStudy_dir))
                continue
                
            bag_of_PatientStudy_images_paths.append(this_PatientStudy_dir)
            bag_of_PatientStudy_DiagnosisLabels.append(this_PatientStudy_DiagnosisLabel)

            

        
        return bag_of_PatientStudy_images_paths, bag_of_PatientStudy_DiagnosisLabels
    
    
        
        
        
        
        
        
        
        
        

#####################DS1      
class EchoDataset(Dataset):
    
    def __init__(self, PatientStudy_list, TMED2SummaryTable, ML_DATA_dir, training_seed=0, transform_fn=None):
        
        self.PatientStudy_list = PatientStudy_list
        self.TMED2SummaryTable = TMED2SummaryTable #note: using the patient_id column in TMED2SummaryTable can uniquely identify a patient_study (there is NO same patient_study belong to different parts: diagnosis_labeled/, unlabeled/, view_and_diagnosis_labeled_set/, view_labeled AT THE SAME TIME)
        
#         self.ML_DATA_dir = ML_DATA_dir #'Echo_MIL/AS_Diagnosis/ML_DATA/TMED2Release'
        self.ML_DATA_dir = ML_DATA_dir + '_dataloaderV2'#'Echo_MIL/AS_Diagnosis/ML_DATA/TMED2Release'

        self.data_root_dir = os.path.join(self.ML_DATA_dir, 'view_and_diagnosis_labeled_set/labeled_unlabeled_combined') 
    
        self.training_seed=training_seed
        
        self.transform_fn = transform_fn
        
#         self.bag_of_PatientStudy_images_2D, self.bag_of_PatientStudy_images_Doppler, self.bag_of_PatientStudy_DiagnosisLabels = self._create_bags()
        self.bag_MRN, self.bag_of_PatientStudy_images_2D, self.bag_of_PatientStudy_images_Doppler, self.bag_of_PatientStudy_DiagnosisLabels = self._create_bags()
        
        
    
    def _create_bags(self):
        
        bag_of_PatientStudy_images_2D = []
        bag_of_PatientStudy_images_Doppler = []
        
#         bag_of_PatientStudy_image_viewrelevance = []
        bag_of_PatientStudy_DiagnosisLabels = []
#         num_cineloop_with_only_1_frame = 0
        bag_MRN = []

        
        for PatientStudy in self.PatientStudy_list:
            
#             if PatientStudy in ['2696500s1', '2961147s1']:
#                 continue
            this_PatientStudy_dir = os.path.join(self.data_root_dir, PatientStudy)
            
            #get diagnosis label for this PatientStudy
            this_PatientStudyRecords_from_TMED2SummaryTable = self.TMED2SummaryTable[self.TMED2SummaryTable['patient_study']==PatientStudy]
            assert this_PatientStudyRecords_from_TMED2SummaryTable.shape[0]!=0, 'every PatientStudy from the studylist should be found in TMED2SummaryTable'
            
            this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel = list(set(this_PatientStudyRecords_from_TMED2SummaryTable.diagnosis_label.values)) 
            assert len(this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel)==1, 'every PatientStudy should only have one diagnosis label'
            
            this_PatientStudy_DiagnosisLabel = this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel[0]
            this_PatientStudy_DiagnosisLabel = DiagnosisStr_to_Int_Mapping[this_PatientStudy_DiagnosisLabel]
            
            
            assert os.path.exists(this_PatientStudy_dir), 'every PatientStudy from the studylist should be found {}'.format(this_PatientStudy_dir)
            
            
            this_PatientStudy_images_2D = load_pickle(os.path.join(this_PatientStudy_dir, '2DImages.pkl'))['images']
            
            this_PatientStudy_images_Doppler = np.load(os.path.join(this_PatientStudy_dir, 'dopplers.npy'))
            
            if this_PatientStudy_images_2D.shape[0]<1:
                print('NO 2D images {}'.format(this_PatientStudy_dir))
                continue
            
            elif this_PatientStudy_images_Doppler.shape[0]<1:
                print('NO Doppler images {}'.format(this_PatientStudy_dir))
                continue
                
            else:
                
                this_PatientStudy_images_2D = torch.stack([torch.stack([self.transform_fn(Image.fromarray(frame)) for frame in video]) for video in this_PatientStudy_images_2D])
            
                this_PatientStudy_images_Doppler = torch.stack([self.transform_fn(Image.fromarray(frame)) for frame in this_PatientStudy_images_Doppler])
#                 print('this_PatientStudy_images_2D: {}, this_PatientStudy_images_Doppler: {}'.format(this_PatientStudy_images_2D.shape, this_PatientStudy_images_Doppler.shape)) #this_PatientStudy_images_2D: torch.Size([39, 8, 3, 112, 112]), this_PatientStudy_images_Doppler: torch.Size([10, 3, 160, 200])

                
                
                bag_of_PatientStudy_images_2D.append(this_PatientStudy_images_2D)
                bag_of_PatientStudy_images_Doppler.append(this_PatientStudy_images_Doppler)
                bag_of_PatientStudy_DiagnosisLabels.append(this_PatientStudy_DiagnosisLabel)
                bag_MRN.append(PatientStudy)

            

            
#         return bag_of_PatientStudy_images_2D, bag_of_PatientStudy_images_Doppler, bag_of_PatientStudy_DiagnosisLabels
        return bag_MRN, bag_of_PatientStudy_images_2D, bag_of_PatientStudy_images_Doppler, bag_of_PatientStudy_DiagnosisLabels
    
    
    def __len__(self):
        return len(self.bag_of_PatientStudy_images_2D)

    
    def __getitem__(self, index):
        
        bag_image_2D = self.bag_of_PatientStudy_images_2D[index]
        bag_image_Doppler = self.bag_of_PatientStudy_images_Doppler[index]
        
        
        DiagnosisLabel = self.bag_of_PatientStudy_DiagnosisLabels[index]
        MRN = self.bag_MRN[index]
#         print('DiagnosisLabel: {}'.format(DiagnosisLabel))

#         print('index: {} bag label: {}'.format(index, DiagnosisLabel))

#         return bag_image_2D, bag_image_Doppler, DiagnosisLabel
        return MRN, bag_image_2D, bag_image_Doppler, DiagnosisLabel
    
    
    
