import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
#https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py

def prRed(prt): print("\033[91m{}\033[0m" .format(prt))
    
class MultimodalNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(MultimodalNetwork, self).__init__()
        
        self.L=250
        self.D=128
        self.num_classes=num_classes
        
        self.encoder_2D = SAMIL()
        self.encoder_Doppler = DopplerEncoder()

    
        self.modality_attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        
        
        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.num_classes),
#             nn.Sigmoid()
        )

    def forward(self, data_2D, data_Doppler):
        
        representation_2D,  A_V= self.encoder_2D(data_2D)
#         print('Inside MultimodalNetwork, representation_2D: {}, A_V: {}'.format(representation_2D.shape, A_V.shape)) #torch.Size([1, 250]), torch.Size([1, 51])
        
        representation_Doppler = self.encoder_Doppler(data_Doppler)
#         print('Inside MultimodalNetwork, representation_Doppler: {}'.format(representation_Doppler.shape)) # torch.Size([1, 250])
        
        combined_representation = torch.cat((representation_2D, representation_Doppler), dim=0)
#         print('Inside MultimodalNetwork, combined_representation: {}'.format(combined_representation.shape)) # torch.Size([2, 250])

        A = self.modality_attention(combined_representation)  
#         print('Inside MultimodalNetwork, A: {} shape: {}'.format(A, A.shape)) #torch.Size([2, 1])
        
        A = torch.transpose(A, 1, 0)  # KxN
#         print('Inside MultimodalNetwork, after transpose A: {} shape: {}'.format(A, A.shape)) # torch.Size([1, 2])

        A = F.softmax(A, dim=1)  # softmax over N
#         print('Inside MultimodalNetwork, after transpose A: {} shape: {}'.format(A, A.shape)) # torch.Size([1, 2])

        
        final_representation = torch.mm(A, combined_representation) 
#         print('Inside MultimodalNetwork, final_representation: {}'.format(final_representation.shape)) # torch.Size([1, 250])

        out = self.classifier(final_representation)
#         print('Inside MultimodalNetwork, out: {} shape: {}'.format(out, out.shape)) # torch.Size([1, 3])
        
        
        
        return out, A_V
    
    
    

class DopplerEncoder(nn.Module):
    def __init__(self):
        super(DopplerEncoder, self).__init__()
        
        self.L=250
        self.D=128
        
        self.encoder = torchvision.models.swin_t(pretrained=True)
        self.encoder.head = nn.Linear(768, self.L) #768 is the architectural-defined dimension in swin_t
        
        
#         self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1]))
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        
        
    def forward(self, x):
#         print('DopplerEncoder input shape: {}'.format(x.shape)) #torch.Size([1, 29, 3, 160, 200])
        x = x.squeeze(0)
#         print('DopplerEncoder ater squeeze, input shape: {}'.format(x.shape)) #torch.Size([29, 3, 160, 200])
        
        
        feat = self.encoder(x)
#         print('DopplerEncoder, feat shape: {}'.format(feat.shape)) #torch.Size([29, 250])
        

        A = self.attention(feat)  # NxK
#         print('DopplerEncoder, A: {} shape: {}'.format(A, A.shape)) #torch.Size([29, 1])

        A = torch.transpose(A, 1, 0)  # KxN
#         print('DopplerEncoder, after transpose A: {} shape: {}'.format(A, A.shape)) #torch.Size([1, 29])

        A = F.softmax(A, dim=1)  # softmax over N
#         print('DopplerEncoder, A after softmax: {} shape: {}'.format(A, A.shape))# torch.Size([1, 29])
        

        final_doppler_representation = torch.mm(A, feat)  # KxL
#         print('DopplerEncoder, final_doppler_representation: {} shape: {}'.format(final_doppler_representation, final_doppler_representation.shape)) # torch.Size([1, 250])

        
        return final_doppler_representation
        

class SAMIL(nn.Module):
    def __init__(self):
        super(SAMIL, self).__init__()
        
        self.freeze_f1_weights = False
        self.freeze_f1_runningstats = False
        self.this_swin_outdim = 768
        
        self.L = 250
        self.D = 128
        self.K = 1
        
#         self.feature_extractor_part1 = torchvision.models.video.r3d_18(pretrained=self.ImageNetWeights)
        self.feature_extractor_part1 = torchvision.models.video.swin3d_t(weights="DEFAULT")
    
        swin_layers = list(self.feature_extractor_part1.children())[:-2]
        
        self.feature_extractor_part1 = nn.Sequential(*swin_layers)
        
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        

        if self.freeze_f1_weights == 'True':
            print('!!!!!!!!!!!!!!!Freeze f1 weights Excuted!!!!!!!!!!!!!')
            # Freeze the parameters of f1
            for param in self.feature_extractor_part1.parameters():
                param.requires_grad = False
        
        if self.freeze_f1_runningstats=='True':
            print('!!!!!!!!!!!!!!!!Freeze f1 running stats Excuted!!!!!!!!!!!!!')
            self.feature_extractor_part1.eval()
        
        
#         self.feature_extractor_part1 = nn.Sequential(
# #             nn.Conv2d(1, 20, kernel_size=5),
#             nn.Conv2d(3, 20, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#              #hz added
#             nn.Conv2d(50, 100, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(100, 200, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#         )

        self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(50 * 4 * 4, self.L),
            nn.Linear(self.this_swin_outdim, self.L),
            nn.ReLU(),
        )
        
#         self.feature_extractor_part3 = nn.Sequential(
            
#             nn.Linear(self.L, self.B),
#             nn.ReLU(),
#             nn.Linear(self.B, self.L),
#             nn.ReLU(),
#         )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

#         self.attention_weights = nn.Linear(self.D, self.K)


    def forward(self, x):
        
        #         print('Inside forward: input x shape: {}'.format(x.shape))
        x = x.squeeze(0)
#         print('Inside forward: after squeeze x shape: {}'.format(x.shape))

        #3D resnet and swin3d_t take the shape (batch_size, channels, temporal_length, height, width), our dataloader returns (batch_size, temporal_length, channel, height, width)
        x = x.permute(0,2,1,3,4)
#         print('Inside forward: after x.permute shape: {}'.format(x.shape))

        H = self.feature_extractor_part1(x)
#         print('Inside forward: after feature_extractor_part1 H shape: {}'.format(H.shape))
       
        H = H.permute(0, 4, 1, 2, 3) #change to # B, C, _T, _H, _W
#         print('Inside forward: after permute shape: {}'.format(H.shape))
        
        
        H = self.avgpool(H)
#         print('Inside forward: after avgpool H shape: {}'.format(H.shape))
        
        
#         H = H.view(-1, 50 * 4 * 4)
        H = H.view(-1, self.this_swin_outdim)
#         print('Inside forward: after view H shape: {}'.format(H.shape))

        H = self.feature_extractor_part2(H)  # NxL
#         print('Inside forward: after feature_extractor_part2 H shape: {}'.format(H.shape))


        A_V = self.attention_V(H)  # NxK
#         print('Inside forward: A_V is {}, shape: {}'.format(A_V, A_V.shape))

        A_V = torch.transpose(A_V, 1, 0)  # KxN
#         print('Inside forward: A_V is {}, shape: {}'.format(A_V, A_V.shape))

        A_V = F.softmax(A_V, dim=1)  # softmax over N
#         print('Inside forward: A_V (View) is {}, shape: {}'.format(A_V, A_V.shape))
        
    
#         H = self.feature_extractor_part3(H)
    
        A_U = self.attention_U(H)  # NxK
#         print('Inside forward: A_U is {}, shape: {}'.format(A_U, A_U.shape))

        A_U = torch.transpose(A_U, 1, 0)  # KxN
#         print('Inside forward: A_U is {}, shape: {}'.format(A_U, A_U.shape))

        A_U = F.softmax(A_U, dim=1)  # softmax over N
#         print('Inside forward: A_U (Diagnosis) is {}, shape: {}'.format(A_U, A_U.shape))
        
#         A = A_V * A_U
#         print('Inside forward: final A is {}, shape: {}'.format(A, A.shape))
        A = torch.exp(torch.log(A_V) + torch.log(A_U)) #numerically more stable when dealing with probabilities?

        A = A/torch.sum(A)
#         A = F.softmax(A, dim=1)
#         print('Inside forward: final A is {}, shape: {}'.format(A, A.shape))
#         A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
#         print('Inside forward: A is {}, shape: {}'.format(A, A.shape))

#         A = torch.transpose(A, 1, 0)  # KxN
# #         print('Inside forward: A is {}, shape: {}'.format(A, A.shape))

#         A = F.softmax(A, dim=1)  # softmax over N
# #         print('Inside forward: A is {}, shape: {}'.format(A, A.shape))

        M = torch.mm(A, H)  # KxL #M can be regarded as final representation of this bag
#         print('Inside forward: M is {}, shape: {}'.format(M, M.shape))
        

        return M, A_V #only view regularize one branch of the attention weights

    

#         out = self.classifier(M)
        

#         return out, A_V #only view regularize one branch of the attention weights

    