# import torch
# import torch.nn as nn




# '''
# Maximize the correlations across multimodal-fused features 
# extracted from MultiAttn through Soft-HGR loss.
# '''
# class SoftHGRLoss(nn.Module):

#     def __init__(self):
#         super().__init__()  
    
    
#     '''
#     Calculate the inner products between feature mappings.
#     '''
#     def feature_mapping(self, feature_X, feature_Y):
#         feature_mapping_X_Y = torch.mean(torch.sum(feature_X * feature_Y, dim = -1), dim = 0)

#         return feature_mapping_X_Y
    

#     '''
#     Calculate the inner products between feature covariances. 
#     '''
#     def feature_covariance(self, feature_X, feature_Y):
#         # Assuming feature_X and feature_Y are [batch_size, seq_len, num_features]
#         # You might want to average across the sequence length or handle differently based on your context.
#         feature_X = feature_X.mean(dim=1)  # Reduce to [batch_size, num_features]
#         feature_Y = feature_Y.mean(dim=1)  # Reduce to [batch_size, num_features]

#         # Computing covariance matrices now on 2D data
#         cov_feature_X = torch.cov(feature_X.T)  # Transpose to [num_features, batch_size]
#         cov_feature_Y = torch.cov(feature_Y.T)  # Transpose to match dimensions for torch.cov

#         # Compute trace of the product of covariance matrices
#         feature_covariance_X_Y = torch.trace(torch.mm(cov_feature_X, cov_feature_Y))

#         # Normalize by number of samples (optional based on scaling requirements)
#         feature_covariance_X_Y /= self.num_samples

#         return feature_covariance_X_Y



#     def forward(self, f_t, f_a, f_v):
#         self.num_samples = f_t.shape[0]

#         all_features = [f_t, f_a, f_v]
#         total_loss = 0.0
#         for i in range(len(all_features) - 1):
#             for j in range(i + 1, len(all_features)):
#                 feature_mapping_i_j = self.feature_mapping(all_features[i], all_features[j])
#                 feature_covariance_i_j = self.feature_covariance(all_features[i], all_features[j])
#                 soft_hgr_loss_i_j = feature_mapping_i_j - feature_covariance_i_j / 2
#                 total_loss += soft_hgr_loss_i_j
        
#         loss = - total_loss / self.num_samples

#         return loss

import torch
import torch.nn as nn




'''
Maximize the correlations across multimodal-fused features 
extracted from MultiAttn through Soft-HGR loss.
'''
class SoftHGRLoss(nn.Module):

    def __init__(self):
        super().__init__()  
    
    
    '''
    Calculate the inner products between feature mappings.
    '''
    def feature_mapping(self, feature_X, feature_Y):
        feature_mapping_X_Y = torch.mean(torch.sum(feature_X * feature_Y, dim = -1), dim = 0)

        return feature_mapping_X_Y
    

    '''
    Calculate the inner products between feature covariances. 
    '''
    def feature_covariance(self, feature_X, feature_Y):
        cov_feature_X = torch.cov(feature_X)
        cov_feature_Y = torch.cov(feature_Y)
        # We empirically find that scaling the feature covariance by a factor of 1 / num_samples 
        # leads to enhanced training stability and improvements in model performances.
        feature_covariance_X_Y = torch.trace(torch.matmul(cov_feature_X, cov_feature_Y)) / self.num_samples
        return feature_covariance_X_Y


    def forward(self, f_t, f_a, f_v):
        self.num_samples = f_t.shape[0]

        all_features = [f_t, f_a, f_v]
        total_loss = 0.0
        for i in range(len(all_features) - 1):
            for j in range(i + 1, len(all_features)):
                feature_mapping_i_j = self.feature_mapping(all_features[i], all_features[j])
                feature_covariance_i_j = self.feature_covariance(all_features[i], all_features[j])
                soft_hgr_loss_i_j = feature_mapping_i_j - feature_covariance_i_j / 2
                total_loss += soft_hgr_loss_i_j
        
        loss = - total_loss / self.num_samples

        return loss