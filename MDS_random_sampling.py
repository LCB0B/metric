
# -*- coding: utf-8 -*-


from copy import deepcopy
# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as f# create a dummy data 
import matplotlib.pyplot as plt
import networkx as nx
import timeit
from sklearn import metrics
import scipy.sparse as sparse
import scipy.stats as stats
from sklearn import metrics
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd
from torch_sparse import spspmm
import pandas as pd




class MDS(nn.Module):
    def __init__(self,data,relation, input_size,latent_dim,sample_size,device):
        super(MDS, self).__init__()
        self.input_size=input_size
       
        #self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
        
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device))
        
        #self.alpha=nn.Parameter(torch.randn(self.input_size,device=device))
        #create indices to index properly the receiver and senders variable

        self.relation = relation
        self.pdist = nn.PairwiseDistance(p=2,eps=0)

        self.sampling_weights=torch.ones(self.input_size,device=device)
        self.sample_size=sample_size


        self.latent_z=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))

        self.latent_z.data=torch.randn(self.input_size,latent_dim,device=device)

    

    
    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,torch.ones(edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
        
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]

        return sample_idx,sparse_i_sample,sparse_j_sample


    def sample(self):
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        return sample_idx
        
    
    
    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def MDS_likelihood(self,epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        '''
        self.epoch=epoch

        #sample_idx,sparse_sample_i,sparse_sample_j=self.sample_network()

        # mat=torch.exp(-((self.latent_z[sample_idx].unsqueeze(1)-self.latent_z[sample_idx]+1e-06)**2).sum(-1)**0.5)
        # z_pdist1=0.5*torch.mm(torch.exp(self.gamma[sample_idx].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma[sample_idx]).unsqueeze(-1))))
        # #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
        # z_pdist2=(-((((self.latent_z[sparse_sample_i]-self.latent_z[sparse_sample_j]+1e-06)**2).sum(-1)))**0.5+self.gamma[sparse_sample_i]+self.gamma[sparse_sample_j]).sum()
        # log_likelihood_sparse=z_pdist2-z_pdist1
        sample_idx = self.sample()
        log_likelihood = ((torch.cdist(self.latent_z[sample_idx],self.latent_z[sample_idx]) - self.relation[sample_idx][:,sample_idx])**2 / self.relation[sample_idx][:,sample_idx].fill_diagonal_(5) ).sum()**0.5
        
        

        return log_likelihood

    def link_prediction(self):

        with torch.no_grad():
            z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_z[self.removed_j])**2).sum(-1))**0.5
            logit_u_miss=-z_pdist_miss+self.gamma[self.removed_i]+self.gamma[self.removed_j]
            rates=torch.exp(logit_u_miss)
            self.rates=rates

            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            #fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    
    
    def get_latent_coord(self):
        return self.latent_z.data
    
