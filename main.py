from copy import deepcopy
# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as f  # create a dummy data
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
import MDS_random_sampling

start = timeit.default_timer()
CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.set_default_tensor_type('torch.FloatTensor')
if device.type != 'cpu':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train(n_epochs, epoch_rez, sample_rate, dataset, latent_dim, save, plot):
    print(latent_dim)
    print(dataset)
    losses = np.zeros(n_epochs)
    # ROC = np.zeros(n_epoch s/ /epoch_rez)
    # PR = np.zeros(n_epoch s/ /epoch_rez)

    # ################################################################################################################################################################
    # ################################################################################################################################################################

    relation = torch.from_numpy(np.loadtxt(dataset + '/relation.csv', delimiter=",")).to(device)

    # network size
    N = len(relation)
    # sample size of blocks-> sample_size*(sample_size-1)/2 pairs
    sample_size = int(sample_rate * N)
    # Missing_data refers to dyads that are not observed, setting it to True does not consider these pairs in the likelihood
    # Missing_data should be set to False for link_prediction since we do not consider these interactions as missing but as zeros.
    # def __init__(self,sparse_i,sparse_j, input_size,latent_dim,sample_size,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None):
    model = MDS_random_sampling.MDS(torch.randn(N, latent_dim), relation, N, latent_dim=latent_dim,
                                    sample_size=sample_size, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), 0.01)

    for epoch in range(n_epochs):

        loss = model.MDS_likelihood(epoch=epoch) / sample_size
        losses[epoch] = loss.item()

        optimizer.zero_grad()  # clear the gradients.
        loss.backward()  # backpropagate
        optimizer.step()  # update the weights
        if epoch % epoch_rez == 0:
            # roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr

            print('Epoch: ', epoch)
            print('Loss: ', loss.item())
            # print('ROC:',roc)
            # print('PR:',pr)
            # ROC[epoch//epoch_rez] = roc
            # PR[epoch//epoch_rez] = pr
    if save:
        # Save latent and loss
        z = model.get_latent_coord()
        z_np = z.numpy()
        z_df = pd.DataFrame(z_np)
        z_df.to_csv(f'output/{dataset}_{latent_dim}_{n_epochs}_{sample_rate}_coord.csv')

        np.savetxt(f'output/{dataset}_{latent_dim}_{n_epochs}_{sample_rate}_loss.csv', losses, delimiter=",")

    if plot:
        plt.figure()
        plt.plot(losses)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(f'output/{dataset}_{latent_dim}_{n_epochs}_{sample_rate}_loss.png')

        plt.figure()
        plt.scatter(z_np[:,0],z_np[:,1],s=0.1)
        plt.savefig(f'output/{dataset}_{latent_dim}_{n_epochs}_{sample_rate}_scatter.png')
    return


plt.style.use('ggplot')
torch.autograd.set_detect_anomaly(True)
# cv=CV_missing_data(input_size=full_rank.shape[0],sparse_i_idx=sparse_i,sparse_j_idx=sparse_j,percentage=0.2)
# sparse_i_rem_cv,sparse_j_rem_cv,non_sparse_i_cv,non_sparse_j_cv=cv.CV_Missing_ij()

latent_dims = [2]
dataset = 'mnist'
n_epochs = 100
epoch_rez = 10
sample_rates = [0.01, 0.1, 1]
save = 1
plot = 1

for latent_dim in latent_dims:
    for sample_rate in sample_rates:
        train(n_epochs, epoch_rez, sample_rate, dataset, latent_dim, save, plot)
