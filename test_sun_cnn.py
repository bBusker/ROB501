import torch
import torch.utils.data
import numpy as np
import scipy.io as sio
from sun_cnn import CNN, dataloader

'''
Initialize the Network
'''
binsize=45 #degrees
bin_edges = np.arange(-180,180+1,binsize)
num_bins = bin_edges.shape[0] - 1
cnn = CNN(num_bins) #Initialize our CNN Class
cnn.load_state_dict(torch.load('best_model_{}.pth'.format(binsize)))


dsets = {x: dataloader('{}.mat'.format(x),binsize=binsize, mode='test') for x in ['test']} 
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=50, shuffle=False, num_workers=4) for x in ['test']}

pred_list = np.zeros((0))
top1_incorrect = 0
for mode in ['test']:    #iterate 
    cnn.train(False)    # Set model to Evaluation mode
    cnn.eval()

    for data in dset_loaders[mode]:    #Iterate through all data (each iteration loads a minibatch)
        image = data
        image = image.type(torch.FloatTensor)

        pred = cnn(image)   # Forward pass through the network
        _, predicted = torch.max(pred.data, 1) #from the network output, get the class prediction
        pred_list = np.hstack((pred_list, predicted.numpy()))

print("Testing with a binsize of {} degrees - saving predictions to predictions_{}.txt".format(binsize,binsize))

np.savetxt('predictions_{}.txt'.format(binsize),pred_list.astype(int))
