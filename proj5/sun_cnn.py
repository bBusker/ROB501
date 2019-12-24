import torch
import torch.utils.data 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#REMOVE AFTER+=========================================================================================
# from visdom import Visdom
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### set a random seed for reproducibility (do not change this)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

### For network v5
# def conv(in_planes, out_planes, kernel_size=3):
#     return torch.nn.Sequential(
#         torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
#         torch.nn.BatchNorm2d(out_planes),
#         torch.nn.ReLU(inplace=True),
#         # torch.nn.Dropout(0.3)
#     )

def conv(in_planes, out_planes, kernel_size=3):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm2d(out_planes),
        torch.nn.Dropout(0.2)
    )

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def fc(in_feat, out_feat):
    return torch.nn.Sequential(
        torch.nn.Linear(in_feat, out_feat),
        torch.nn.ReLU(inplace=True),
    )

### Define the Convolutional Neural Network Model
class CNN(torch.nn.Module):
    def __init__(self, num_bins): 
        super(CNN, self).__init__()
        self.feat_layers = 16
        
        ### Network v1
        # nn_block = []
        # nn_block += [torch.nn.Conv2d(3, 16, stride=2, kernel_size=(9,9))]
        # nn_block += [torch.nn.MaxPool2d((3,3), stride=3)]
        # nn_block += [torch.nn.ReLU()]
        # nn_block += [torch.nn.Conv2d(16, 16, stride=1, kernel_size=(5,5))]#, padding=1)]
        # nn_block += [torch.nn.MaxPool2d((3, 3), stride=1)]
        # nn_block += [torch.nn.ReLU()]
        # nn_block += [torch.nn.Dropout(0.5)]
        # nn_block += [torch.nn.Conv2d(16, num_bins, kernel_size=(4,30))]

        ### Network v5
        # nn_block = []
        # nn_block += conv(3, self.feat_layers, 1)
        # nn_block += conv(self.feat_layers, self.feat_layers, 7)
        # nn_block += conv(self.feat_layers, self.feat_layers, 5)
        # nn_block += [torch.nn.MaxPool2d((3,3), stride=3)]
        # nn_block += [torch.nn.Conv2d(self.feat_layers, self.feat_layers, kernel_size=3, padding=1, stride=1), torch.nn.ReLU(inplace=True)]
        # nn_block += [torch.nn.Dropout(0.5)]
        # nn_block += [torch.nn.Conv2d(self.feat_layers, num_bins, kernel_size=(3, 9), padding=0)]

        ### Network v6
        nn_block = []
        nn_block += conv(3, self.feat_layers, 9)
        nn_block += conv(self.feat_layers, self.feat_layers, 7)
        nn_block += [torch.nn.MaxPool2d((3, 3), stride=3)]
        nn_block += conv(self.feat_layers, self.feat_layers * 2, 5)
        nn_block += conv(self.feat_layers * 2, self.feat_layers * 2, 3)
        nn_block += [torch.nn.MaxPool2d((3, 3), stride=3)]
        nn_block += [Flatten(), fc(5376, 4096), torch.nn.Dropout(0.5)]
        nn_block += [torch.nn.Linear(4096, num_bins)]


        self.network = torch.nn.Sequential(*nn_block)

    ###Define what the forward pass through the network is
    def forward(self, x):
        x = self.network(x)
        x = x.squeeze()

        return x

### Define the custom PyTorch dataloader for this assignment
class dataloader(torch.utils.data.Dataset):
    """Loads the KITTI Odometry Benchmark Dataset"""
    def __init__(self, matfile, binsize=45, mode='train'):
        self.data = sio.loadmat(matfile)
        
        self.images = self.data['images']
        # Zero mean
        self.images -= 94.98759942130943
        self.mode = mode
        
        if self.mode != 'test':
            self.azimuth = self.data['azimuth']
            ### Generate targets for images by 'digitizing' each azimuth angle into the appropriate bin (from 0 to num_bins)
            bin_edges = np.arange(-180,180+1,binsize)
            self.targets = (np.digitize(self.azimuth,bin_edges) -1).reshape((-1))

    def __len__(self):
        return int(self.images.shape[0])
  
    def __getitem__(self, idx):
        img = self.images[idx]

        if self.mode != 'test':
            return img, self.targets[idx]
        else:
            return img

# For visualization --- REMOVE LATER
# class VisdomLinePlotter(object):
#     """Plots to Visdom"""
#     def __init__(self, env_name='main'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}
#     def plot(self, var_name, split_name, title_name, x, y):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=title_name,
#                 xlabel='Epochs',
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

if __name__ == "__main__":
    '''
    Visdom
    For visualization -- REMOVE LATER
    '''
    # global plotter
    # plotter = VisdomLinePlotter(env_name="sun_cnn")

    '''
    Initialize the Network
    '''
    binsize=20 #degrees **set this to 20 for part 2**
    bin_edges = np.arange(-180,180+1,binsize)
    num_bins = bin_edges.shape[0] - 1
    cnn = CNN(num_bins) #Initialize our CNN Class
    cnn.cuda()
    
    '''
    Uncomment section to get a summary of the network (requires torchsummary to be installed):
        to install: pip install torchsummary
    '''
    from torchsummary import summary
    inputs = torch.zeros((1,3,68,224))
    summary(cnn, input_size=(3, 68, 224))
    
    '''
    Training procedure
    '''
    
    CE_loss = torch.nn.CrossEntropyLoss(reduction='sum') #initialize our loss (specifying that the output as a sum of all sample losses)
    params = list(cnn.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-5, weight_decay=0.0) #initialize our optimizer (Adam, an alternative to stochastic gradient descent)
    step_lr = torch.optim.lr_scheduler.StepLR
    scheduler = step_lr(optimizer, step_size=5, gamma=0.5)
    
    ### Initialize our dataloader for the training and validation set (specifying minibatch size of 128)
    dsets = {x: dataloader('{}.mat'.format(x),binsize=binsize) for x in ['train', 'val']} 
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'val']}
    
    loss = {'train': [], 'val': []}
    top1err = {'train': [], 'val': []}
    top5err = {'train': [], 'val': []}
    best_err = 1
    ### Iterate through the data for the desired number of epochs
    for epoch in range(0,200):
        # scheduler.step()
        # print("LR {}: ".format(scheduler.get_lr()))
        for mode in ['train', 'val']:    #iterate 
            epoch_loss=0
            top1_incorrect = 0
            top5_incorrect = 0
            if mode == 'train':
                cnn.train(True)    # Set model to training mode
            else:
                cnn.train(False)    # Set model to Evaluation mode
                cnn.eval()
            
            dset_size = dset_loaders[mode].dataset.__len__()
            for data in dset_loaders[mode]:    #Iterate through all data (each iteration loads a minibatch)
                image, target = data
                image, target = image.type(torch.cuda.FloatTensor), target.type(torch.cuda.LongTensor)
                
                optimizer.zero_grad()    #zero the gradients of the cnn weights prior to backprop
                pred = cnn(image)   # Forward pass through the network
                minibatch_loss = CE_loss(pred, target)  #Compute the minibatch loss
                epoch_loss += minibatch_loss.item() #Add minibatch loss to the epoch loss 
                
                if mode == 'train': #only backprop through training loss and not validation loss       
                    minibatch_loss.backward()
                    optimizer.step()
                
                _, predicted = torch.max(pred.data, 1) #from the network output, get the class prediction
                top1_incorrect += (predicted != target).sum().item() #compute the Top 1 error rate
                
                top5_val, top5_idx = torch.topk(pred.data,5,dim=1)
                top5_incorrect += ((top5_idx != target.view((-1,1))).sum(dim=1) == 5).sum().item() #compute the top5 error rate
    
                
            loss[mode].append(epoch_loss/dset_size)
            top1err[mode].append(top1_incorrect/dset_size)
            top5err[mode].append(top5_incorrect/dset_size)
    
            print("{} Loss: {}".format(mode, loss[mode][epoch]))
            print("{} Top 1 Error: {}".format(mode, top1err[mode][epoch]))    
            print("{} Top 5 Error: {}".format(mode, top5err[mode][epoch])) 
            if mode == 'val':
                print("Completed Epoch {}".format(epoch))
                if top1err['val'][epoch] < best_err:
                    best_err = top1err['val'][epoch]
                    best_epoch = epoch
                    torch.save(cnn.state_dict(), 'best_model_{}.pth'.format(binsize))

        ## For visualization REMOVE LATER
        # plotter.plot('acc', 'train', 'Top1Acc', epoch, top1err['train'][-1])
        # plotter.plot('acc', 'val', 'Top1Acc', epoch, top1err['val'][-1])
        # plotter.plot('loss', 'train', 'Loss', epoch, loss['train'][-1])
        # plotter.plot('loss', 'val', 'Loss', epoch, loss['val'][-1])



           
    print("Training Complete")
    print("Lowest validation set error of {} at epoch {}".format(np.round(best_err,2), best_epoch))        
    # '''
    # Plotting
    # '''
    # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.grid()
    # ax1.plot(loss['train'],linewidth=2)
    # ax1.plot(loss['val'],linewidth=2)
    # #ax1.legend(['Train', 'Val'],fontsize=12)
    # ax1.legend(['Train', 'Val'])
    # ax1.set_title('Objective', fontsize=18, color='black')
    # ax1.set_xlabel('Epoch', fontsize=12)
    #
    # ax2.grid()
    # ax2.plot(top1err['train'],linewidth=2)
    # ax2.plot(top1err['val'],linewidth=2)
    # ax2.legend(['Train', 'Val'])
    # ax2.set_title('Top 1 Error', fontsize=18, color='black')
    # ax2.set_xlabel('Epoch', fontsize=12)
    #
    # ax3.grid()
    # ax3.plot(top5err['train'],linewidth=2)
    # ax3.plot(top5err['val'],linewidth=2)
    # ax3.legend(['Train', 'Val'])
    # ax3.set_title('Top 5 Error', fontsize=18, color='black')
    # ax3.set_xlabel('Epoch', fontsize=12)
    # plt.tight_layout()
    # plt.show()
    # fig.savefig('net-train.pdf')