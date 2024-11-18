import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import PIL
from PIL import Image
import clip
from functools import partial
import random
import json
from typing import Union, List, Optional
from tqdm import tqdm
import utils
import open_clip
import diffusers

    
#Subclass for GNET
class TrunkBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super(TrunkBlock, self).__init__()
        self.conv1 = nn.Conv2d(feat_in, int(feat_out*1.), kernel_size=3, stride=1, padding=1, dilation=1)
        self.drop1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(feat_in, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(self.conv1.bias, 0.0) # current
        
    def forward(self, x):
        return torch.nn.functional.relu(self.conv1(self.drop1(self.bn1(x))))

#Subclass for GNET
class PreFilter(nn.Module):
    def __init__(self):
        super(PreFilter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )        
        
    def forward(self, x):
        c1 = self.conv1(x)
        y = self.conv2(c1)
        return y 

#Subclass for GNET
class EncStage(nn.Module):
    def __init__(self, trunk_width=64, pass_through=64):
        super(EncStage, self).__init__()
        self.conv3  = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=0)
        self.drop1  = nn.Dropout2d(p=0.5, inplace=False) ##
        self.bn1    = nn.BatchNorm2d(192, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True) ##
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ##
        self.tw = int(trunk_width)
        self.pt = int(pass_through)
        ss = (self.tw + self.pt)
        self.conv4a  = TrunkBlock(128, ss)
        self.conv5a  = TrunkBlock(ss, ss)
        self.conv6a  = TrunkBlock(ss, ss)
        self.conv4b  = TrunkBlock(ss, ss)
        self.conv5b  = TrunkBlock(ss, ss)
        self.conv6b  = TrunkBlock(ss, self.tw)
        ##
        torch.nn.init.xavier_normal_(self.conv3.weight, gain=torch.nn.init.calculate_gain('relu'))        
        torch.nn.init.constant_(self.conv3.bias, 0.0)
        
    def forward(self, x):
        c3 = (torch.nn.functional.relu(self.conv3(self.drop1(self.bn1(x))), inplace=False))
        c4a = self.conv4a(c3)
        c4b = self.conv4b(c4a)
        c5a = self.conv5a(self.pool1(c4b))
        c5b = self.conv5b(c5a)
        c6a = self.conv6a(c5b)
        c6b = self.conv6b(c6a)

        return [torch.cat([c3, c4a[:,:self.tw], c4b[:,:self.tw]], dim=1), 
                torch.cat([c5a[:,:self.tw], c5b[:,:self.tw], c6a[:,:self.tw], c6b], dim=1)], c6b
    
#Subclass for GNET
class GEncoder(nn.Module):
    def __init__(self, mu, trunk_width, pass_through=64 ):
        super(GEncoder, self).__init__()
        self.mu = nn.Parameter(torch.from_numpy(mu), requires_grad=False) #.to(device)
        self.pre = PreFilter()
        self.enc = EncStage(trunk_width, pass_through) 

    def forward(self, x):
        fmaps, h = self.enc(self.pre(x - self.mu))
        return x, fmaps, h

#Main GNET model class
class Torch_LayerwiseFWRF(nn.Module):
    def __init__(self, fmaps, nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_LayerwiseFWRF, self).__init__()
        self.fmaps_shapes = [list(f.size()) for f in fmaps]
        self.nf = np.sum([s[1] for s in self.fmaps_shapes])
        self.pre_nl  = pre_nl
        self.post_nl = post_nl
        self.nv = nv
        ##
        self.rfs = []
        self.sm = nn.Softmax(dim=1)
        for k,fm_rez in enumerate(self.fmaps_shapes):
            rf = nn.Parameter(torch.tensor(np.ones(shape=(self.nv, fm_rez[2], fm_rez[2]), dtype=dtype), requires_grad=True))
            self.register_parameter('rf%d'%k, rf)
            self.rfs += [rf,]
        self.w  = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        self.b  = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(self.nv,)).astype(dtype=dtype), requires_grad=True))
        
    def forward(self, fmaps):
        phi = []
        for fm,rf in zip(fmaps, self.rfs): #, self.scales):
            g = self.sm(torch.flatten(rf, start_dim=1))
            f = torch.flatten(fm, start_dim=2)  # *s
            if self.pre_nl is not None:          
                f = self.pre_nl(f)
            # fmaps : [batch, features, space]
            # v     : [nv, space]
            phi += [torch.tensordot(g, f, dims=[[1],[2]]),] # apply pooling field and add to list.
            # phi : [nv, batch, features] 
        Phi = torch.cat(phi, dim=2)
        if self.post_nl is not None:
            Phi = self.post_nl(Phi)
        vr = torch.squeeze(torch.bmm(Phi, torch.unsqueeze(self.w,2))).t() + torch.unsqueeze(self.b,0)
        return vr
    
class GNet8_Encoder():
    
    def __init__(self, subject = 1, device = "cuda", model_path = "gnet_multisubject.pt"):
        
        # Setting up Cuda
        self.device = torch.device(device)
        torch.backends.cudnn.enabled=True
        # Subject number
        self.subject = subject
        
        # Vector type
        self.vector = "images"
        
        # x size
        subject_sizes = [0, 15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386]
        self.x_size = subject_sizes[self.subject]
        
        # Reload joined GNet model files
        self.joined_checkpoint = torch.load(model_path, map_location=self.device)
        
        self.subjects = list(self.joined_checkpoint['voxel_mask'].keys())
        self.gnet8j_voxel_mask = self.joined_checkpoint['voxel_mask']
        self.gnet8j_voxel_roi  = self.joined_checkpoint['voxel_roi']
        self.gnet8j_voxel_index= self.joined_checkpoint['voxel_index']
        self.gnet8j_brain_nii_shape= self.joined_checkpoint['brain_nii_shape']
        self.gnet8j_val_cc = self.joined_checkpoint['val_cc']
        
            
    
    def load_image(self, image_path):
        
        image = PIL.Image.open(image_path).convert('RGB')
        
        w, h = 227, 227  # resize to integer multiple of 64
        imagePil = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
        image = np.array(imagePil).astype(np.float32) / 255.0
        
        return image  
    
    # Rebuild Model
    def _model_fn(self, _ext, _con, _x):
        '''model consists of an extractor (_ext) and a connection model (_con)'''
        _y, _fm, _h = _ext(_x)
        return _con(_fm)

    def _pred_fn(self, _ext, _con, xb):
        return self._model_fn(_ext, _con, torch.from_numpy(xb).to(self.device))  
                    
    def subject_pred_pass(self, _pred_fn, _ext, _con, x, batch_size):
        pred = _pred_fn(_ext, _con, x[:batch_size]) # this is just to get the shape
        pred = np.zeros(shape=(len(x), pred.shape[1]), dtype=np.float32) # allocate
        for rb,_ in utils.iterate_range(0, len(x), batch_size):
            pred[rb] = utils.get_value(_pred_fn(_ext, _con, x[rb]))
        return pred

    def gnet8j_predictions(self, image_data, _pred_fn, trunk_width, pass_through, checkpoint, mask, batch_size, device=torch.device("cuda:0")):
        
        subjects = list(image_data.keys())

        if(mask is None):
            subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()} 
        else:
            subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()}    
            subject_nv[subjects[0]] = int(torch.sum(mask == True)) 

        # allocate
        subject_image_pred = {s: np.zeros(shape=(len(image_data[s]), subject_nv[s]), dtype=np.float32) for s in subjects}
        # print(subject_image_pred)
        _log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)
        
        best_params = checkpoint['best_params']
        # print(best_params)
        shared_model = GEncoder(np.array(checkpoint['input_mean']).astype(np.float32), trunk_width=trunk_width, pass_through=pass_through).to(device)
        shared_model.load_state_dict(best_params['enc'])
        shared_model.eval() 

        # example fmaps
        rec, fmaps, h = shared_model(torch.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))                                     
        for s in subjects:
            sd = Torch_LayerwiseFWRF(fmaps, nv=subject_nv[s], pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
            params = best_params['fwrfs'][s]
            
            if(mask is None):
                sd.load_state_dict(params)
            
            else:
                masked_params = {}
                for key, value in params.items():
                    masked_params[key] = value[mask]
                    
                sd.load_state_dict(masked_params)
                
            # print(params['w'].shape)
            # print(params['b'].shape)
            # sd.load_state_dict(best_params['fwrfs'][s])
            sd.eval() 
            # print(sd)
            
            subject_image_pred[s] = self.subject_pred_pass(_pred_fn, shared_model, sd, image_data[s], batch_size)

        return subject_image_pred

    def predict(self, images, mask = None):
        self.stim_data = {}
        data = []
        w, h = 227, 227  # resize to integer multiple of 64
        
        if(isinstance(images, list)):
            for i in range(len(images)):
                
                imagePil = images[i].convert("RGB").resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
                image = np.array(imagePil).astype(np.float32) / 255.0
                data.append(image)
            
        elif(isinstance(images, torch.Tensor)):
            for i in range(images.shape[0]):
                
                imagePil = utils.process_image(images[i], w, h)
                image = np.array(imagePil).astype(np.float32) / 255.0
                data.append(image)
            
        
        self.stim_data[self.subject] = np.moveaxis(np.array(data), 3, 1)

        gnet8j_image_pred = self.gnet8j_predictions(self.stim_data, self._pred_fn, 64, 192, self.joined_checkpoint, mask, batch_size=100, device=self.device)

        return torch.from_numpy(gnet8j_image_pred[self.subject])
    
