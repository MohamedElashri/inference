from utils import TrackIntervalsToKDE_HDplusUNet100 as Model
from utils import collect_t2kde_arrays
from utils import Loss
from utils import select_gpu
from utils import pv_locations_updated_res, pv_locations_res, filter_nans_res, get_reco_resolution, compare_res_reco

import torch
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

loss = Loss(epsilon=3e-6)
def eventID(intervalNumber):
    eventNumber = int((intervalNumber)/40)
    localInterval = intervalNumber - eventNumber*40
    return eventNumber,localInterval
device = select_gpu(0)
validation = collect_t2kde_arrays('data/pv_HLT1CPU_MinBiasMagDown_14Nov_t2hists_Arrays_validation_allEvents.npy',
                            batch_size=64,
                            pin_memory=True,
                            shuffle=False,)
name = 'weights/07Sept2023_t2hists_HDplusUNet100_iter12Ca_200epochs_2em5_5p0_final.pyt'

nOut1 = 20
nOut2 = 20
nOut3 = 20
nOut4 = 20
nOut5 = 20
latentChannels = 8
nUNetChannels = 64
model = Model(nOut1, nOut2, nOut3, nOut4, nOut5, latentChannels=latentChannels, n=nUNetChannels)

print("our model: \n\n",model)

d = torch.load(name, map_location='cpu')


print(" \n","  for pretrained_dict")
index = 0
for k,v in d.items():
    print("index, k =  ",index,"  ",k)
    index = index+1

model.load_state_dict(d)
model.eval()

with torch.no_grad():
    print("device = ",device)
    print("validation.dataset.tensors[0].shape = ",validation.dataset.tensors[0].shape)
    vdt0 = validation.dataset.tensors[0]
    vdt1 = validation.dataset.tensors[1]
    print("vdt0.shape = ",vdt0.shape)
    print("vdt1.shape = ",vdt1.shape)
    nSplit = []
    for ii in range(256):
        nSplit.append((ii+1)*8000)
        
    print("nSplit = ",nSplit)  
    vdt0Split = torch.tensor_split(vdt0,nSplit, dim=0)
    vdt1Split = torch.tensor_split(vdt1,nSplit, dim=0)

    print("len(vdt0Split) = ",len(vdt0Split))
    
    defaultSplitSize = vdt0Split[0].shape[0]
    print("defaultSplitSize = ",defaultSplitSize)



with torch.no_grad():
    
    iChunk = 0
    print("iChunk = ",iChunk)
    outputs = model(vdt0Split[iChunk]).cpu().numpy()
    ##print("outputs.shape = ",outputs.shape)
    labels = vdt1Split[iChunk].cpu().numpy()
    ##print("label.shape = ",labels.shape)
    inputs = vdt0Split[iChunk].cpu().numpy()
    ##print("inputs.shape = ",inputs.shape)
    print("  ")


listOfEvents = np.arange(50)
for jj in listOfEvents:

    event_label  = np.asarray([])
    event_output = np.asarray([])
    print("jj = ",jj)
    print("inputs.shape = ",inputs.shape)
    event_chisq = 0.
    for interval in range(jj*40,jj*40+40):
        input = inputs[interval]
        label = np.asarray(labels[interval])
        output = np.asarray(outputs[interval])
        
        nFeatures = len(label)
        
        sigma = 0.01
        diff = np.subtract(output,label)
        diffS = diff/sigma
        chisq = np.power(diffS,2)
        nanMask = np.isnan(label)
        countGoods = np.count_nonzero(~np.isnan(label))
        chisq_to_sum = ma.array(chisq,mask=nanMask)                           
        local_ave_chisq = chisq_to_sum.sum()/countGoods
        event_chisq += local_ave_chisq
        ymax = max(np.max(label),np.max(output))
        if (np.isnan(ymax)):
            ymax = 0.5
        if (ymax>0.1):
            ymax = max(1.15*ymax,3.5)
            print(" interval = ",interval,"   ymax = ", ymax)
            myMask = np.zeros(100)
            masker = np.isnan(label)
            myMask = np.ma.array(myMask,mask=masker)
            myMask = myMask.filled(fill_value = 2.0)
            plt.figure()
            plt.plot(output)
            plt.plot(label, color='r')
            plt.bar(np.arange(100),height=myMask,width=0.9,color='lightgrey',edgecolor='lightgrey',alpha=1.0)
            plt.plot(output, color='blue',alpha=0.50)
            plt.bar(np.arange(100),height=output,width=0.9,color='blue',edgecolor='blue',alpha=0.75)
            plt.bar(np.arange(100),height=label,width=0.9,color='red',edgecolor='red',alpha=0.75)
            plt.bar(np.arange(100),height=output,width=0.9,color='blue',edgecolor='blue',alpha=0.5)
            plt.xlim(0.,100.)
            plt.ylim((0.,ymax))
            plt.show()
        
        event_label  = np.concatenate((event_label,label))
        event_output = np.concatenate((event_output,output))
    bin_threshold      = 0.01
    integral_threshold = 0.50
    min_width          = 2
    label32 = np.float32(event_label)
    true_pvs = pv_locations_res(label32,threshold=bin_threshold,integral_threshold=integral_threshold,min_width=min_width)
    predicted_pvs = pv_locations_res(event_output,threshold=bin_threshold,integral_threshold=integral_threshold,min_width=min_width)
    true_pvs_updated = pv_locations_updated_res(event_label,threshold=bin_threshold,integral_threshold=integral_threshold,min_width=min_width)
    predicted_pvs_updated = pv_locations_updated_res(event_output,threshold=bin_threshold,integral_threshold=integral_threshold,min_width=min_width)
    filtered_predicted_pvs = filter_nans_res(predicted_pvs,event_label)
    filtered_predicted_updated_pvs =  filter_nans_res(predicted_pvs_updated,event_label)
    print("  ")
    print(" ---- ")
    print(" ")
    predicted_reso = get_reco_resolution(filtered_predicted_updated_pvs,event_output,1.0,steps_extrapolation=1,ratio_max=0.1,debug=False)
    target_reso = get_reco_resolution(true_pvs_updated,event_label,1.0,steps_extrapolation=1,ratio_max=0.1,debug=False)
    print("true PV positions (in mm)        = ",[ "{:0.2f}".format(0.1*x-100.) for x in  true_pvs_updated])
    print("predicted PV positions (in mm)   = ",[ "{:0.2f}".format(0.1*x-100.) for x in filtered_predicted_updated_pvs])
    print("predicted_reso (in mm)           = ",[ "{:0.3f}".format(0.1*x) for x in predicted_reso])
    print("target_reso (in mm)              = ",[ "{:0.3f}".format(0.1*x) for x in target_reso])
    
    
    reco_window = 5*predicted_reso
    Found, Missed, FalsePositive = compare_res_reco(true_pvs_updated,filtered_predicted_updated_pvs,reco_window,debug=False)
    print("  --> Found, Missed, FalsePositive =  ",Found, Missed, FalsePositive)
    event_ave_chisq = event_chisq/40
    print("event number = ",jj,"    event_ave_chisq = ",event_ave_chisq)
    plt.figure()
    plt.plot(event_output)
    plt.plot(event_label, color='r')
    plt.plot(event_output, color='blue',alpha=0.50)
    ymax = max(1.1*np.max(event_label),1.0)
    ymax = max(1.1*np.max(event_output),1.0)
    if (np.isnan(ymax)):
        ymax = 3.5
    plt.ylim((0.,ymax))
    #plt.show()

        