import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import os, sys, time
import awkward as ak
from functools import partial
from math import sqrt as sqrt


####################################################################################################
def select_gpu(selection=None):
    """
    Select a GPU if available.

    selection can be set to get a specific GPU. If left unset, it will REQUIRE that a GPU be selected by environment variable. If -1, the CPU will be selected.
    """

    if str(selection) == "-1":
        return torch.device("cpu")  # Corrected "CPU" to "cpu"

    # This must be done before any API calls to Torch that touch the GPU
    if selection is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selection)

    if not torch.cuda.is_available():
        print("Selecting CPU (CUDA not available)")
        return torch.device("cpu")  # Corrected "CPU" to "cpu"
    elif selection is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is *required* when running with CUDA available"
        )

    print(torch.cuda.device_count(), "available GPUs (initially using device 0):")
    for i in range(torch.cuda.device_count()):
        print(" ", i, torch.cuda.get_device_name(i))

    return torch.device("cuda:0")
####################################################################################################


class Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-5, coefficient=1.0):
        '''
        Epsilon is a parameter that can be adjusted.
        coefficient adjust asymmetry; 1.0 <==> symmetric
        '''

        # You must call the original constructor (torch.nn.Module.__init__(self))!
        super().__init__()
        
        # Now you can add things
        self.epsilon = epsilon
        self.coefficient = coefficient

    def forward(self, x, y):
        nFeatures = 100
        nEvts = y.shape[0]
        print("nEvts = ", nEvts)
        print("y.shape = ",y.shape)
        print("x.shape = ".x.shape)

        y_kde = y  
        # Compute r, only including non-nan values. r will probably be shorter than x and y.
        valid = ~torch.isnan(y_kde)
        r = torch.abs((x[valid] + self.epsilon) / (y_kde[valid] + self.epsilon))

        # Compute -log(2r/(r² + 1))
        alpha = -torch.log(2*r / (r**2 + 1))
        alpha = alpha * (1.0 + self.coefficient * torch.exp(-r))

        beta = alpha.sum() / nFeatures


        sigma = 0.01
        diff = torch.sub(x[valid],y_kde[valid])
        diff = diff/sigma
        chisq = torch.pow(diff,2)
        ave_chisq = chisq.sum()/nFeatures
        chi4 = 0.00001*torch.pow(diff,4)

        ave_chi4 = chi4.sum()/nFeatures

        ave_beta  = beta/nEvts
        ave_chisq = ave_chisq/nEvts
        ave_chi4  = ave_chi4/nEvts


        return ave_chisq
####################################################################################################

# Use ak.Array instead of awkward.JaggedArray
def collect_t2kde_arrays(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    slice=None,
    **kargs,
):
    """
    This function collects arrays. It does not split it up. You can pass in multiple files.
    Example: collect_data('a.h5', 'b.h5')

    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

    Xlist = []
    Ylist = []

    for XY_file in files:
        print("XY_file = ", XY_file)
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), open(XY_file, "rb") as f:
            X_in = np.load(f)
            Y_in = np.load(f)
            Xlist.append(X_in)
            Ylist.append(Y_in)

    X = np.concatenate(Xlist, axis=0)
    Y = np.concatenate(Ylist, axis=0)
    print("outer loop X.shape = ", X.shape)

    if slice:
        X = X[slice, :]
        Y = Y[slice, :]

    with Timer(start=f"Constructing {X.shape[0]} event dataset"):
        x_t = torch.tensor(X)
        y_t = torch.tensor(Y)

        if device is not None:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

        dataset = TensorDataset(x_t, y_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)
    print("x_t.shape = ", x_t.shape)
    print("x_t.shape[0] = ", x_t.shape[0])
    print("x_t.shape[1] = ", x_t.shape[1])
    print("y_t.shape = ", y_t.shape)

    return loader
####################################################################################################

class Timer(object):
    __slots__ = "message verbose start_time".split()

    def __init__(self, message=None, start=None, verbose=True):
        """
        If message is None, add a default message.
        If start is not None, then print start then message.
        Turn off all printing with verbose.
        """

        if verbose and start is not None:
            print(start, end="", flush=True)
        if message is not None:
            self.message = message
        elif start is not None:
            self.message = " took {time:.4} s"
        else:
            self.message = "Operation took {time:.4} s"

        self.verbose = verbose
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.verbose:
            print(self.message.format(time=self.elapsed_time()))
####################################################################################################

class ConvBNrelu(nn.Sequential):
    """convolution => [BN] => ReLU

    This class simply combines a few layers into a "block", which will be very commonly used throughout multiple different models.
    You can specify the parameters of the conv layer when you initialize, and the rest will be automatically sorted out.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super(ConvBNrelu, self).__init__(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(p)
#         Swish_module(),
)

class Up(nn.Sequential):
    """transpose convolution => convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ConvBNrelu(out_channels, out_channels, kernel_size=kernel_size, p=p))


####################################################################################################

downsample_options = {
    'ConvBNrelu':ConvBNrelu,
}

upsample_options = {
    'Up':Up,
}

def combine(x, y, mode='concat'):
    if mode == 'concat':
        return torch.cat([x, y], dim=1)
    elif mode == 'add':
        return x+y
    else:
        raise RuntimeError(f'''Invalid option {mode} from choices 'concat' or 'add' ''')


####################################################################################################
class TrackIntervalsToKDE_HDplusUNet100(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=20, nOut2=20, nOut3=20, nOut4=20, nOut5=20, 
                 latentChannels=8, n=64, sc_mode='concat', dropout_p=.25, 
                 dropout_fc6=0.20, d_selection='ConvBNrelu', u_selection='Up'):
        super(TrackIntervalsToKDE_HDplusUNet100, self).__init__()

        # Layer parameters
        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
        self.nOut4 = nOut4
        self.nOut5 = nOut5
        self.latentChannels = latentChannels

        # Linear layers to process input data
        self.layer1 = nn.Linear(in_features=9, out_features=self.nOut1, bias=True)
        self.layer2 = nn.Linear(in_features=self.layer1.out_features, out_features=self.nOut2, bias=True)
        self.layer3 = nn.Linear(in_features=self.layer2.out_features, out_features=self.nOut3, bias=True)
        self.layer4 = nn.Linear(in_features=self.layer3.out_features, out_features=self.nOut4, bias=True)
        self.layer5 = nn.Linear(in_features=self.layer4.out_features, out_features=self.nOut5, bias=True)

        # The latent representation outputting `latentChannels * 100` bins for intervals
        self.layer6A = nn.Linear(in_features=self.layer5.out_features, out_features=latentChannels * 100, bias=True)

        # ======================================================================
        #                            U-Net Models
        # ======================================================================

        # Concatenation factor based on `sc_mode`
        factor = 2 if sc_mode == 'concat' else 1
        self.mode = sc_mode

        # Dropout layers
        self.fc6dropout = nn.Dropout(dropout_fc6)
        self.p = dropout_p

        # Ensure valid downsampling and upsampling block selections
        assert d_selection in downsample_options.keys(), f'Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}'
        assert u_selection in upsample_options.keys(), f'Selection for upsampling block {u_selection} not present in available options - {upsample_options.keys()}'

        # Downsampling and upsampling blocks
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]

        # Downsampling layers
        self.rcbn1 = d_block(self.latentChannels, n, kernel_size=25, p=dropout_p)
        self.rcbn2 = d_block(n, n, kernel_size=7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size=5, p=dropout_p)

        # Upsampling layers
        self.up1 = u_block(n, n, kernel_size=5, p=dropout_p)
        self.up2 = u_block(n * factor, n, kernel_size=5, p=dropout_p)

        # Output layers
        self.out_intermediate = nn.Conv1d(n * factor, n, 5, padding=2)
        self.outc = nn.Conv1d(n, 1, 5, padding=2)

        # MaxPooling layer for downsampling
        self.d = nn.MaxPool1d(2)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)

        # Get dimensions of input: (batch_size, nFeatures, nTrks)
        nEvts = x.shape[0]
        nFeatures = x.shape[1]
        nTrks = x.shape[2]

        # Generate masks based on track values greater than -98
        mask = x[:, 0, :] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1, -1, 100)

        # Transpose input for feature processing
        x = x.transpose(1, 2)

        # Pass through linear layers with LeakyReLU activations
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        x = leaky(self.layer6A(x))  # Produces latentChannels x 100 bins for intervals

        # Print Track 0 features and its corresponding Latent mapping layer natively across PyTorch
        
        # Apply dropout
        x = self.fc6dropout(x)

        # Reshape the output
        x = x.view(nEvts, nTrks, self.latentChannels, 100)

        # Multiply mask and sum over tracks
        f2 = torch.unsqueeze(f2, 2)
        
        x = torch.mul(f2, x)
        y0 = torch.sum(x, dim=1)

        # Pass through downsampling layers
        x1 = self.rcbn1(y0)  # Size: 100
        x2 = self.d(self.rcbn2(x1))  # Size: 50
        x = self.d(self.rcbn3(x2))  # Size: 25

        # Pass through upsampling layers
        x = self.up1(x)  # Size: 50
        x = self.up2(combine(x, x2, mode=self.mode))  # Size: 100

        # Final output layers
        x = self.out_intermediate(combine(x, x1, mode=self.mode))  # Size: 100
        logits_x0 = self.outc(x)  # Final convolution layer

        # Apply softplus activation and squeeze to remove extra dimensions
        y_prime = F.softplus(logits_x0).squeeze()

        # Scale the final output
        y_pred = torch.mul(y_prime, 0.001)
        return y_pred

####################################################################################################

def pv_locations_updated_res(
    targets,
    threshold,
    integral_threshold,
    min_width
):
    """
    Compute the z positions from the input KDE using the parsed criteria.
    
    Inputs:
      * targets: 
          Numpy array of KDE values (predicted or true)

      * threshold: 
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold: 
          The total integral required to trigger a hit - such as 0.2

      * min_width: 
          The minimum width (in bins) of a feature - such as 2

    Returns:
      * array of float32 values corresponding to the PV z positions
      
    """
    # Counter of "active bins" i.e. with values above input threshold value
    state = 0
    # Sum of active bin values
    integral = 0.0
    # Weighted Sum of active bin values weighted by the bin location
    sum_weights_locs = 0.0

    # Make an empty array and manually track the size (faster than python array)
    items = np.empty(150, np.float32)
    # Number of recorded PVs
    nitems = 0

    # Account for special case where two close PV merge KDE so that
    # targets[i] never goes below the threshold before the two PVs are scanned through
    peak_passed = False
    local_peak_value = 0.0
    
    # Loop over the bins in the KDE histogram
    for i in range(len(targets)):
        # If bin value above 'threshold', then trigger
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_weights_locs += i * targets[i]  # weight times location

            if (targets[i]>local_peak_value):
                local_peak_value = targets[i]
                local_peak_index = i
## -------------------------------------

            if ((targets[i-1]>targets[i]+0.05) and (targets[i-1]>1.1*targets[i])):
                peak_passed = True
            
        if (targets[i] < threshold or i == len(targets) - 1 or (targets[i-1]<targets[i] and peak_passed)) and state > 0:
            #if (targets[i] < threshold or i == len(targets) - 1) and state > 0:

            # Record a PV only if 
            if state >= min_width and integral >= integral_threshold:
                # Adding '+0.5' to account for the bin width (i.e. 50 microns)
                items[nitems] = (sum_weights_locs / integral) + 0.5 
                nitems += 1

            # reset state
            state = 0
            integral = 0.0
            sum_weights_locs = 0.0
            peak_passed=False
            local_peak_value = 0.0
            
    return items[:nitems]


####################################################################################################

def pv_locations_res(
    targets,
    threshold,
    integral_threshold,
    min_width
):
    """
    Compute the z positions from the input KDE using the parsed criteria.
    
    Inputs:
      * targets: 
          Numpy array of KDE values (predicted or true)

      * threshold: 
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold: 
          The total integral required to trigger a hit - such as 0.2

      * min_width: 
          The minimum width (in bins) of a feature - such as 2

    Returns:
      * array of float32 values corresponding to the PV z positions
      
    """
    # Counter of "active bins" i.e. with values above input threshold value
    state = 0
    # Sum of active bin values
    integral = 0.0
    # Weighted Sum of active bin values weighted by the bin location
    sum_weights_locs = 0.0

    # Make an empty array and manually track the size (faster than python array)
    items = np.empty(150, np.float32)
    # Number of recorded PVs
    nitems = 0

    # Loop over the bins in the KDE histogram
    for i in range(len(targets)):
        # If bin value above 'threshold', then trigger
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_weights_locs += i * targets[i]  # weight times location

        if (targets[i] < threshold or i == len(targets) - 1) and state > 0:

            # Record a PV only if 
            if state >= min_width and integral >= integral_threshold:
                # Adding '+0.5' to account for the bin width (i.e. 50 microns)
                items[nitems] = (sum_weights_locs / integral) + 0.5 
                nitems += 1

            # reset state
            state = 0
            integral = 0.0
            sum_weights_locs = 0.0


    return items[:nitems]
####################################################################################################

def filter_nans_res(
    items,
    mask
):
    """
    Method to mask bins in the predicted KDE array if the corresponding bin in the true KDE array is 'nan'.
    
    Inputs:
      * items: 
          Numpy array of predicted PV z positions

      * mask: 
          Numpy array of KDE values (true PVs)


    Returns:
      * Numpy array of predicted PV z positions
      
    """
    # Create empty array with shape array of predicted PV z positions
    retval = np.empty_like(items)
    # Counter of 
    max_index = 0
    # Loop over the predicted PV z positions
    for item in items:
        index = int(round(item))
        not_valid = np.isnan(mask[index])
        if not not_valid:
            retval[max_index] = item
            max_index += 1

    return retval[:max_index]


####################################################################################################

def get_reco_resolution(
    pred_PVs_loc,
    predict,
    nsig_res,
    steps_extrapolation,
    ratio_max,
    debug
):
    """
    Compute the resolution as a function of predicted KDE histogram 

    Inputs:
      * pred_PVs_loc: 
          Numpy array of computed z positions of the predicted PVs (using KDEs)

      * predict: 
          Numpy array of predictions

      * nsig_res: 
          Empirical value representing the number of sigma wrt to the std resolution 
          as a function of FHWM

      * threshold: 
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold: 
          The total integral required to trigger a hit - such as 0.2

      * min_width: 
          The minimum width (in bins) of a feature - such as 2

      * debug: 
          flag to print output for debugging purposes


    Ouputs: 
        Numpy array of filtered and sorted (in z values) expected resolution on the reco PVs z position.
    """
    
    #    # Get the z position from the predicted KDEs distribution
    #    predict_values = pv_locations_updated_res(predict, threshold, integral_threshold, min_width)

    

    predict = np.nan_to_num(predict)
    reco_reso = np.empty_like(pred_PVs_loc)

    rms = 1./sqrt(12.)

    steps = steps_extrapolation
    
    i_predict_pv=0
        
    if steps==0:

        # This is for the case where we do not extrapolate values in between bins
        for predict_pv in pred_PVs_loc:
            predict_pv_ibin = int(predict_pv)
            predict_pv_KDE_max = predict[predict_pv_ibin]

            FHWM = ratio_max*predict_pv_KDE_max

            ibin_min = -1
            ibin_max = -1

            for ibin in range(predict_pv_ibin,predict_pv_ibin-20,-1):
                predict_pv_KDE_val = predict[ibin]
                if predict_pv_KDE_val<FHWM:
                    ibin_min = ibin
                    break

            for ibin in range(predict_pv_ibin,predict_pv_ibin+20):
                predict_pv_KDE_val = predict[ibin]
                if predict_pv_KDE_val<FHWM:
                    ibin_max = ibin
                    break

            FHWM_w = (ibin_max-ibin_min)
            if(debug): 
                print("FHWM_w",FHWM_w)
            stantdard_dev = FHWM_w/2.335
            reco_reso[i_predict_pv] = nsig_res*stantdard_dev
            i_predict_pv+=1
                
    else:

        if (debug):
            print(" pred_PVs_loc = ",pred_PVs_loc)        
        for predict_pv in pred_PVs_loc:
            predict_pv_ibin = int(predict_pv)
            predict_pv_KDE_max = predict[predict_pv_ibin]

            FHWM = ratio_max*predict_pv_KDE_max

            if (debug):
                print(" ***** ")
                print(" step != 0 ")
                print(" predict_pv,  predict_pv_ibin,  predict_pv_KDE_max = ",
                        predict_pv,  predict_pv_ibin,  predict_pv_KDE_max)

            ibin_min_extrapol = -1
            ibin_max_extrapol = -1
            found_min = False
            found_max = False
            for ibin in range(predict_pv_ibin,predict_pv_ibin-20,-1):
                if not found_min:
                    predict_pv_KDE_val_ibin = predict[ibin]
                    predict_pv_KDE_val_prev = predict[ibin-1]

                    # Apply a dummy linear extrapolation between the two neigbour bins values 
                    delta_steps = (predict_pv_KDE_val_prev - predict_pv_KDE_val_ibin)/steps
                    for sub_bin in range(int(steps)):
                        predict_pv_KDE_val_ibin -= delta_steps*sub_bin

                        if predict_pv_KDE_val_ibin<FHWM:
                            ibin_min_extrapol = int(ibin*steps-sub_bin)/steps
                            found_min=True

            for ibin in range(predict_pv_ibin,predict_pv_ibin+20):
                if not found_max:
                    predict_pv_KDE_val_ibin = predict[ibin]
                    predict_pv_KDE_val_next = predict[ibin+1]

                    # Apply a dummy linear extrapolation between the two neigbour bins values 
                    delta_steps = (predict_pv_KDE_val_ibin - predict_pv_KDE_val_next)/steps
                    for sub_bin in range(int(steps)):
                        predict_pv_KDE_val_ibin -= delta_steps*sub_bin

                        if predict_pv_KDE_val_ibin<FHWM:
                            ibin_max_extrapol = (ibin*steps+sub_bin)/steps
                            found_max=True
                sumsq = 0.
                sumContents = 0.
                if (found_min and found_max):
                  for index in range (int(ibin_min_extrapol),int(ibin_max_extrapol)+1):
                    contents = predict[index]
                    if (debug):
                      print("index, contents = ",index,contents)
                    sumsq += (index+0.5-predict_pv)*(index+0.5-predict_pv)*contents
                    sumContents += contents
                    if (debug):
                      print("index+0.05, predict_pv, contents, sumsq, sumContents = ",
                             index+0.05, predict_pv, contents, sumsq, sumContents)
                  rms = sumsq/sumContents
                  if (debug):
                      print("rms = {:0.2f}".format(rms))
                  if (debug):
                     print("rms = {:0.2f}".format(rms))
                     print("  ")


            if ( debug and (not (found_min and found_max)) ):
              print(" not (found_min and found_max) ")
            FHWM_w = (ibin_max_extrapol-ibin_min_extrapol)
            if (debug):
                print("FHWM_w",FHWM_w)
            stantdard_dev = FHWM_w/2.335
            reco_reso[i_predict_pv] = nsig_res*stantdard_dev
            reco_reso[i_predict_pv] = nsig_res*rms
            i_predict_pv+=1
        
    return reco_reso
####################################################################################################
def compare_res_reco(
    target_PVs_loc,
    pred_PVs_loc,
    reco_res,
    debug
):
    """
    Method to compute the efficiency counters: 
    - succeed    = number of successfully predicted PVs
    - missed     = number of missed true PVs
    - false_pos  = number of predicted PVs not matching any true PVs

    Inputs argument:
      * target_PVs_loc: 
          Numpy array of computed z positions of the true PVs (computed from target histograms)

      * pred_PVs_loc: 
          Numpy array of computed z positions of the predicted PVs (computed from predicted histograms)

      * reco_res: 
          Numpy array with the "reco" resolution computed from predicted histograms

      * debug: 
          flag to print output for debugging purposes
    
    
    Returns:
        succeed, missed, false_pos
    """
    
    # Counters that will be iterated and returned by this method
    succeed = 0
    missed = 0
    false_pos = 0
        
    # Get the number of predicted PVs
    len_pred_PVs_loc = len(pred_PVs_loc)
    # Get the number of true PVs 
    len_target_PVs_loc = len(target_PVs_loc)

    # Decide whether we have predicted equally or more PVs than trully present
    # this is important, because the logic for counting the MT an FP depend on this
    if len_pred_PVs_loc >= len_target_PVs_loc:
        if debug:
            print("In len(pred_PVs_loc) >= len(target_PVs_loc)")

        # Since we have N(pred_PVs) >= N(true_PVs), 
        # we loop over the pred_PVs, and check each one of them to decide 
        # whether they should be labelled as S, FP. 
        # The number of MT is computed as: N(true_PVs) - S
        # Here the number of iteration is fixed to the original number of predicted PVs
        for i in range(len_pred_PVs_loc):
            if debug:
                print("pred_PVs_loc = ",pred_PVs_loc[i])
            # flag to check if the predicted PV is being matched to a true PV
            matched = 0

            # Get the window of interest: [min_val, max_val] 
            # The window is obtained from the value of z of the true PV 'j'
            # +/- the resolution as a function of the number of tracks for the true PV 'j'
            min_val = pred_PVs_loc[i]-reco_res[i]
            max_val = pred_PVs_loc[i]+reco_res[i]
            if debug:
                print("resolution = ",(max_val-min_val)/2.)
                print("min_val = ",min_val)
                print("max_val = ",max_val)

            # Now looping over the true PVs.
            for j in range(len(target_PVs_loc)):
                # If condition is met, then the predicted PV is labelled as 'matched', 
                # and the number of success is incremented by 1
                if min_val <= target_PVs_loc[j] and target_PVs_loc[j] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ",succeed)
                    # the true PV is removed from the original array to avoid associating 
                    # one predicted PV to multiple true PVs
                    # (this could happen for PVs with close z values)
                    target_PVs_loc = np.delete(target_PVs_loc,[j])
                    # Since a predicted PV and a true PV where matched, go to the next predicted PV 'i'
                    break
            # In case, no true PV could be associated with the predicted PV 'i'
            # then it is assigned as a FP answer
            if not matched:                
                false_pos +=1
                if debug:
                    print("false_pos = ",false_pos)
        # the number of missed true PVs is simply the difference between the original 
        # number of true PVs and the number of successfully matched true PVs
        missed = (len_target_PVs_loc-succeed)
        if debug:
            print("missed = ",missed)

    else:
        if debug:
            print("In len(pred_PVs_loc) < len(target_PVs_loc)")
        for i in range(len_target_PVs_loc):
            if debug:
                print("target_PVs_loc = ",target_PVs_loc[i])
            # flag to check if the true PV is being matched to a predicted PV
            matched = 0
            # Now looping over the predicted PVs.
            for j in range(len(pred_PVs_loc)):                
                # Get the window of interest: [min_val, max_val] 
                # The window is obtained from the value of z of the true PV 'i'
                # +/- the resolution as a function of the number of tracks for the true PV 'i'
                min_val = pred_PVs_loc[j]-reco_res[j]
                max_val = pred_PVs_loc[j]+reco_res[j]
                if debug:
                    print("pred_PVs_loc = ",pred_PVs_loc[j])
                    print("resolution = ",(max_val-min_val)/2.)
                    print("min_val = ",min_val)
                    print("max_val = ",max_val)
                # If condition is met, then the true PV is labelled as 'matched', 
                # and the number of success is incremented by 1
                if min_val <= target_PVs_loc[i] and target_PVs_loc[i] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ",succeed)
                    # the predicted PV is removed from the original array to avoid associating 
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    pred_PVs_loc = np.delete(pred_PVs_loc,[j])
                    # Since a predicted PV and a true PV where matched, go to the next true PV 'i'
                    reco_res = np.delete(reco_res,[j])
                    break
            # In case, no predicted PV could be associated with the true PV 'i'
            # then it is assigned as a MT answer
            if not matched:
                missed += 1
                if debug:
                    print("missed = ",missed)
                    
        # the number of false positive predicted PVs is simply the difference between the original 
        # number of predicted PVs and the number of successfully matched predicted PVs
        false_pos = (len_pred_PVs_loc - succeed)
        if debug:
            print("false_pos = ",false_pos)

    return succeed, missed, false_pos
####################################################################################################