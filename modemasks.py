# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:33:30 2022

@author: MODELAB-LP11c

Generates LG mode phase masks library

"""

import sys

import pathlib
p = pathlib.Path(__file__).parent.parent
path_to_fibremodes = p
#print(path_to_xenics)
sys.path.append(str(path_to_fibremodes))

import time
#import mark_lib as mkl
import cupy as cp
#import ModesGen as mg
from fibremodes import ModesGen as mg

from pylab import *

def Mask_Modes_GS_gpu( wavelength , f_SMF , f_MMF , MMF_d, pixel_size , 
                      mask_size , w0_slm, MMF_mfd, max_group, crosstalk = 25, filtering = 2, 
                      downscale = 1, ext_FACTOR = 0, num_masks_out = False):
    wlength = wavelength
    goal_crosstalk = crosstalk
    #SLM plane calculation:
    px_slm = pixel_size
    num_px_slm = mask_size
    ext_samples = ext_FACTOR * num_px_slm 
    total_samples = (num_px_slm + ext_samples) //downscale
    samples_slm = num_px_slm // downscale #number of samples
    slm_size = (px_slm*downscale) * (num_px_slm // downscale) #in um --> Real size
    x_slm = arange(-samples_slm//2,samples_slm//2,1) * (px_slm*downscale)
    X_slm, Y_slm = meshgrid(x_slm,x_slm)
    R = sqrt(X_slm**2 + Y_slm**2)
    area = zeros((total_samples,total_samples))
    total_length = (px_slm*downscale) * (total_samples)
    #SLM illumination beam
    slm_ill = exp(-(R/w0_slm)**2) #exp(-(R/w0_slm)**2)
    slm_ill = slm_ill / sqrt(sum(abs(slm_ill)**2))
    c = area.shape[0]//2
    dd = slm_ill.shape[0]//2
    area[c-dd:c+dd,c-dd:c+dd] = slm_ill
    INPUT_BEAM = area    
    #Fibre faced plane(K-space)
    k0 = 2 *np.pi / wlength
    samples = total_samples# SLM number of pixels / downscale
    r = MMF_d / 2 #radius of the fiber
    px_mmf = wlength * f_MMF / total_length #um
    x = arange(-samples//2,samples//2,1) * px_mmf
    X,Y = np.meshgrid(x,x) 
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)
    #Generate the modes
    LGbases = mg.LGmodes(MMF_mfd, max_group, samples, px_mmf, generateModes = True, wholeSet = False, engine='GPU', multicore = True)
    #Assigning modes to other variables
    mm_gra = LGbases.LGmodesArray
    index = LGbases.index
    #delete de object
    del LGbases
    #GPU memory managment
    mempool = mg.mgcl.cp.get_default_memory_pool()
    mempool.free_all_blocks()
    #time.sleep(15)
    #Spatial Filter
    core_diam_samples = dd / px_mmf
    RR = sqrt(X**2 + Y**2)
    R0 = zeros((samples,samples))
    R1 = zeros( (samples,samples) )
    RR_gpu = cp.sqrt(X_gpu**2 + Y_gpu**2)
    R0_gpu = cp.zeros((samples,samples))
    R1_gpu = cp.zeros( (samples,samples) )
    crop = int(core_diam_samples)
    crop = filtering*r
    R0_gpu[RR_gpu < crop] = 1
    
    def filter(sel,plt=False):
        fil_mod = zeros(R0.shape)
        fil_selectivity = sel
        wfil = X.shape[0] // fil_selectivity
        fil = exp(-((RR-crop ) /wfil)**2)
        fil_mod = (fil*(negative(R0)+1))
        fil_mod = fil_mod / fil_mod.max()
        if plt == True:
            imshow(fil_mod,cmap='jet'),colorbar()
            figure()
            plot(fil_mod[fil.shape[0]//2,:]),title('Spatial Filter Profile')
        return( fil_mod )    
    
    #GS parameters allocation
    Modes = cp.asarray(mm_gra).astype(cp.complex64) #Modes at the fiber output (already normalized)
    SLM_MODE = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(Modes))) #Modes at the SLM plane
    SLM_MODE = SLM_MODE / cp.sqrt(cp.sum(cp.absolute(SLM_MODE**2),(1,2)))[:,None,None] #Normalize power 1
    ######Reshaping stuff###### --> Mode bases are at the SLM plane (FULL RESOLUTION)
    Modes1D_bases = cp.reshape(Modes,(Modes.shape[0],-1))
    Modes1D_bases = cp.conj(cp.transpose(Modes1D_bases))
    ###################
    modes_idx = index
    num_indep_modes = cp.asarray(array(where(modes_idx[1,:] == 0 )).shape[1])
    num_modes = Modes.shape[0]
    Total_modes = ((Modes.shape[0] - num_indep_modes) * 2 ) + num_indep_modes
    phase_mask_gpu = cp.zeros( (int(Total_modes),)+ Modes.shape[-2:] , cp.complex64) # Index the final masks
    power_ratio = cp.zeros(num_modes)
    similarity = cp.zeros(num_modes)
    FMatrix = cp.zeros((Modes.shape[0],Modes.shape[0]),cp.complex64)
    INPUT_BEAM_gpu = cp.asarray(INPUT_BEAM).astype(cp.complex64)
    ####
    filter_mod = filter(sel=2,plt=False)
    filter_mod_gpu = cp.asarray(filter_mod)    
    
    ##############SELECT NUM OF MASK TO OUTPUT##########################
    num_modes_length = num_modes
    if num_masks_out == False:
        num_modes = arange(num_modes)
    elif type(num_masks_out) ==  tuple and len(num_masks_out) == 2:
        num_modes = arange(num_masks_out[0], num_masks_out[1])
    else:
         num_modes = arange(num_modes)
         
    ##############SELECT NUM OF MASK TO OUTPUT##########################

    #GS the shit
    completeSet = False
    counter = 0
    for j in num_modes: #Loop for each mask --> one mask for each mode
        Target_Mode = Modes[j]
        Target_Mode = Target_Mode / cp.sqrt(cp.sum(cp.absolute(Target_Mode**2))) # Target
        SLM_Target_Mode = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(Target_Mode)))
        SLM_Target_Mode = SLM_Target_Mode / sqrt(cp.sum(cp.absolute(SLM_Target_Mode**2)))# Target in SLM plane
        A = SLM_Target_Mode
        B = INPUT_BEAM_gpu #input
        w = 1 #scatter power weight
        iterations = 0
        filter_ON = True #Keeps the power close to the imposed limited avoiding super high freq.
        overlap_w = cp.zeros(num_modes_length)
        Max_Iterations = 300 #Before to crash
        #Modified GS   
        for i in range (Max_Iterations):
            C = cp.absolute(B) * cp.exp(1j*cp.angle(A)) #SLM plane
            D = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(C))) #Fiber plane
            D = D / sqrt(cp.sum(cp.absolute(D)**2)) #Fiber plane normalized to 1

            #Control the ampount of power delivered to the fiber or to the clading:
            #w parameter controls how much light goes to the core and how much goes to the clading
            #Total power always must be 1

            scatter = ( D * (cp.negative(R0_gpu) + 1) ) #This gets some trash light pattern outside the core
            scatter_p = cp.sqrt(cp.sum(cp.absolute(scatter)**2)) #power of the scatter field
            scatter = (scatter / scatter_p) #Normalize scattered power to unitary power
            scatter2 = cp.sqrt(1-w) * scatter # decrease w throwing more light outside
            if filter_ON == False:
                scatter = filter_mod_gpu * scatter #Apodization of the scatter field, dont generate super crazy high freq.
                scatter_p = cp.sqrt(cp.sum(cp.absolute(scatter)**2)) #Normalize it again to power = 1
                scatter = scatter/scatter_p
                scatter2 = cp.sqrt(1-w) * scatter


            E_p = cp.sqrt(cp.sum(cp.absolute((Target_Mode*R0_gpu)**2))) #This is what I want where exactly I want it (inside core)
            E1 = Target_Mode * R0_gpu / E_p
            E2 = E1 * w #Scaled decrease the power there
            E = E2 + scatter2 #Create the whole field (Core + cladding with power = 1)
            #print(sum(abs(E)**2))
            A = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E))) #Back to SLM plane
            #A_norm = A / sqrt(cp.sum(cp.absolute(A)**2))

            #w = w - 1/Max_Iterations
            w = (w*0.99) #This is the amount of power we get rid at every iteration

            #Goal checking at the Fiber plane --> Probably should be better to check this at SLM where resolution is better
            D_1D = cp.reshape(D,-1).astype(cp.complex64) # This is my mode at the SLM (abs(Gaussian) + angle(Best one))
            FMatrix[j,:] = cp.matmul(D_1D,Modes1D_bases)
            overlap_w = FMatrix[j,:]
            P_goal = cp.absolute(overlap_w[j])**2
            crosstalk = cp.sum(cp.absolute(overlap_w)**2) - P_goal
            SNR_dB = 10*cp.log10(P_goal / crosstalk)


            if SNR_dB>=goal_crosstalk:
                power_ratio[j] = cp.sum(cp.absolute(scatter2)**2) * 100 # Percentage of power scatter outside the core
                similarity[j] = P_goal
                print('Mode',j,'Done - ','Iteration ',i)
                show()
                break
            elif i == Max_Iterations-1:
                power_ratio[j] = cp.sum(cp.absolute(scatter2)**2) * 100 # Percentage of power scatter outside the core
                similarity[j] = P_goal
                print('Mode', j, 'crased - ', 'Iteration ', i)
                show()
            cp.cuda.Stream.null.synchronize()
        computed_GS = cp.absolute(INPUT_BEAM_gpu) * cp.exp(1j*cp.angle(C)) #INPUT ILLUMINATION + computed mask phase
        #computed_GS = cp.absolute(SLM_Target_Mode) * cp.exp(1j*cp.angle(C)) #Target Mode at the SLM + computed mask phase
        cp.cuda.Stream.null.synchronize()
        if completeSet:
            phase_mask_gpu[counter,...] = (computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2])
            counter+=1
        else:
            if modes_idx[1,j] == 0: #zero n index in Larrange poly          
                #phase_mask.append((computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2]))
                phase_mask_gpu[counter,...] = (computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2])
                counter+=1
            else:
                #phase_mask.append((computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2]))
                phase_mask_gpu[counter,...] = (computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2])
                counter+=1
                #cmplex = (cp.conj(computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2]) * exp(1j*pi))
                #Conjugate mirror the array -> F*(-kx,-ky) -- FT --> f*(x,y) --- SLM mask mirrored and conjugated = Target* mode
                cmplex = cp.flipud(cp.fliplr(cp.conj(computed_GS[c-samples_slm//2:c+samples_slm//2,c-samples_slm//2:c+samples_slm//2])))
                #phase_mask.append(cmplex)
                phase_mask_gpu[counter,...] = (cmplex)
                counter+=1

    phase_mask = cp.asnumpy(phase_mask_gpu) # Here I am returning X num mask corresponding to X number of modes 
    #- I could reduce the size if I am only interested in a few mask --> case of num_masks_out != False
    power_ratio = cp.asnumpy(power_ratio)
    similarity = cp.asnumpy(similarity)
    MT = cp.asnumpy(FMatrix)

    if completeSet == False:  
        EFFICIENCY_debug = array(mg.mgcl.ComputeAllLGmodes_list(similarity,index))
    else:
        EFFICIENCY_debug = similarity
        
    result = {'masks':phase_mask,
              'efficiency':EFFICIENCY_debug,
              'TM':MT}
        
    
    #Cleaning time
    
    del Modes
    del SLM_MODE
    del Modes1D_bases
    del modes_idx
    del num_indep_modes
    del phase_mask_gpu
    del power_ratio
    del similarity
    del FMatrix
    del INPUT_BEAM_gpu
    del filter_mod_gpu
    
    del X_gpu
    del Y_gpu
    del RR_gpu
    del R0_gpu
    del R1_gpu
    
    del Target_Mode
    del SLM_Target_Mode
    del A
    del B
    del C
    del D
    del overlap_w
    del scatter
    del scatter_p
    del scatter2
    del E_p
    del E1
    del E2
    del E
    del D_1D
    del P_goal
    del crosstalk
    del SNR_dB
    del computed_GS
    del cmplex
    
    #Very important step - cupy does not clean anything for you and the fft planning needs space. I could maybe reserve this memory space at the begining to avoid planning all the time in case 
    #mode generation is run on a loop
    cache = cp.fft.config.get_plan_cache()
    cache.clear()

    mempool.free_all_blocks()
        
    return(result)

if __name__ == '__main__':
    

    masks = Mask_Modes_GS_gpu( wavelength = 1.3 , f_SMF = 25e3 , f_MMF = 10e3  , 
                          pixel_size = 9.2 , MMF_d = 62.5 , mask_size = 1024 , w0_slm = 1.945025401e3 ,
                          MMF_mfd = 13.714671176 , max_group = 22 ,
                          crosstalk = 25, downscale = 1, ext_FACTOR = 0, num_masks_out=(16,20))
    


