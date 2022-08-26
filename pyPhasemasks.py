"""
Created on Tue Jan 11 16:33:30 2022

@author: MODELAB-LP11c

Class that creates enviroment from user desired specification that 
Generates arbitrary phase masks from a set of intput LG coeficients
under the created enviroment.

Enviroment:
    -Optical System specs: waveleght, focal lenght at the illumination , focal lenght at the fibre coupling -- By default SMF side regular fibre specs
    -SLM specs: numer of pixels avaliable, pixel size
    -fibre specs: diameter, Max number of mode groups
    -GS specs: downscale, upsample, goal_fidelity, spatial filter, gaussian filter
"""

import sys

import pathlib
p = pathlib.Path(__file__).parent.parent
path_to_MODULES = p
#print(path_to_MODULES)
sys.path.append(str(path_to_MODULES))

from pylab import *
import cupy as cp
from fibremodes import ModesGen as mg
#from fibremodes.overlaps import *


###################### FUTURE REFACTORING - USING COMPOSITION ############################
# I was thinking to create a object that contains all parameters of the set up. 
# Init this object can be done using json dictionary or manually
# Use the instance of that object to Init the proper GS object
class GS_gpu_workpack:
    def __init__(self, **kargs):
        pass

class calculate_slm_plane:
    pass
        
class calculate_k_space_plane:
    pass

class calculate_Illumination:
    pass

class calculate_filters:
    pass

class allocate_GPU_memory:
    pass

###################### FUTURE REFACTORING - USING COMPOSITION ############################

class SetupGenericMaskGSenvioment():
    def __init__(self, wavelength, fin, fout, masksize, slm_pixel_size, MMF_mfd, MMFdiameter, MaxModegroups, maskscount = 2, goal_fidelity = 0.99, downscale = 1, upsample = 0, MAX_iterations = 300, W0_slm = None):
        print('Initializing GS enviroment ')
        #define object atributes
        ############## INIT PARAMETERS ################
        self.wavelngth = wavelength
        self.fin = fin
        self.fout = fout
        self.masksize = masksize #obviusly keep it below number of avaliable pixel on the slm
        self.slm_pixel_size  = slm_pixel_size
        self.MMF_mfd = MMF_mfd
        self.MMFdiameter = MMFdiameter
        self.r_mmf = MMFdiameter/2
        self.MaxModegroups = MaxModegroups
        self.maskcount = maskscount
        self.goal_fidelity = goal_fidelity
        self.downscale = downscale # Allows to increase computation performance at expenses of reducing Kmax (at MMF plane) - Mask can be upscaled later on
        self.upsample = upsample # Allows to increase resolution at the MMF plane at expenses of increasing computation time - Mask will be cropped to desired specs
        self.MAX_Iterations = MAX_iterations
        self.W0_slm = W0_slm
    
        self._lastAtribute = 1 #I use to indicate after this entry the rest are contructor atribbutes
        
        ################# INIT PARAMETERS ###########################
        #Calculate slm plane (the whole area of the SLM includding aretificial extension, Area in polar only real slm, number of samples of the whooe AREA, Exntesion in um, samples of the real slm (including downsampling))
        print("SLM plane:")
        self.SLM_PLANE, self.R, self.SLM_PLANE_N, self.SLM_PLANE_L, self.SLM_N = self.define_slm_plane(pixelcount = masksize, pixel_size = slm_pixel_size, downscale = downscale, upsampling = upsample)
        self.SLM_center = self.SLM_PLANE_N//2
        #ILLUMINATION BEAM
        self.dsmf = 9 #single mode fibre core diameter
        self.NAsmf = 0.1 #Numerical aperture
        print("Illumination:")
        self.ILLUMINATION = self.define_IlluminationBeam(wavelength, self.R, self.SLM_PLANE, fin, self.dsmf,self.NAsmf, slm_W0 = self.W0_slm)
        #Calculate K-SPACE
        print("K-Space:")
        self.pixel_mmf, self.Xk, self.Yk = self.define_Kspace(wlength = wavelength, N = self.SLM_PLANE_N, L = self.SLM_PLANE_L, f = fout )
        self.Xk_gpu = cp.asarray(self.Xk)
        self.Yk_gpu = cp.asarray(self.Yk)
        #Calculate modes
        self.LGmodes = None
        self.LGindex = None
        self.mempool = None
        self.calculateLGmodes()
        self.modescount = self.LGmodes.shape[0]
        #Filter calculation
        print("Damping light Filter:")
        self.R0_gpu = None
        self.gaussianFilter_gpu = None
        self.calcFilterMask_gpu(croppingFactor = 4, selectivity = 2)
        
        #Allocate all the memory to perform the GS
        print("Allocating memory in the GPU")
        Modes = cp.asarray(self.LGmodes).astype(cp.complex64) #Modes at the fiber output (already normalized)
        self.Modes1D_bases = cp.reshape(Modes,(Modes.shape[0],-1))
        self.Modes1D_bases_conj = cp.conj(cp.transpose(self.Modes1D_bases)) 
        del(Modes)
        self.mempool.free_all_blocks()
        self.ILLUMINATION_gpu = cp.asarray(self.ILLUMINATION).astype(cp.complex64)
        self.phase_masks = cp.zeros( (self.maskcount, self.SLM_N, self.SLM_N) , cp.complex64) #Store the masks - Cropped masks in case we extended SLM area
        self.fidelity = cp.zeros(self.maskcount, cp.float64) #Keep track of the fidelity during GS running
        self.efficiency = cp.zeros(self.maskcount, cp.float64) #How much power has been scattered # It should need to be on the GPU
        self.overlap = cp.zeros(self.maskcount, cp.float64) #Equivalent to fidelity but before power normalization
        self.iteration_convergence = np.zeros(self.maskcount, int32)
        #Do not bother allocating memoery for this is you did not downscale
        if downscale > 1:  
            self.phase_masks_rescaled = cp.zeros((self.maskcount, self.masksize , self.masksize), cp.complex64)
        else:
            self.phase_masks_rescaled = None
        
        print('Enviroment complete')
    
    def __repr__(self):
        specs = {}
        for key,item in self.__dict__.items():
            if key == '_lastAtribute':
                break
            specs[key] = item
        message = "GS initialization parameters:" + str(specs)
        return(message)     
        
    def setNumberOfMasks(self, count):
        self.maskcount = count
    
    #Input is an array with each row is a set of coef representing a fields
    def calcMasksFromCoefs_gpu(self, inputcoefs, filter = False, rescale = True, printting = False):
        
        input_coefs_gpu = cp.asarray(inputcoefs).astype(cp.complex64)
        targetFields = cp.matmul(input_coefs_gpu, self.Modes1D_bases) #Reconstruction of the fields as 1D vectors
        self.calMasksFromField_gpu(targetFields, rescale = rescale, printting = printting)
        
        return 1
    
    #Input field already in the gpu
    def calMasksFromField_gpu(self, inputfields, filter = False, rescale = True, printting = False):
        maskcounter = 0
        
        fields_dimension = inputfields.shape
        if len(fields_dimension) == 2:
            num_target_fields = fields_dimension[0]
        else:
            num_target_fields = 1
            inputfields = inputfields[None,...]
            
        maskcount = self.maskcount

        if num_target_fields > maskcount:
            print(num_target_fields,' target fields has been provided but only there is memory avaliable for', maskcount)
            num_masks = maskcount
        else:
            num_masks = num_target_fields
        
        Max_iterations = self.MAX_Iterations
        ModesH1D = self.Modes1D_bases_conj
        Modes1D = self.Modes1D_bases
        ILLUMINATION = self.ILLUMINATION_gpu
        TargetFields_gpu = inputfields # In 1D arrays
        N = self.SLM_PLANE_N
        downscale = self.downscale > 1
        
        #If we increased SLM area to ingrese k-space resolution
        crop = self.SLM_N // 2 # Real dimension (well... including downsample, this is fixed later by rescaling)
        c = self.SLM_center
        #Rescaling
        c2 = self.masksize//2 # Real dimensions
  
        #Gaussian filter check
        if filter == None:
            filter_ON = False
        else:
            filter_ON = True
            filter_mod_gpu = self.gaussianFilter_gpu
        #filter
        R0_gpu = self.R0_gpu
                
        #Loop through each target field
        for j in range (num_masks): 
            Target_Mode_1D = TargetFields_gpu[j]
            Target_Mode_1D = (Target_Mode_1D / cp.sqrt(cp.sum(cp.absolute(Target_Mode_1D)**2))) # Normalize in case they were not 
            Target_Mode = cp.reshape(Target_Mode_1D, (N,N)) # I need to do the 2D FFT
            SLM_Target_Mode = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(Target_Mode)))
            SLM_Target_Mode = (SLM_Target_Mode/ cp.sqrt(cp.sum(cp.absolute(SLM_Target_Mode)**2)))# Target in SLM plane
            ############################################
            A = SLM_Target_Mode
            B = ILLUMINATION # Constant
            overlap_coefs = cp.zeros(self.modescount, cp.complex64) #reset every time 
            w = 1 #scatter power weight
            #Modified GS  
            for i in range (Max_iterations):
                C = cp.multiply(cp.absolute(B), cp.exp(1j*cp.angle(A))) #SLM plane
                D = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(C))) #Fiber plane
                D = D / cp.sqrt(cp.sum(cp.absolute(D)**2)) #Fiber plane normalized to 1
                
                #Goal checking at the Fiber plane --> Decompose current field, reconstruct and overlap with the target
                D_1D = cp.reshape(D,-1).astype(cp.complex64) #Reshape the current field at the fiber faced
                overlap_coefs = cp.matmul(D_1D, ModesH1D) #Modal decomposition of the field - using transpose Conj of the modes
                
                recounstruction = cp.matmul(overlap_coefs, Modes1D)
                recounstruction = cp.divide(recounstruction,cp.sqrt(cp.sum(cp.absolute(recounstruction)**2)))#norm
                self.fidelity[j] = cp.sum(cp.absolute(cp.multiply(recounstruction,cp.conj(Target_Mode_1D))))#overlap
                #print(i,self.fidelity[j])
                if self.fidelity[j] >= self.goal_fidelity:
                    self.efficiency[j] = w # Mask efficiency respect total power provided into the slm
                    self.overlap[j] = cp.sum(abs(overlap_coefs)**2) #Mask efficiency respect the target field
                    self.iteration_convergence[j] = i
                    if printting == True:
                        print('Mask', j,'Done - ','Iteration ',i)
                        show()
                    break
                elif i == Max_iterations-1:
                    self.efficiency[j] = w # Mask efficiency respect total power provided into the slm
                    self.overlap[j] = cp.sum(abs(overlap_coefs)**2) #Mask efficiency respect the target field
                    self.iteration_convergence[j] = i
                    if printting == True:
                        print('Mask', j,'crased - ','Iteration ',i)
                        show()
                
                #Control the ampount of power delivered to the fiber or to the clading:
                #w parameter controls how much light goes to the core and how much goes to the clading
                #Total power always must be 1
                
                scatter = ( D * (cp.negative(R0_gpu) + 1) ) #This gets some trash light pattern outside the core
                scatter_p = cp.sqrt(cp.sum(cp.absolute(scatter)**2)) #power of the scatter field
                scatter = (scatter / scatter_p) #Normalize scattered power to unitary power
                scatter2 = cp.sqrt(1-w) * scatter # decrease w throwing more light outside
                
                if filter_ON == True:
                    scatter = filter_mod_gpu * scatter #Apodization of the scatter field, dont generate super crazy high freq.
                    scatter_p = cp.sqrt(cp.sum(cp.absolute(scatter)**2)) #Normalize it again to power = 1
                    scatter = scatter/scatter_p
                    scatter2 = cp.sqrt(1-w) * scatter

                E_p = cp.sqrt(cp.sum(cp.absolute((Target_Mode*R0_gpu)**2))) #This is what I want where exactly I want it (inside core)
                E1 = Target_Mode * R0_gpu / E_p
                E2 = E1 * w #Scaled decrease the power there
                E = E2 + scatter2 #Create the whole field (Core + cladding with power = 1)
                A = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E))) #Back to SLM plane
                
                w = (w*0.99) #This is the amount of power we get rid at every iteration

            computed_GS = cp.multiply(cp.absolute(ILLUMINATION) , cp.exp(1j*cp.angle(C))) #INPUT ILLUMINATION + computed mask phase 
            self.phase_masks[maskcounter] = (computed_GS[c-crop:c+crop,c-crop:c+crop]) #Phase_Mask should be  already cropped to the desired size
            maskcounter+=1   
        
        if  rescale == True and downscale == True:
            FFT_phase_masks = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.phase_masks)))
            self.phase_masks_rescaled[:,c2-crop:c2+crop,c2-crop:c2+crop] = FFT_phase_masks #Padding
            IFFT_phase_masks_rescaled =  cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(self.phase_masks_rescaled)))
            #I can skip the normalization since I will be just taking the phase masks
            #IFF_Field_Intensity = cp.sqrt(cp.sum(cp.absolute(IFFT_phase_masks_rescaled)**2, (1,2))) 
            #IFFT_phase_masks_rescaled = IFFT_phase_masks_rescaled / IFF_Field_Intensity[:,None,None] # Target in SLM plane
            self.phase_masks_rescaled = IFFT_phase_masks_rescaled 
            
    
    def calcFilterMask_gpu(self, croppingFactor = 4, selectivity = 1):
        samples = self.SLM_PLANE_N
        RR = sqrt(self.Xk**2 + self.Yk**2)
        R0 = zeros((samples,samples))
        RR_gpu = cp.sqrt(self.Xk_gpu**2 + self.Yk_gpu**2)
        self.R0_gpu = cp.zeros((samples,samples))
        crop = croppingFactor * self.r_mmf
        print('Forbiden area r <',crop,'um')
        self.R0_gpu[RR_gpu < crop] = 1  
        
        def calGaussianFilter(selectivity = selectivity):
            fil_mod = zeros(R0.shape)

            fil_selectivity = selectivity
            wfil = self.Xk.shape[0] // fil_selectivity
            fil = exp(-((RR-crop ) /wfil)**2)
            fil_mod = (fil*(negative(R0)+1))
            fil_mod = fil_mod / fil_mod.max()
            
            return(fil_mod)
        
        self.gaussianFilter_gpu = cp.asarray(calGaussianFilter())
        
        
    def calculateLGmodes(self):
        self.mempool = mg.mgcl.cp.get_default_memory_pool()
        LGbases = mg.LGmodes(self.MMF_mfd, self.MaxModegroups, self.SLM_PLANE_N, self.pixel_mmf, generateModes = True, wholeSet = True, engine='GPU', multicore=True) #object
        self.LGmodes = LGbases.LGmodesArray__
        self.LGindex = LGbases.index   
        del LGbases
        self.mempool.free_all_blocks()    

        
    @staticmethod    
    def define_slm_plane (pixelcount, pixel_size, downscale, upsampling, printting = True):
        """
        IN:
            pixelcount (int): number of pixel (only working for square masks)
            pixel_size (float) : pixel size in um
            downscale (int) : factor to downscale the slm, it is like increase the pixel size keeping real total dimensions
            upsampling (_type_): In case we will be padding the SLM space to increase resolution at the Fourier space
        OUT:          
        """
        
        ext_samples = upsampling * pixelcount
        total_samples = (pixelcount + ext_samples) // downscale #Total number of sample slm + padding
        slm_samples = pixelcount // downscale #Total number of samples avaliable inside the slm area
        slm_size = (pixel_size * downscale) * slm_samples
        artif_pixel = (pixel_size * downscale)
        x_slm = arange(-slm_samples//2,slm_samples//2,1) * artif_pixel
        X_slm, Y_slm = meshgrid(x_slm,x_slm)
        R_SLM = sqrt(X_slm**2 + Y_slm**2)
        SLM_PLANE = zeros((total_samples,total_samples))
        total_length = total_samples * artif_pixel
        
        if printting == True:
            print("SLM size =",slm_size,'um')
            print("Artificial SLM size =",total_length,'um')
            fs = total_samples / total_length
            print("Spatial resolution",fs,'um-1')
            print('Image plane samples',total_samples,'x',total_samples)
            print('Useful SLM samples',slm_samples,'x',slm_samples)
            
        return SLM_PLANE, R_SLM, total_samples, total_length, slm_samples
    
    @staticmethod
    def define_IlluminationBeam(wlength, R, AREA, f_in, d_smf, NA_smf, slm_W0 = None, printting = True):
        
        if slm_W0 != None:
            w0_slm = slm_W0
        else:
            V_smf = (2*pi*NA_smf * d_smf/2) / wlength
            w0_smf = d_smf/2* (0.65 + (1.619/V_smf**1.5) + (2.879/V_smf**6))#2*wlength / (pi*NA_smf)
            zr = pi * w0_smf**2 / wlength
            w0_slm = w0_smf * sqrt(1 + (f_in/zr)**2)
            
        slm_ill = exp(-(R/w0_slm)**2) #exp(-(R/w0_slm)**2)
        slm_ill = slm_ill / sqrt(sum(abs(slm_ill)**2))
        c = AREA.shape[0]//2
        #slm_c = num_px_slm //2
        dd = slm_ill.shape[0]//2
        AREA[c-dd:c+dd,c-dd:c+dd] = slm_ill
        INPUT_BEAM = AREA
        
        if printting == True:
            print('Beam waist in the SLM -->',w0_slm *1e-3,' mm' )

        return INPUT_BEAM 
        
    @staticmethod
    def define_Kspace(wlength, N, L, f, printting = True ):
        #K space - fiber faced
        k0 = 2 *np.pi / wlength
        samples = N# SLM number of pixels / downscale
        px_kspace = wlength * f / L #um
        x = arange(-samples//2,samples//2,1) * px_kspace
        Xk,Yk = np.meshgrid(x,x) 
        #Fourier Plane Data
        if printting == True:
            print('Pixel size FFT',px_kspace,'um')
            print('MAX spectral distance +-', px_kspace*samples//2, 'um') 
        
        return px_kspace, Xk, Yk
            
        
if __name__ == '__main__':
    
    masks = SetupGenericMaskGSenvioment(wavelength = 1.3, fin = 25e3, fout = 8e3, masksize = 960, slm_pixel_size = 9.2,
                          MMF_mfd = 12.2, MMFdiameter = 62.5, MaxModegroups = 22, maskscount = 1, 
                          goal_fidelity = 0.99, downscale = 2, upsample = 0, MAX_iterations = 300)
    
    #gaus_filter = cp.asnumpy((cp.negative(masks.R0_gpu) + 1) * masks.gaussianFilter_gpu )#* masks.gaussianFilter_gpu)  
    #imshow(gaus_filter),colorbar()
    #show()
    #import some target coefs
    import h5py
    import hdf5storage  
    
    path_to_mat_file = 'C://LAB//Coding//Python//Data//T_matrices//'
    mat_file_name = 'MTM_f8mm_17_22MG_IL66_MDL-75_458modes_z=30um_looks good.mat'              
    file_to_open = path_to_mat_file + mat_file_name
    data = hdf5storage.loadmat(file_to_open) # it loads data as a dictionary
    U = data['Uw'][0,...]
    print('Transmission Matrix dim: ', U.shape)
    
    path_to_mat_file = 'C:\\LAB\\Coding\\Python\\Data\\pattern_coefs\\modal_decompositions\\'
    mat_file_name = 'pattern_coefs_smilySad_nmax_22.mat'         
    file_to_open = path_to_mat_file + mat_file_name
    data = hdf5storage.loadmat(file_to_open) # it loads data as a dictionary
    TargetCoefs = data['coefs'] #Reconstruction of this is my Target at the camera
    #H and V come interleaved
    
    InputCoefs = matmul(TargetCoefs,(transpose(U))) #Backpropgation of the Target towars the input fiber -- NO PANIC - CONJUGATE and TRY AGAIN
    #InputCoefs = TargetCoefs
    H_input_coefs = InputCoefs[0::2] #indexing here is odd elements to H
    V_input_coefs = InputCoefs[1::2]

    Target = empty((2,H_input_coefs.shape[0]),np.complex64)    
    Target[0,:] = H_input_coefs
    Target[1,:] = V_input_coefs
    print(Target.shape)
    
    t = mg.mgcl.times
    t.tic()
    masks.calcMasksFromCoefs_gpu(Target, printting = True)
    t.toc()
    
    ##Change coefs and re do to check the time
    ##InputCoefs = TargetCoefs
    #InputCoefs = matmul(TargetCoefs,(transpose(U))) #Backpropgation of the Target towars the input fiber -- NO PANIC - CONJUGATE and TRY AGAIN
    #H_input_coefs = InputCoefs[0::2] #indexing here is odd elements to H
    #V_input_coefs = InputCoefs[1::2]

    #Target = empty((2,H_input_coefs.shape[0]),np.complex64)    
    #Target[0,:] = H_input_coefs
    #Target[1,:] = V_input_coefs
    #t.tic()
    #masks.calcMasksFromCoefs_gpu(Target, printting = True)
    #t.toc()
    
    #ph_masks = cp.asnumpy(masks.phase_masks)
    #ph_masks_rescaled = cp.asnumpy(masks.phase_masks_rescaled)
    #f,ax = subplots(1,2, figsize=(10,10))
    #ax[0].imshow(angle(ph_masks[0]))
    #ax[1].imshow(angle(ph_masks_rescaled[0]))
    #print("fidelity:", masks.fidelity, 'efficiency:', masks.efficiency)
    #print(sum(abs(ph_masks[0])**2) , sum(abs(ph_masks_rescaled[0])**2) )
    #print("MAX:",(abs(ph_masks[0])**2).max() , (abs(ph_masks_rescaled[0])**2).max() )

    #show()

        
        