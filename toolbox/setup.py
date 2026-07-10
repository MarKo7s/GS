from numpy import arange, meshgrid, zeros
import numpy as np
import cupy as cp

from fibremodes.analytical.LG.ModesGen import LGmodes
from fibremodes.analytical.HG.fibremodes import makeHGModes

SUPPORTED_MODES_BASIS = ['LG', 'HG', 'custom']
COVERGENCE_METRIC = ['overlap', 'crosstalk']

#TODO: Add custom mode basis support, in this scenario the user needs to handle the mode class
class optical_setup:
    """Optical System Configuration Class.

    Holds wavelength, focal lengths, SLM pixel specs, MMF parameters, and SMF
    illumination inputs. Passed into slm_plane, k_space_plane, illumination,
    filters, and modes.
    
    Input arguments:

    wavelength (float): wavelength in um
    fin (float): focal length of the input lens in mm (single mode fibre collimation beam into the SLM)
    fout (float): focal length of the output lens in mm (output: injection to the MMF)
    masksize (int): size of the mask in pixels
    slm_pixel_size (float): size of the SLM pixel in um
    MMF_mfd (float): mode field diameter of the MMF in um
    MMF_diameter (float): diameter of the MMF core in um
    MMF_mode_groups (int): number of mode groups of the MMF
    mode_basis (str): basis of the modes. Supported: 'LG', 'HG', 'custom'
    smf_in_NA (float): numerical aperture of the SMF input (used for SLM illumination calculation)
    smf_in_diameter (float): diameter of the SMF input in um (used for SLM illumination calculation)
    W0_slm (narray or None): if illumination profile is provided, it will be used instead of the smf specs provided.
    """

    def __init__(self, wavelength, fin, fout, masksize, slm_pixel_size, MMF_mfd, MMF_diameter,
                MMF_mode_groups, mode_basis='LG', smf_in_NA=0.1, smf_in_diameter=9, W0_slm=None):
        self.wavelength = wavelength
        self.fin = fin
        self.fout = fout
        self.masksize = masksize
        self.slm_pixel_size = slm_pixel_size
        self.MMF_mfd = MMF_mfd
        self.MMF_diameter = MMF_diameter
        self.MMF_mode_groups = MMF_mode_groups
        self.mode_basis = mode_basis
        if mode_basis not in SUPPORTED_MODES_BASIS:
            raise ValueError(f"Unsupported mode basis: {mode_basis}")
        #This is use for the illumination beam definition
        self.smf_in_NA = smf_in_NA
        self.smf_in_diameter = smf_in_diameter
        self.W0_slm = W0_slm


class GS_mask_setup:
    """Scalar configuration for GS mask computation (no arrays).

    Controls SLM downsampling/upsampling and iteration limit. Passed into
    slm_plane and GS engines.
    """

    def __init__(self, num_masks = None, convergence_metric = 'overlap', goal_value = 0.99, convergence_check_interval = 1,
                    downscale=1, upsample=0, MAX_iterations=300, filtering:bool = False,
                    filtering_factor=2, filtering_mask_radius_factor=4):
     
        self.num_masks = num_masks
        self.convergence_metric = convergence_metric
        if convergence_metric not in COVERGENCE_METRIC:
            raise ValueError(f"Unsupported convergence metric: {convergence_metric}")
        self.goal_value = goal_value
        self.convergence_check_interval = convergence_check_interval
        self.downscale = downscale
        self.upsample = upsample
        self.MAX_iterations = MAX_iterations
        self.filtering = filtering
        self.filtering_factor = filtering_factor # Streght to supress high frequencies
        self.filtering_mask_radius_factor = filtering_mask_radius_factor # This is a safety guard in function of the fibre core radius

#! This classes take optical setup and masksetup as input, in theory user does not need to
#! deal with them and the parent classes will handle the setup and the computation.
class slm_plane:
    """SLM geometry at the mask plane (CPU numpy arrays).

    Computes the padded/downsampled SLM grid and polar radius R.
    Outputs: SLM_PLANE, R, SLM_PLANE_N, SLM_PLANE_L, SLM_N.
    Rebuild a new instance when downscale or upsample changes.
    """

    def __init__(self, opticalsetup: optical_setup, masksetup: GS_mask_setup, printting=True):
        pixelcount = opticalsetup.masksize
        pixel_size = opticalsetup.slm_pixel_size
        downscale = masksetup.downscale
        upsampling = masksetup.upsample
        self.SLM_PLANE, self.R, self.SLM_PLANE_N, self.SLM_PLANE_L, self.SLM_N = self.define_slm_plane(pixelcount, pixel_size, downscale, upsampling, printting)

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
        R_SLM = np.sqrt(X_slm**2 + Y_slm**2)
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


class k_space_plane:
    """Fibre-face (Fourier) plane coordinates in k-space (CPU numpy arrays).

    Depends on slm_plane grid size and optical_setup.fout.
    Outputs: px_kspace, Xk, Yk. Rebuild when slm_plane changes.
    """

    def __init__(self, opticalsetup: optical_setup, slmplane: slm_plane, printting=True):
        wlength = opticalsetup.wavelength
        N = slmplane.SLM_PLANE_N
        L = slmplane.SLM_PLANE_L
        f = opticalsetup.fout
        self.px_kspace, self.Xk, self.Yk, self.max_spectral_distance_um = self.define_Kspace(wlength, N, L, f, printting)

    @staticmethod
    def define_Kspace(wlength, N, L, f, printting = True ):
        #K space - fiber faced
        k0 = 2 * np.pi / wlength
        samples = N# SLM number of pixels / downscale
        px_kspace = wlength * f / L #um
        x = np.arange(-samples//2,samples//2,1) * px_kspace
        Xk,Yk = np.meshgrid(x,x) 
        #Fourier Plane Data
        if printting == True:
            print('Pixel size FFT',px_kspace,'um')
            print('MAX spectral distance +-', px_kspace*samples//2, 'um') 
        
        return px_kspace, Xk, Yk,  px_kspace*samples//2


class illumination:
    """Gaussian illumination beam on the SLM plane (CPU numpy array).

    Embeds the normalized beam into SLM_PLANE geometry from slm_plane.
    Output: ILLUMINATION — upload to gpu_memory before the GS loop.
    """

    def __init__(self, opticalsetup: optical_setup, slmplane: slm_plane, printting=True):
        wlength = opticalsetup.wavelength
        R = slmplane.R
        AREA = slmplane.SLM_PLANE
        f_in = opticalsetup.fin
        d_smf = opticalsetup.smf_in_diameter
        NA_smf = opticalsetup.smf_in_NA
        slm_W0 = opticalsetup.W0_slm
        self.ILLUMINATION = self.define_IlluminationBeam(wlength, R, AREA, f_in, d_smf, NA_smf, slm_W0, printting)

    @staticmethod
    def define_IlluminationBeam(wlength, R, AREA, f_in, d_smf, NA_smf, slm_W0 = None, printting = True):
        
        if slm_W0 != None:
            w0_slm = slm_W0
        else:
            V_smf = (2*np.pi*NA_smf * d_smf/2) / wlength
            w0_smf = d_smf/2* (0.65 + (1.619/V_smf**1.5) + (2.879/V_smf**6))#2*wlength / (pi*NA_smf)
            zr = np.pi * w0_smf**2 / wlength
            w0_slm = w0_smf * np.sqrt(1 + (f_in/zr)**2)
            
        slm_ill = np.exp(-(R/w0_slm)**2) #exp(-(R/w0_slm)**2)
        slm_ill = slm_ill / np.sqrt(np.sum(np.abs(slm_ill)**2)) #Normalize the illumination beam
        c = AREA.shape[0]//2
        #slm_c = num_px_slm //2
        dd = slm_ill.shape[0]//2
        AREA[c-dd:c+dd,c-dd:c+dd] = slm_ill
        INPUT_BEAM = AREA
        
        if printting == True:
            print('Beam waist in the SLM -->',w0_slm *1e-3,' mm' )

        return INPUT_BEAM 


class modes:
    """LG mode basis at the fibre plane (CPU numpy arrays). Not yet implemented.

    Will provide LGmodes (M, N, N), index, Modes1D_bases, and Modes1D_bases_conj.
    LGmodes stays on CPU for mode-mask engine (one slice uploaded per job);
    1D bases are uploaded to gpu_memory for decomposition / crosstalk checks.
    """

    def __init__(self, opticalsetup: optical_setup, slmplane: slm_plane, kspaceplane: k_space_plane):
        basis = opticalsetup.mode_basis
        if basis == 'LG':
            LGmodesObject = LGmodes(opticalsetup.MMF_mfd, opticalsetup.MMF_mode_groups, slmplane.SLM_PLANE_N, kspaceplane.px_kspace, 
            generateModes = True, wholeSet = True, engine='GPU', multicore=True) #object]
            self.index= LGmodesObject.index
            self.modes_array_all = LGmodesObject.LGmodesArray__ # Full modes piramid
            self.modes_array = LGmodesObject.LGmodesArray  # Half modes piramid

            #delete the mode object to free gpu memory
            del LGmodesObject
            cp.get_default_memory_pool().free_all_blocks()

        elif basis == 'HG':
            HGmodes = makeHGModes(slmplane.SLM_PLANE_N, opticalsetup.MMF_mfd, opticalsetup.MMF_mode_groups)
            self.index= None
            self.modes_array_all = HGmodes
            self.modes_array = None

        elif basis == 'custom':
            self.index = None
            self.modes_array_all = None
            self.modes_array = None
            print('Custom mode basis selected, the user needs to handle the mode class. Use the set_mode_basis method to set the mode basis.')

        else:
            raise ValueError(f"Unsupported mode basis: {basis}")

    def set_mode_basis(self, modes):
        self.modes_array_all = modes


class filters:
    """Spatial core mask and Gaussian scatter filter (CPU numpy arrays).

    Depends on slm_plane, k_space_plane, and optical_setup.MMF_diameter.
    Outputs: R0 (core mask), gaussianFilter — upload both to gpu_memory for GS.
    Rebuild a new instance when k-space grid or fibre/filter params change.
    """

    def __init__(self, slmplane: slm_plane, opticalsetup: optical_setup, kspaceplane: k_space_plane,
                 filtering_mask_radius_factor=4, filtering_factor=2):
        samples = slmplane.SLM_PLANE_N
        RR = np.sqrt(kspaceplane.Xk**2 + kspaceplane.Yk**2)
        R0 = zeros((samples,samples))
        self.R0 = zeros((samples,samples))
        self.R_core = np.zeros_like(R0)
        r_filter = filtering_mask_radius_factor * opticalsetup.MMF_diameter/2
        r_core = opticalsetup.MMF_diameter/2
        print('Forbiden area r <',r_filter,'um')
        self.R0[RR < r_filter] = 1  
        self.R_core[RR < r_core] = 1
        self.kspaceplane = kspaceplane

        self.gaussianFilter = self._calGaussianFilter(self.kspaceplane.Xk, RR, R0, r_filter, filtering_factor)
        

    @staticmethod
    def _calGaussianFilter(X, RR, R0, crop, selectivity):
        fil_mod = zeros(R0.shape)

        fil_selectivity = selectivity
        wfil = X.shape[0] // fil_selectivity
        fil = np.exp(-((RR-crop ) /wfil)**2)
        fil_mod = (fil*(np.negative(R0)+1))
        fil_mod = fil_mod / fil_mod.max()
        
        return(fil_mod)


#! Containers of arrays where operations are performed.
class cpu_memory:
    """Lightweight CPU array catalog. Stores references by name (no copy).

    Register arrays from domain objects after build/rebuild, then upload to gpu_memory.
    Typical names: ILLUMINATION, R0, gaussianFilter, LGmodes, Modes1D_bases, Modes1D_bases_conj.
    LGmodes (M, N, N) stays on CPU; per-mask slices are uploaded ad hoc by the engine.
    """

    def __init__(self):
        self._arrays = {}

    def register(self, name, arr):
        self._arrays[name] = arr
        setattr(self, name, arr)

    def get(self, name):
        return self._arrays[name]

    def clear(self, *names):
        for name in names:
            self._arrays.pop(name, None)
            if hasattr(self, name):
                delattr(self, name)

    def clear_all(self):
        for name in list(self._arrays):
            if hasattr(self, name):
                delattr(self, name)
        self._arrays.clear()

    def names(self):
        return list(self._arrays)

    def __contains__(self, name):
        return name in self._arrays


class gpu_memory:
    """GPU array registry for GS engines. Device copies and GPU-native buffers.

    Setup arrays (upload from cpu_memory or replace from domain objects):
      ILLUMINATION, R0, gaussianFilter, Modes1D_bases, Modes1D_bases_conj
    Runtime buffers (allocate): phase_masks, fidelity, efficiency, worker scratch

    Sync workflow:
      1. Build/rebuild domain objects
      2. cpu.register(...) for each CPU array
      3. gpu.upload(cpu, ...) or gpu.replace(...) after partial rebuild
      4. On grid-size change: gpu.clear_all() then full re-upload
    """

    def __init__(self):
        self._arrays = {}

    def register(self, name, gpu_array):
        self.clear(name)
        self._arrays[name] = gpu_array
        setattr(self, name, gpu_array)

    def replace(self, name, cpu_array, dtype=None):
        self.clear(name)
        gpu_array = cp.asarray(cpu_array)
        if dtype is not None:
            gpu_array = gpu_array.astype(dtype)
        self._arrays[name] = gpu_array
        setattr(self, name, gpu_array)

    def upload(self, cpu, *names, dtype=None):
        for name in names:
            if dtype is not None:
                dtype = cpu.get(name).dtype
            else:
                dtype = None
            self.replace(name, cpu.get(name), dtype=dtype)

    def allocate(self, name, shape, dtype=cp.complex64):
        self.clear(name)
        gpu_array = cp.zeros(shape, dtype=dtype)
        self._arrays[name] = gpu_array
        setattr(self, name, gpu_array)

    def get(self, name):
        return self._arrays[name]

    def clear(self, *names):
        for name in names:
            self._arrays.pop(name, None)
            if hasattr(self, name):
                delattr(self, name)

    def clear_all(self):
        for name in list(self._arrays):
            if hasattr(self, name):
                delattr(self, name)
        self._arrays.clear()

    def free_pool(self):
        cp.get_default_memory_pool().free_all_blocks()
