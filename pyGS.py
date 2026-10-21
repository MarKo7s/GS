try:
    from .toolbox.setup import *
    from .toolbox.gs_operators import *
    from .utils.library import save_library
    from .logger import log_info, log_mask_result, log_run_preamble
except ImportError:
    from toolbox.setup import *
    from toolbox.gs_operators import *
    from utils.library import save_library
    from logger import log_info, log_mask_result, log_run_preamble

import numpy as np
import cupy as cp
import time

from pylab import *
from scipy.ndimage import zoom

GS_ENGINE_MODE = ['fields', 'fibremodes']
OPTIONS = {'fields': {'number_masks': None},
            'fibremodes': {'mode_range': None}}

#Typically fields use overlap metrid and modes use crosstalk metric. This could or could not be a open parameter

def _as_float(value):
    return float(cp.asnumpy(value))


def _left_edge_radius_px(offsets):
    """Radius of the left edge, already stored as center minus the row index."""
    if offsets is None:
        return None
    offsets = np.asarray(offsets).reshape(-1)
    if offsets.size == 0:
        return None
    return float(offsets[0])


class GS:
    def __init__(self, opticalsetup: optical_setup, masksetup: GS_mask_setup, engine_mode: str = 'fields', **options ):
        if engine_mode not in GS_ENGINE_MODE:
            raise ValueError(f"Unsupported engine mode: {engine_mode}")
        self.engine_mode = engine_mode
        print('ENGINE MODE SELECTED:', engine_mode)

        self.opticalsetup = opticalsetup
        self.masksetup = masksetup

        #! Calculate the SLM plane, K-space plane, Illumination, Modes and Filters.
        #All this is cpu bases for now
        #Calculate the SLM plane. This is the area of the SLM including the artificial extension.
        self.slmplane = slm_plane(opticalsetup, masksetup)
        #Calculate the K-space plane (Fourier space). This correspondt to the fibre input plane
        self.kspaceplane = k_space_plane(opticalsetup, self.slmplane)
        #Calculate the Illumination on to the slm.
        self.illumination = illumination(opticalsetup, self.slmplane)
        
        # Calculate the Filters. This need to be re-calculate if slmplane or kspaceplane changes.
        self.filters = filters(self.slmplane, opticalsetup, self.kspaceplane,
                         filtering_factor = self.masksetup.filtering_factor,
                         filtering_mask_radius_factor = self.masksetup.filtering_mask_radius_factor)
        #Calculate the Modes. This need to be re-calculate if slmplane or kspaceplane changes.
        self.modes = modes(opticalsetup, self.slmplane, self.kspaceplane)

        _selected_basis = self.opticalsetup.mode_basis #what user provided

        self._options = options
        self._engine_mode = engine_mode
        self._GS_ready = False

        if _selected_basis == 'custom':
            print('Custom mode basis selected, to continue input the basis manually using set_mode_basis method to finalize the setup')
            return
    
        self._GS_setup()
        self._feasibility_check_done = self.feasibility_setup_check()

    def set_mode_basis(self, modes: np.ndarray):
        self.modes.set_mode_basis(modes)
        self._GS_setup()
        self._feasibility_check_done = self.feasibility_setup_check()

    def feasibility_setup_check(self, thr_pow: float = 0.01):
        """ Check setup can generate targets"""
        # If NA SMF and NA MMF is provided this equivalent to: NA_SMF / NA_MMF >  f_out / f_in (This does not account for power guide by cladding)
        # In general we can estimate it checking the illumination power distribution and the mode basis
        # The illumination needs to extended beyond the projected MMF modes on to the SLM plane.
        if self.modes._modes_SLM is None:
            self.modes.modes_at_SLM_plane()
            modes_SLM = self.modes.modes_SLM
        else:
            modes_SLM = self.modes._modes_SLM

        incoherent_sum_modes_SLM = sum(abs(modes_SLM[:,...]), axis = 0)
        incoherent_sum_modes_SLM /= incoherent_sum_modes_SLM.max()
        illumination_norm = self.illumination.ILLUMINATION / abs(self.illumination.ILLUMINATION).max() #norm to 1 to compare

        #Check for illumination power laying outside the pupil
        pupil_radius = self.opticalsetup.pupil_um / 2
        illumination_outside_pupil = self.illumination.ILLUMINATION.copy()
        illumination_outside_pupil[self.slmplane.R <= pupil_radius] = 0
        illumination_outside_pupil_power = sum(abs(illumination_outside_pupil)**2) #this is not the biggest problem, could poroduce ringing artifacts
        self._illumination_outside_pupil_power = illumination_outside_pupil_power

        #Do tails still carry power higher than modes power distrution does?
        cc = illumination_norm.shape[0]//2
        ill_cut = illumination_norm[illumination_norm.shape[0]//2, :]
        modes_cut = incoherent_sum_modes_SLM[incoherent_sum_modes_SLM.shape[0]//2, :]

        thr = sqrt(thr_pow)
        ill_cut_pos0 =  where(ill_cut > thr)[0] #Get the position of the first pixel that is above the threshold
        modes_cut_pos0 = where(modes_cut > thr)[0] #Get the position of the first pixel that is above the threshold

        self._ill_coverage_rad_px_approx = cc - ill_cut_pos0
        self._modes_coverage_rad_px_approx = cc - modes_cut_pos0

        illumination_min = ill_cut.min()
        modes_min = modes_cut.min()
        check_tails_power = illumination_min > modes_min #This checks illumination provides support to the target modes
        self._illumination_support_check = check_tails_power

        #check that modes are inside aperture as well
        incoherent_sum_modes_SLM_unitary = incoherent_sum_modes_SLM / sqrt(sum(abs(incoherent_sum_modes_SLM)**2))
        incoherent_sum_moodes_outside_aperture = incoherent_sum_modes_SLM_unitary
        incoherent_sum_moodes_outside_aperture[self.slmplane.R <= pupil_radius] = 0
        modes_outside_aperture_power = sum(abs(incoherent_sum_moodes_outside_aperture)**2)
        self._mode_basis_power_outside_pupil = modes_outside_aperture_power

        print('--------------------------------')
        print('Feasibility setup integrity check:')
        illumination_pow_check = illumination_outside_pupil_power < thr_pow
        self._illumination_pow_check_passed = illumination_pow_check
        modes_pow_check = modes_outside_aperture_power < thr_pow
        self._modes_pow_check_passed = modes_pow_check
        print('Illumination outside pupil power:', f'{illumination_outside_pupil_power * 100:.2f}' , '% -->', illumination_pow_check)
        print('Modes outside aperture power:', f'{modes_outside_aperture_power * 100:.2f}' , '% -->', modes_pow_check)
        print('Illumination overlap support:', check_tails_power)
        self._illumination_overlap_support_check_passed = check_tails_power
        self._setup_feasible = illumination_pow_check and modes_pow_check and check_tails_power
        print('Setup feasible? ', self._setup_feasible)

        #Calculate NA_MMF max
        self._NA_MMF_system_max = self.opticalsetup.smf_in_NA * self.opticalsetup.fin / self.opticalsetup.fout
        print('This system support up to NA_MMF:', f'{self._NA_MMF_system_max:.2f}')
        print('--------------------------------')

        return True



    def _GS_setup(self):
        """ It will allocate all the memory needed for the GS engine. """
        #! Check consistency of the setup:
        #* Downscaling factor reduce the high frequencies, this is good but we need to still be able to 
        #* scatter power outside the core (R0). Do not allow to downscale such as the max spectral distance is greater
        #* than the forbiden area radius.
        _forbiden_area_radius_um = self.opticalsetup.MMF_diameter/2 * self.masksetup.filtering_mask_radius_factor
        if _forbiden_area_radius_um > self.kspaceplane.max_spectral_distance_um:
            raise ValueError(f"Forbiden area radius {_forbiden_area_radius_um} is greater than the max spectral distance {self.kspaceplane.max_spectral_distance_um}")
            
        #! Allocate the memory for the GS engine.
        self.cpu_memory = cpu_memory()
        self.gpu_memory = gpu_memory()
        #! Register useful already calculate cpu-arrays in the cpu_memory catalog.
        # Register in the cpu and the allocate in the gpu
        #* GS needs always, independed of the target field, illumination, modes in 2D, filters: R0, GaussianFilter
        ILLUMINATION = self.illumination.ILLUMINATION
        R0 = self.filters.R0
        R_CORE = self.filters.R_core
        GAUSSIAN_FILTER = self.filters.gaussianFilter
        #! What to do depending on the engine and selected modes basis
        _selected_mode_basis = self.opticalsetup.mode_basis #what user provided

        if _selected_mode_basis == 'LG':
            if self._engine_mode == 'fields':
                modes_array = self.modes.modes_array_all
            elif self._engine_mode == 'fibremodes':
                modes_array = self.modes.modes_array_all 
        else:
            modes_array = self.modes.modes_array_all
        
        #! Get total number of masks
        if self._engine_mode == 'fields':
            
            numer_masks = self._options.get('number_masks') or 2
            pattern_sequence_scan_array = np.arange(numer_masks)
            modes_array = self.modes.modes_array_all 

            print('Fields engine mode selected: memory allocation for', numer_masks, 'masks')
  
        elif self._engine_mode == 'fibremodes':
            _half_piramide = self.modes.modes_array is not None #check if fibre mode support half piramide
            #Otherwise use the full catalog of modes
            if _half_piramide:
                modes_array = self.modes.modes_array
            else:
                modes_array = self.modes.modes_array_all

            modes_range = self._options.get('mode_range')
            print('MODES RANGE:', modes_range)

            if modes_range is not None:
                #If we are working with half piramid, we need to be careful.
                if _half_piramide:
                    piramid_scheme = self.modes.piramid_scheme
                    print('User requested modes between', modes_range[0], 'and', modes_range[1], '--> total number of masks',  modes_range[1] - modes_range[0] + 1)
                    #First mode asked 
                    first_mode_in_half_piramide = next(k for k, rows in piramid_scheme.items() if modes_range[0] in rows)   
                    last_mode_in_half_piramide = next(k for k, rows in piramid_scheme.items() if modes_range[1] in rows)
                    pattern_sequence_scan_array = np.arange(first_mode_in_half_piramide, last_mode_in_half_piramide + 1) #this are the modes patterns to loop
                    #The real number of mask are the modes in between all those half piramide indices
                    _init = piramid_scheme[first_mode_in_half_piramide][0]
                    _end = piramid_scheme[last_mode_in_half_piramide][-1] #get last if 2 or first if 1
                    numer_masks = int(_end - _init + 1)
                    print('Algorithm will return modes from', _init, 'to', _end, '--> with a total number of masks', numer_masks)
                else:
                    numer_masks = int(modes_range[1] - modes_range[0] + 1)
                    pattern_sequence_scan_array = np.arange(modes_range[0], modes_range[1] + 1)    
            else:
                if _half_piramide:
                    numer_masks = self.modes.total_modes_in_the_basis #the number of total mask matches the full piramid
                    num_modes_in_half_piramide = list(self.modes.piramid_scheme.keys())[-1] + 1
                    pattern_sequence_scan_array = np.arange(num_modes_in_half_piramide) # if we work with half piramide the mask array is not the same.
                else:
                    numer_masks = self.modes.total_modes_in_the_basis
                    pattern_sequence_scan_array = np.arange(numer_masks)

            print('Fibre modes engine mode selected: memory allocation for', numer_masks, 'masks --> number of modes', modes_array.shape[0])
        
            
        self._pattern_sequence_scan_array = pattern_sequence_scan_array
        modes_index = self.modes.index
        #prepare modes as a 2D array (so modes in 1D actually)
        num_modes = modes_array.shape[0]
        modes_array_1D =  np.reshape(modes_array,(modes_array.shape[0],-1)) #use for overlaps, mode decomposition etc

        self.cpu_memory.register('ILLUMINATION', ILLUMINATION)
        self.cpu_memory.register('R0', R0)
        self.cpu_memory.register('R_CORE', R_CORE)
        self.cpu_memory.register('GAUSSIAN_FILTER', GAUSSIAN_FILTER)
        self.cpu_memory.register('MODES_ARRAY_1D', modes_array_1D)
        self.cpu_memory.register('MODES_INDEX', modes_index)

        #! Upload to the gpu
        self.gpu_memory.upload(self.cpu_memory, 'ILLUMINATION', dtype=cp.complex64)
        self.gpu_memory.upload(self.cpu_memory, 'R0', dtype=cp.float32)
        self.gpu_memory.upload(self.cpu_memory, 'R_CORE', dtype=cp.float32)
        self.gpu_memory.upload(self.cpu_memory, 'GAUSSIAN_FILTER', dtype=cp.float32)
        self.gpu_memory.upload(self.cpu_memory, 'MODES_ARRAY_1D', dtype=cp.complex64)
        _modes_array_hermitian = cp.conj(cp.transpose(self.gpu_memory.get('MODES_ARRAY_1D')))
        self.gpu_memory.register('MODES_ARRAY_1D_HERMITIAN', _modes_array_hermitian) #complex conjugate of the modes array
        #! This is used for padding the phase mask to the original size
        self.gpu_memory.allocate('PHASE_MASK_RESCALED', (self.opticalsetup.masksize, self.opticalsetup.masksize), dtype=cp.complex64)

        #! Allocate CPU memeory to dump the masks
        #This is already with the right dimension.
        self.maskcount = numer_masks
        self.phase_masks = np.zeros((numer_masks, self.opticalsetup.masksize, self.opticalsetup.masksize), dtype=np.float32)
        self.cpu_memory.register('PHASE_MASKS', self.phase_masks) #This not need to be in GPU

        self.efficiency = np.zeros(numer_masks, dtype=np.float32)
        self.iteration_convergence = np.zeros(numer_masks, dtype=np.int32)
        self.D_core_coefs = np.zeros((numer_masks, num_modes), dtype=np.complex64)
        self.convergence_history = np.zeros((numer_masks, self.masksetup.MAX_iterations), dtype=np.float32)
        self.efficiency_history = np.zeros((numer_masks, self.masksetup.MAX_iterations), dtype=np.float32)
        # Bits, high to low: gradient fail, SNR fail, overlap fail. 0 passed, 7 all three failed.
        self.mask_passed =  np.zeros(numer_masks, dtype=np.uint8)
        self.basis_coupling_snr = np.zeros(numer_masks, dtype=np.float32)
        self.dumping_field = np.zeros(numer_masks, dtype=np.float32)
        self.core_field = np.zeros(numer_masks, dtype=np.float32)
        self.leaked_power = np.zeros(numer_masks, dtype=np.float32)

        #! Gather the information regarding the resampling of the mask

        self._mask_center_physical = self.opticalsetup.masksize // 2
        self._mask_center = self.slmplane.SLM_PLANE_N // 2
        self._crop_phase_mask = self.slmplane.SLM_N // 2

        self._masks_rescaling_properties = {'physical_center': self._mask_center_physical, 'center': self._mask_center, 'crop': self._crop_phase_mask}
        self._GS_ready = True
    def updateIllumination(self, smf_in_NA=None, smf_in_diameter=None, W0_slm=None, printting=True):
        self.illumination.update(smf_in_NA, smf_in_diameter, W0_slm, printting)
        ILLUMINATION = self.illumination.ILLUMINATION
        #update mem space
        self.gpu_memory.register('ILLUMINATION', ILLUMINATION, dtype=cp.complex64)
        self.cpu_memory.replace('ILLUMINATION', ILLUMINATION)

    def get_illumination(self):
        downscale = self.masksetup.downscale
        if downscale > 1:
            illumination = zoom(self.illumination.ILLUMINATION, zoom = downscale, order=1)
            illumination = illumination / sqrt(sum(abs(illumination)**2))
            return illumination
        else:
            illumination =self.illumination.ILLUMINATION
            illumination = illumination / sqrt(sum(abs(illumination)**2))
            return illumination
    

    def _prepare_run(self, inpucoefs: np.ndarray | None = None):
        _modes = self.gpu_memory.get('MODES_ARRAY_1D')
        _metric = self.masksetup.convergence_metric
        _filtering = self.masksetup.filtering
        _downscaling = True if self.masksetup.downscale > 1 else False
        _goal = self.masksetup.goal_value
        _max_iterations = self.masksetup.MAX_iterations
        _minium_efficiency = self.masksetup.minium_efficiency
        _pattern_sequence_scan = self._pattern_sequence_scan_array
        _N = self.slmplane.SLM_PLANE_N
        maskcount = self.maskcount

        if self.engine_mode == 'fields':
            if inpucoefs is None:
                raise ValueError('Input coefficients are required for fields engine mode')

            input_coefs_gpu = cp.asarray(inpucoefs).astype(cp.complex64)
            _modes = self.gpu_memory.get('MODES_ARRAY_1D')
            targetFields = cp.matmul(input_coefs_gpu, _modes)

            fields_dimension = targetFields.shape
            if len(fields_dimension) == 2:
                num_target_fields = fields_dimension[0]
            else:
                num_target_fields = 1
                targetFields = targetFields[None, ...]
                input_coefs_gpu = input_coefs_gpu[None, ...]

            input_coefs_gpu = input_coefs_gpu / cp.sqrt(cp.sum(cp.absolute(input_coefs_gpu)**2, axis=1))[:, None]

            if num_target_fields > maskcount:
                print(num_target_fields, ' target fields has been provided but only there is memory avaliable for', maskcount)
                num_masks = maskcount
                num_patterns = maskcount
            else:
                num_masks = num_target_fields
                num_patterns = num_target_fields

            _pattern_sequence_scan = np.arange(num_patterns) #overwrite the pattern sequence to the number of target fields

        elif self.engine_mode == 'fibremodes':
            targetFields = _modes
            #num_mask could no tbe same as number of patter (half-piramide case)
            target_Field_count = targetFields.shape[0] 
            num_patterns = target_Field_count
            num_masks = maskcount

        targetField_2D = cp.reshape(targetFields[:num_patterns, ...], (num_patterns, _N, _N))



        return targetField_2D, num_masks, {
            'metric': _metric,
            'filtering': _filtering,
            'downscaling': _downscaling,
            'goal': _goal,
            'max_iterations': _max_iterations,
            'minium_efficiency': _minium_efficiency,
            'convergence_check_interval': self.masksetup.convergence_check_interval,
            'pattern_sequence_scan': _pattern_sequence_scan,
            'min_coupling_SNR_dB': self.masksetup.min_coupling_SNR_dB,
        }

    def _print_run_summary(self, num_masks: int, mask_idx, t0: float, verbose: bool):
        elapsed_time = (time.perf_counter() - t0) * 1000
        if num_masks != mask_idx:
            message = f'{mask_idx}/{num_masks} masks done in {elapsed_time:.1f} ms - '
        else:
            message = f'All {num_masks} masks done in {elapsed_time:.1f} ms'
        log_info(message)
        if verbose:
            print(message)

    def get_gs_properties(self) -> dict:
        """Geometry and catalog values already held on the engine."""
        optical = self.opticalsetup
        mask = self.masksetup
        slm = self.slmplane
        downscale = mask.downscale
        useful = int(optical.masksize) // int(downscale)
        samples = int(slm.SLM_PLANE_N)
        length = float(slm.SLM_PLANE_L)
        mode_array = self.modes.modes_array if self.modes.modes_array is not None else self.modes.modes_array_all
        number_of_modes = None if mode_array is None else int(mode_array.shape[0])
        memory = getattr(self, "cpu_memory", None)
        if memory is not None and "MODES_ARRAY_1D" in memory:
            number_of_modes = int(memory.get("MODES_ARRAY_1D").shape[0])
        waist = self.illumination._calculated_W0
        properties = {
            "engine_mode": self.engine_mode,
            "mode_basis": optical.mode_basis,
            "gs_ready": bool(self._GS_ready),
            "maskcount": getattr(self, "maskcount", None),
            "number_of_modes": number_of_modes,
            "mode_range": self._options.get("mode_range"),
            "slm_size_um": float(optical.slm_pixel_size) * useful * float(downscale),
            "artificial_slm_size_um": length,
            "spatial_resolution_per_um": samples / length,
            "image_plane_samples": samples,
            "useful_slm_samples": int(slm.SLM_N),
            "slm_pixel_size_um": float(optical.slm_pixel_size),
            "pixel_size_fft_um": float(self.kspaceplane.px_kspace),
            "max_spectral_distance_um": float(self.kspaceplane.max_spectral_distance_um),
            "beam_waist_slm_um": None if waist is None else float(waist),
            "beam_waist_mode": self.illumination.mode,
            "core_diameter_um": float(optical.MMF_diameter),
            "core_radius_um": float(optical.MMF_diameter) / 2,
            "forbidden_radius_um": float(mask.filtering_mask_radius_factor) * float(optical.MMF_diameter) / 2,
            "pupil_radius_um": float(optical.pupil_um) / 2.0,
        }
        if not hasattr(self, "_setup_feasible"):
            return properties
        properties.update(
            {
                "illumination_outside_pupil_power": float(self._illumination_outside_pupil_power),
                "illumination_pow_check_passed": bool(self._illumination_pow_check_passed),
                "mode_basis_power_outside_pupil": float(self._mode_basis_power_outside_pupil),
                "modes_pow_check_passed": bool(self._modes_pow_check_passed),
                "illumination_overlap_support_passed": bool(self._illumination_overlap_support_check_passed),
                "setup_feasible": bool(self._setup_feasible),
                "NA_MMF_system_max": float(self._NA_MMF_system_max),
                "ill_coverage_rad_px_approx": _left_edge_radius_px(self._ill_coverage_rad_px_approx),
                "modes_coverage_rad_px_approx": _left_edge_radius_px(self._modes_coverage_rad_px_approx),
            }
        )
        return properties

    def save(self, folder=None, label=None):
        if not self._GS_ready:
            raise RuntimeError("GS engine is not ready to save")
        return save_library(self, folder=folder, label=label)

    def run(self, inpucoefs: np.ndarray | None = None, verbose: bool = False):
        if not self._GS_ready:
            message = 'It seems the GS engine is not ready, probably the mode basis is not set yet.'
            print(message)
            log_info(message)
            return

        log_run_preamble(self.get_gs_properties())
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()

        targetField_2D, num_masks, ctx = self._prepare_run(inpucoefs)

        #pattern idx could be from lets say 100 to 120 but it will index in the range of 0 to 20 (that is mask_idx)
        mask_idx = 0 

        for iter_idx, pattern_idx in enumerate(ctx['pattern_sequence_scan']):
            _results = self._single_masks_GS_loop(
                targetField_2D[pattern_idx, ...],
                self._masks_rescaling_properties,
                metric=ctx['metric'],
                goal=ctx['goal'],
                max_iterations=ctx['max_iterations'],
                minium_efficiency=ctx['minium_efficiency'],
                convergence_check_interval=ctx['convergence_check_interval'],
                filtering=ctx['filtering'],
                downscale=ctx['downscaling'],
                mask_idx = iter_idx,
                min_coupling_SNR_dB=ctx['min_coupling_SNR_dB'],
                verbose=verbose,
            )

            mask_idx = self._store_GS_results_on_cpu(_results, mask_idx, verbose=verbose) #increments mask_idx by 1 everytime is called

            if self.engine_mode == 'fibremodes' and self.modes.piramid_scheme is not None:
                #check if target mode needs complementary as:
                pair_modes = self.modes.piramid_scheme[pattern_idx]
                if len(pair_modes) == 2:
                    mask_idx = self._store_GS_results_on_cpu(_results, mask_idx, complementary_mask=True, verbose=verbose)
    
        self._print_run_summary(num_masks, mask_idx, t0, verbose)

    #! Helper function to move tmp results from a stream to the cpu memory (final buffer results)
    def _store_GS_results_on_cpu(self, slot: dict, mask_idx: int, complementary_mask: bool = False, verbose: bool = False):
        message = (
            f'Indexing results to index: {mask_idx} - Complementary mask: {complementary_mask}'
            f'\n*****************************************************'
        )
        log_info(message)
        if verbose:
            print(message)
        input_phase_mask = slot['phase_mask'] if not complementary_mask else self._calculate_complementary_mask(slot['phase_mask'])
        new_mask_idx = self._index_GS_results(slot, input_phase_mask, mask_idx) #everytime we index we increment  
        return new_mask_idx

    def _index_GS_results(self, slot: dict, phase_mask_gpu: cp.ndarray, mask_idx: int, ):
        i = mask_idx
        self.phase_masks[i] = cp.asnumpy(cp.angle(phase_mask_gpu)) #store the mask itself only
        self.efficiency[i] = slot['efficiency']
        self.D_core_coefs[i] = cp.asnumpy(slot['D_core_coefs'])
        self.convergence_history[i] = cp.asnumpy(slot['convergence'])
        self.iteration_convergence[i] = slot['iteration_convergence']
        self.efficiency_history[i] = cp.asnumpy(slot['efficiency_history'])
        self.mask_passed[i] = slot['passed']
        self.basis_coupling_snr[i] = slot['basis_coupling_snr']
        self.dumping_field[i] = slot['dumping_field']
        self.core_field[i] = slot['core_field']
        self.leaked_power[i] = slot['leaked_power']
        return i + 1


    def _calculate_complementary_mask(self, gpu_mask: cp.ndarray):
        cmplex_mirrored = cp.flipud(cp.fliplr(cp.conj(gpu_mask)))
        return cmplex_mirrored


    def _single_masks_GS_loop(self, target_field_gpu: cp.ndarray,  masks_rescaling_properties: dict = None,
                            target_field_coefs_gpu: cp.ndarray | None = None, 
                            metric:str = 'overlap', goal:float = 0.99, max_iterations: int = 300,
                            minium_efficiency: float = 0.1,
                            min_coupling_SNR_dB: float = 20,
                            convergence_check_interval: int = 1, filtering: bool = False, 
                            downscale:bool = False,
                            mask_idx:int = 0,
                            verbose:bool = False):
        
        target_field = normalize_field(target_field_gpu, power_field(target_field_gpu)) 
        target_at_slm = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(target_field)))
        A = normalize_field(target_at_slm, power_field(target_at_slm))
        B = self.gpu_memory.get('ILLUMINATION')
        R0 = self.gpu_memory.get('R0')
        R_CORE = self.gpu_memory.get('R_CORE')
        GAUSSIAN_FILTER = self.gpu_memory.get('GAUSSIAN_FILTER')
        MODES_ARRAY_1D_HERMITIAN = self.gpu_memory.get('MODES_ARRAY_1D_HERMITIAN')

        #If not provide, reshape to 1D and perform the modal decomposition
        if target_field_coefs_gpu is None:
            target_field_coefs_gpu = cp.matmul(target_field.flatten().astype(cp.complex64), MODES_ARRAY_1D_HERMITIAN)

        w = 1 #scatter power weight
        R1 = cp.negative(R0) + 1

        _convergence = cp.zeros(max_iterations, dtype=cp.float32)
        _efficiency = cp.zeros(max_iterations, dtype=cp.float32)
        _goal_gpu = cp.asarray(goal, dtype=cp.float32) 
        _goal_gpu_previous = _goal_gpu.copy() * 0
        _convergence_cnt = max(1, int(convergence_check_interval))

        _gradient_threshold = -0.001 #if gradient start becoming negative, stop the GS loop
        _gradient_security_stop_overlap_margin = 0.01
        _field_backup = complex_field(B, A)
        C = _field_backup
        #! GS loop
        for i in range(max_iterations):
            #From SLM plane to Fiber plane
            _field_backup = C.copy()
            C = complex_field(B, A)
            D = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(C)))
            D = normalize_field(D, power_field(D))

            #Goal checking at the Fiber plane --> 
            D_core_1D = cp.reshape(D, -1).astype(cp.complex64) #Reshape the current field at the fiber faced normalized to 1
            D_core_coefs = cp.matmul(D_core_1D, (MODES_ARRAY_1D_HERMITIAN)) #Get the coupling into the modes basis
            if metric == 'overlap':
                #*Before checking the goal between target and current field, I need to normalized
                #*all power coupled into the basis so we guide the GS to converge towards the target field.
                D_core_coefs_normalized = normalize_field(D_core_coefs, power_field(D_core_coefs)) 
                #! Here basically depends of the goal target, you can reach same results but one will need 90% 
                #! or otherwise you loose all your power trying to achieve lets say 99%.
                #goal_check = cp.sum(cp.absolute(D_core_coefs_normalized * cp.conj(target_field_coefs_gpu))) # Less restritive check as each mode has non-negative contribution to the sum
                #goal_check = cp.abs(cp.vdot(D_core_coefs_normalized, target_field_coefs_gpu))# Overlap without square
                goal_check = cp.abs(cp.vdot(D_core_coefs_normalized, target_field_coefs_gpu))**2 # Overlap, squared, very hard

            elif metric == 'crosstalk':
                ...


            _convergence[i] = goal_check #keep track of the goal evolution check
            _goal_gradient = goal_check - _goal_gpu_previous
            _goal_gpu_previous = goal_check
            _efficiency[i] = power_field(D_core_coefs) #keep track of the efficiency evolution.
            # Spatial core power (not modal efficiency) — stop if injection drops too low
            _core_power = power_field(D * R_CORE)
            _core_power_too_low = bool((_core_power <= minium_efficiency).get())
            # Only stop by graddient if we were close to the goal by a margin, otherwise keep pushing
            _close_to_goal = bool((goal_check > (_goal_gpu - _gradient_security_stop_overlap_margin)).get())
            _negative_convergence_gradient = (bool((_goal_gradient < _gradient_threshold).get()) and _close_to_goal)

            #* Leakage is the total power in the core - power coupled into the basis.
            #* In theory we want all core coupled into the basis so all power is transmitted into the fiber.
            #* It could be that there is remaining power in the core that cannot be coupled into the basis.
            #* That is ok, the problem is that the basis is not 100% right then it could be transmited decomposed
            #* In the real modes we are not tracking hence creating some speakled at the top of the field.
            #* We then we want to minimized this.
            _leakage = _core_power - _efficiency[i] #Total power in the core - power coupled into the basis (aka modes)
            _leakage = cp.maximum(_core_power - _efficiency[i], 1e-8) # Avoid division by zero
            _basis_coupling_SNR =  10 * cp.log10(_efficiency[i] / _leakage) # Signal to noise ratio of the basis coupling
            #print(i, 'Basis to Core power SNR', _basis_coupling_SNR)
            target_coupling_SNR =  min_coupling_SNR_dB #dB
   
            if i % _convergence_cnt == 0 or i == max_iterations - 1 or _core_power_too_low or _negative_convergence_gradient:
                _did_it_converged = (goal_check >= _goal_gpu) & (_basis_coupling_SNR >= target_coupling_SNR)

                if _did_it_converged.get() or i == max_iterations - 1 or _core_power_too_low or _negative_convergence_gradient:
                    
                    #! Revert to previous iteration: Mask and efficience are reverted the power distribution is not
                    if _negative_convergence_gradient:
                        C = _field_backup #use the previous iteration field to avoid going back and forth
                        i -= 1 #decrement the iteration to avoid skipping the current iteration

                    efficiency = _efficiency[i] #How much power of the total field across cladding + core ended making it into the basis
                    iteration_convergence = i


                    #! Check the power distribution
                    _dumping_field = power_field(D * R1)
                    _core_field = _core_power
                    _allowed_region_field = power_field(D * R0)
                    _power_delivered_inbetween_region = power_field(D * (R0 - R_CORE)) # Power in between core and dumping region
                    _uncontrolled_field_power = _allowed_region_field - efficiency # Total uncrontrol power in the allowed region
                    _leaked_power = _core_field - efficiency # Uncontrolled power in the core
                    _passed = self._did_it_passed(
                        bool((goal_check >= _goal_gpu).get()),
                        bool((_basis_coupling_SNR >= target_coupling_SNR).get()),
                        not _negative_convergence_gradient,
                    )
                    basis_coupling_snr = _as_float(_basis_coupling_SNR)
                    dumping_field = _as_float(_dumping_field)
                    core_field = _as_float(_core_field)
                    leaked_power = _as_float(_leaked_power)

                    message = (
                        f'- {mask_idx} --> Iteration {i} - Efficiency {efficiency:.3f} - Goal check {goal_check:.3f} - Basis to Core power SNR {_basis_coupling_SNR:.2f} dB \n'
                        f'    | Dumping field (%) {_dumping_field * 100:.2f} - Core field (%) {_core_field * 100:.2f} - Uncontrolled leaked power in the core (%) {_leaked_power * 100:.4f}\n'
                        f'    | Goal gradient {_goal_gradient:.4f} - Gradient threshold {_gradient_threshold:.4f}'
                        f'\n ------------------------------------------------'
                    )
                    log_mask_result(message, _passed)
                    if verbose:
                        print(message)

                    # plot(abs(cp.asnumpy(D_core_coefs))**2)
                    # fig, ax = subplots(1,2, figsize = (10, 5))
                    # ax[0].imshow(abs(cp.asnumpy(D[D.shape[0]//2 - 50: D.shape[0]//2 + 50, D.shape[1]//2 - 50: D.shape[1]//2 + 50])**2))
                    # ax[1].imshow(abs(cp.asnumpy(target_field[D.shape[0]//2 - 50: D.shape[0]//2 + 50, D.shape[1]//2 - 50: D.shape[1]//2 + 50]))**2)


                    break

            #Control the ampount of power delivered to the fiber or to the clading:
            #w parameter controls how much light goes to the core and how much goes to the cladding
            #Total power always must be 1
            scatter_field = ( D * R1 ) #Selects the light outside the region of interest (core)

            if filtering:
                scatter_field = GAUSSIAN_FILTER * scatter_field

            scatter_field_normalized = normalize_field(scatter_field, power_field(scatter_field))
            scatter_field_weighted = cp.sqrt(1-w) * scatter_field_normalized # weight the scatter field

            #Create the whole field (Core + cladding with power = 1)
            # target * R0 normalized to the rest of the power which is not scattered + the scatter field
            E = target_field*R0
            E_pow = cp.sqrt(power_field(E))
            E = create_field(E, E_pow, scatter_field_weighted, w)
            A = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E))) #Back to SLM plane

            #Mod the weight
            #w = w * 0.9
            #sigmoid function weight decay
            w_min = 0.1
            tau = 80   # transition mostly done by ~80 iterations (tune this, not max_iterations)
            t = min(i / tau, 1.0)
            w = w_min + (1 - w_min) * 0.5 * (1 + np.cos(np.pi * t))

        #Get some properties of the mask to upsample/downsample or both
        c = masks_rescaling_properties['center']
        c2 = masks_rescaling_properties['physical_center']
        crop = masks_rescaling_properties['crop']

        #computed_GS = cp.absolute(self.gpu_memory.get('ILLUMINATION')) * cp.exp(1j*cp.angle(C))
        computed_GS = 1 * cp.exp(1j*cp.angle(C)) #Make all the frequencies equal weight for upsampling
        phase_mask = computed_GS[c-crop:c+crop,c-crop:c+crop]

        if downscale:
            phase_mask = computed_GS[c-crop:c+crop,c-crop:c+crop]
            phase_mask_rescaled = self.gpu_memory.get('PHASE_MASK_RESCALED')
            fft_phase_mask = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(phase_mask)))
            phase_mask_rescaled[c2-crop:c2+crop,c2-crop:c2+crop] = fft_phase_mask
            phase_mask =  cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(phase_mask_rescaled))) #return the phase

            phase_mask_rescaled *= 0 #Reset the memory


        return {
                'mask_idx': mask_idx,
                'phase_mask': phase_mask,           # GPU
                'efficiency': float(efficiency),
                'D_core_coefs': D_core_coefs,       # GPU
                'convergence': _convergence,
                'iteration_convergence': iteration_convergence,
                'efficiency_history': _efficiency,
                'passed': _passed,
                'basis_coupling_snr': basis_coupling_snr,
                'dumping_field': dumping_field,
                'core_field': core_field,
                'leaked_power': leaked_power,
            }


    # mask_passed bits, high to low: gradient fail, SNR fail, overlap fail.
    #   000 -> 0  passed
    #   001 -> 1  overlap
    #   010 -> 2  SNR
    #   011 -> 3  overlap + SNR
    #   100 -> 4  gradient
    #   101 -> 5  gradient + overlap
    #   110 -> 6  gradient + SNR
    #   111 -> 7  all three
    @staticmethod
    def _did_it_passed(overlapPASS: bool, SNRPASS: bool, gradientPASS: bool):
        bits = f"{int(not gradientPASS)}{int(not SNRPASS)}{int(not overlapPASS)}"
        return int(bits, 2)





















        
