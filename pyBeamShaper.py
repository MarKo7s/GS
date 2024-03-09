import sys
import pathlib
p = pathlib.Path(__file__).parent.parent
path_to_MODULES = p
#print(path_to_MODULES)
sys.path.append(str(path_to_MODULES))

from numpy import *
import cupy as cp

from GS.pyPhasemasks import SetupGenericMaskGSenvioment
from spatial_light_modulator.pyLCOS import LCOS


class BeamShappingAttenuator:
    def __init__(self):
        self.patterndiff = zeros(2)
        self.slmpowdiff = zeros(2)
    
    def FindAttenuation(self, SLMpowAv, PatternpowAv):
        '''
        It work out where to scatter power in other to get the desired power ratio between pols
        
        IN:
            SLMpowAv (input): array dimension 2 [H, V] --> Power avaliable on each arm of the SLM (each pol) in linear units
            PatternpowAv (input): array dimension 2 [H, V] --> Power desired on each arm of the SLM in linear units
        OUT:
            Attenuation (output): attenuation in dB to be applied
            Attenuation Index (output): where to apply the attenuation (H or V - 0 or 1)    
            
        '''
        #Make sure they are 0
        self.patterndiff.fill(0)
        self.slmpowdiff.fill(0)
        
        powdiffpattern, pattIdx = self.power_offset(PatternpowAv)
        self.patterndiff[pattIdx] = powdiffpattern
        
        powdiffslm, slmIdx = self.power_offset(SLMpowAv)
        self.slmpowdiff[slmIdx] = powdiffslm
        
        return self.calcAttenuation(self.slmpowdiff, self.patterndiff)
            
    @staticmethod
    def calcAttenuation(slm, pattern):
        diff = pattern - slm
        attout = diff[0] - diff[1]
        if attout < 0:
            attidx = 0 # Attenuate H
        else:
            attidx = 1 # Attenuate V
        return(abs(attout), attidx)

    @staticmethod
    def power_offset(powerSet):
        Mmax = powerSet.max()
        Mmin = powerSet.min()
        powerOffset = 10*log10(Mmax / Mmin) #Offset in dB between set of power
        Idx = where(powerSet == Mmax)[0][0] #Which element was the largest
        return(powerOffset, Idx) 
    
class BeamShaperGS:
    def __init__(self, GSengineObject, slmObject, MTM = None):
        self.maskgenerator: SetupGenericMaskGSenvioment = GSengineObject
        self.slm: LCOS = slmObject
        self.MTM = MTM
        #Create a instance of BeamShappingAttenuator:
        self.BS_att = BeamShappingAttenuator()
        #Check how many modes are we talking my dudes
        if MTM == None:
            self.modescount = None
        else:
            self.modescount = MTM.shape[0] // 2
            
        #self.coefsin_H = zeros(self.modescount, dtype = complex64)
        #self.coefsin_V = zeros_like(self.coefsin_H)
        self.coefsin = None #zeros(self.modescount * 2, dtype = complex64)
        self.coefsout = None # zeros_like(self.coefsin)
        self.powOutH = 0
        self.powOutV = 0
        self.GSEff = None
        self.slmEff = ones(2)
        self.pol = ['H', 'V']
        
        self.specs = {'HpowOut': 0, 'VpowOut': 0, 'GSeff': {'H': 1 , 'V': 1}, 'attenuation': 0, 'where': None  }
        
        print('SUCCESS')
        
    def set_slm_efficiency(self):
        pass
        
    def set_MTM(self, MTM):
        self.MTM = MTM
        self.modescount = self.MTM.shape[0] // 2
              
    def BackPropagateTargets(self, coefsH, coefsV, normalize = True):
        
        if normalize == True:
            #normalize the input:
            HpowIn = sum(abs(coefsH)**2)
            VpowIn = sum(abs(coefsV)**2)
            #If one is 0 no need to normalize before propagating through the fibre
            if (HpowIn > 0 and VpowIn > 0):
                cam_plane_coefs_H = coefsH / sqrt(HpowIn)
                cam_plane_coefs_V = coefsV / sqrt(VpowIn)
            else:
                cam_plane_coefs_H = coefsH
                cam_plane_coefs_V = coefsV 
        else:
            cam_plane_coefs_H = coefsH
            cam_plane_coefs_V = coefsV 
            
        self.coefsin = zeros(self.modescount * 2, dtype = complex64) #reset 
        self.coefsin[::2] = cam_plane_coefs_H
        self.coefsin[1::2] = cam_plane_coefs_V
        
        self.coefsout  = matmul(self.coefsin,(transpose(self.MTM))) #Backpropgation of the Target towards the input fiber -- NO PANIC - CONJUGATE and TRY AGAIN
        
        self.powOutH = sum(abs(self.coefsout[::2])**2)
        self.powOutV = sum(abs(self.coefsout[1::2])**2)
        
    def GenerateMasks(self):
        self.maskgenerator.calcMasksFromCoefs_gpu(vstack((self.coefsout[::2],self.coefsout[1::2]))) #slice and stack the coefs
        self.GSEff = cp.asnumpy(self.maskgenerator.efficiency)
        
    def BeamShape(self, PatternCoefsH, PatternCoefsV, normalize = True, printting = False) :
        self.BackPropagateTargets(coefsH = PatternCoefsH, coefsV= PatternCoefsV, normalize = normalize )
        self.GenerateMasks()
        
        attmaskdBs, attmaskIdx = self.BS_att.FindAttenuation(SLMpowAv = array([self.slmEff[0]* self.GSEff[0], self.slmEff[1] * self.GSEff[1]] ), PatternpowAv = array([self.powOutH, self.powOutV]))
        
        #Reset parameters of the masks
        self.slm.resetAttenuation()
        self.slm.patternEnabled = 1
        self.slm.zernikesEnabled = 1
        #enable patterns and zernikes
        self.phase_mask = cp.asnumpy(self.maskgenerator.phase_masks_rescaled)
        self.slm.mask_specs['H']['pattern'] = angle(self.phase_mask[0])
        self.slm.mask_specs['V']['pattern'] = angle(self.phase_mask[1])
        self.slm.mask_specs[self.pol[attmaskIdx]]['att_enabled'] = 1
        self.slm.mask_specs[self.pol[attmaskIdx]]['attWeight'] =  attmaskdBs
        self.slm.setmask(pol='HV')
        
        if printting == True:
            print(f'Needed power at each SLM arm: H {self.powOutH}, V {self.powOutV}')
            H_eff = self.GSEff[0]
            V_eff = self.GSEff[1]
            print(f'GS efficiency: H {H_eff:0.2f}% - V {V_eff:0.2f}%')
            print(f'To fullfill requirements: Att.= {attmaskdBs} on {self.pol[attmaskIdx]}')

if __name__ == '__main__':
    pass