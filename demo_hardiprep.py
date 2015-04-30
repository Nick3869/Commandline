# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 09:44:46 2015

@author: Shireen Elhabian

Objective:
    this script shows how to use the hardiprep pipeline on a single subject dataset
    
Input:
    DWI nrrd file
    output directory
"""

from __future__ import division
import os
import glob
import numpy as np
import ntpath
import time

import hardi.nrrd
import hardi.io as hardiIO
import hardi.qc as hardiQC


# the nrrdfilename to be processed
nrrdfilename = 'SAMPLE_DATA/5624895_DSI.nrrd'

# where to put the processing output
outDir       = 'SAMPLE_DATA'

"""
---------------------------------------------------------------------------------
STEP Zero 
---------------------------------------------------------------------------------

(1) create hardi QC directories and copy the original nrrd
(2) fix the dimension and thickness of the given nrrd files
(3) convert them to nifti and generate the btable for dsi_studio visualization
 
"""

#prepDir = hardiQC.PrepareQCsession(nrrdfilename, outDir)
prepDir = os.path.join(outDir, 'HARDIprep_QC')

"""
---------------------------------------------------------------------------------
STEP ONE 
---------------------------------------------------------------------------------

OBJECTIVE:
    dtiprep without motion correction
    check the quality of the individual directions, e.g. missing slices, intensity artifacts, Venetian blind
"""

xmlfilename = os.path.join('PROTOCOLS/protocol_DTIPrep1.2.3_DSI.xml')
#hardiQC.RunDTIPrepStage(nrrdfilename, prepDir, xmlfilename)


"""
---------------------------------------------------------------------------------
STEP TWO 
---------------------------------------------------------------------------------

OBJECTIVE:
    quantify fast bulk motion within each gradient to exclude those having intra-scan
    motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)
    
    here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded
"""

hardiQC.PerformWithinGradientMotionQC(nrrdfilename, prepDir)

"""
---------------------------------------------------------------------------------
STEP TRHEE
---------------------------------------------------------------------------------
Assumptions:
    (1) WithinGradientMotionQC has been performed
    (2) DTIPrep (without motion) has been performed

"""
hardiQC.ExtractBaselineAndBrainMask(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP FOUR
---------------------------------------------------------------------------------

OBJECTIVE:
    resample the corrupted slices (detected via DTIPrep and within-gradient motion) Qc
    in the q-space plus gradient-wise denoising
"""

hardiQC.PerformResampleCorruptedSlicesInQspaceAndGradientDenoise(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP FIVE
---------------------------------------------------------------------------------
OBJECTIVE:
    construct the reference volume for each data set using Qball-FMAM
    
    we will use fit model to all measurements since the phantom data would have the 
    minimal motion yet when move to infants we will need to implement a restore-like modelbased motion correction
"""
hardiQC.ComputeModelBasedReference_FMAM(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP SIX
---------------------------------------------------------------------------------

OBJECTIVE:
    correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already 
    been carried out, as such we can use standard interpolation
"""

hardiQC.PerformModelBasedReferenceMotionCorrectionMCFLIRT12DOF(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP SEVEN
---------------------------------------------------------------------------------

OBJECTIVE:
    brain mask the motion corrected datasets
"""

hardiQC.BrainMaskModelBasedMotionCorrectedDWI(nrrdfilename, prepDir)

"""
---------------------------------------------------------------------------------
STEP EIGHT
---------------------------------------------------------------------------------

OBJECTIVE:
    compute the bias field based on the baseline and apply it on the other gradients
"""
hardiQC.PerformBiasFieldCorrection(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP NINE
---------------------------------------------------------------------------------

OBJECTIVE:
    jointly denoise the bias field corrected data
"""
hardiQC.PerformJointLMMSEDenoising(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP TEN
---------------------------------------------------------------------------------

OBJECTIVE:
    reconstruction of fiber ODFs in subject space
"""
hardiQC.ReconstructFODFs(nrrdfilename, prepDir)


"""
---------------------------------------------------------------------------------
STEP ELEVEN
---------------------------------------------------------------------------------

OBJECTIVE:
    full brain tractography in subject space
"""
hardiQC.PerformFullBrainTractography(nrrdfilename, prepDir)


