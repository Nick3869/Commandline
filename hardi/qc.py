# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 10:00:22 2015

@author: Shireen Elhabian
"""

from __future__ import division
import os
import glob
import numpy as np
import ntpath
import time
import shutil
import copy
import csv

import nibabel as nib
import hardi.io as hardiIO
import hardi.nrrd as nrrd # I modified this package to fix some bugs
import hardi.qc_utils as hardiQCUtils

import dipy.reconst.shm as drecon
from dipy.core.gradients import gradient_table

def PrepareQCsession(nrrdfilename, outDir):
    
    start_time = time.time()
        
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    index        = basename.find('_DWI_65dir')
    if index >= 0:
        phan_name = basename[:index]
    else:
        phan_name = basename
    
    prepDir = os.path.join(outDir, 'HARDIprep_QC')
    if os.path.exists(prepDir):
        shutil.rmtree(prepDir)
    os.mkdir(prepDir)
    os.mkdir(os.path.join(prepDir, 'DWI_65dir'))
    
    # copy the nrrd file
    destnfile  = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.nrrd' %(phan_name))
    cmdStr = 'cp -Rv %s %s' % (nrrdfilename, destnfile)
    os.system(cmdStr)
   
   # prepare the files for subsequent processing
    nrrdfilename   = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.nrrd' % (phan_name) )
    niifilename    = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.nii' % (phan_name) )
    bvecsfilename  = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.bvecs' % (phan_name) )
    bvalsfilename  = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.bvals' % (phan_name) )
    btablefilename = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir_btable.txt' % (phan_name) )
    srcfilename    = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.src.gz' % (phan_name) )
    
    # fix the nrrd file (thickness and directions are the last dimension)
    hardiIO.fixNRRDfile(nrrdfilename)
        
    # convert to nifti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename, niifilename, bvecsfilename, bvalsfilename)

    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename, bvecsfilename, btablefilename)
    
    # save src file
    hardiIO.nifti2src(niifilename, btablefilename, srcfilename)

    end_time = time.time()
    
    print 'PrepareQCsession: time elapsed = %f seconds ...' % (end_time - start_time)
    
    return prepDir

"""
---------------------------------------------------------------------------------
STEP ONE 
---------------------------------------------------------------------------------

OBJECTIVE:
    dtiprep without motion correction
    check the quality of the individual directions, e.g. missing slices, intensity artifacts, Venetian blind
"""

def RunDTIPrepStage(nrrdfilename, prepDir, xmlfilename):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    index        = basename.find('_DWI_65dir')
    if index >= 0:
        phan_name = basename[:index]
    else:
        phan_name = basename
        
        
    nrrdfilename   = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir.nrrd' % (phan_name) )
    qcnrrdfilename = os.path.join(prepDir, 'DWI_65dir/%s_DWI_65dir_QCed.nrrd' % (phan_name) )    
    
    cmdStr = 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (nrrdfilename, xmlfilename, os.path.join(prepDir, 'DWI_65dir'))
    os.system(cmdStr)
        
    end_time = time.time()
    
    print 'RunDTIPrepStage: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP TWO 
---------------------------------------------------------------------------------

OBJECTIVE:
    quantify fast bulk motion within each gradient to exclude those having intra-scan
    motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)
    
    here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded

TODO: 
    (0) fix the dimension and thickness of the given nrrd files
    (1) read the nrrd files (provided directly by Clement) listed in the nrrdfilenames.csv (only process data with > 30 gradients)
    (2) compute the signal drop-out per slice and a score for the whole gradient volume
    (3) report a QC report that includes scores and excluded scans and a nrrd file with the resulting QCed acquisition
    
Assumptions:
    (1) DTIprep stage has been performed
   
"""

def PerformWithinGradientMotionQC(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    nrrdfilename = os.path.join(prepDir, 'DWI_65dir/%s_QCed.nrrd' % (basename) )    
        
    reportfilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCreport.txt' %(basename))
    nrrdfilename_new             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.nrrd' %(basename))
    
    niifilename                  = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.nii' %(basename))
    bvecsfilename                = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.bvecs' %(basename))
    bvalsfilename                = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.bvals' %(basename))
    btablefilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_btable.txt' %(basename))
    srcfilename                  = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.src.gz' %(basename))
   
    # fix the nrrd file (thickness and directions are the last dimension)
    hardiIO.fixNRRDfile(nrrdfilename)
    
    # read the nrrd data along with hardi protocol information
    nrrdData, bvalue, gradientDirections, baselineIndex, options = hardiIO.readHARDI(nrrdfilename)
            
    # detect within gradient motion artifact
    nMotionCorrupted, slice_numbers = hardiQCUtils.DetectWithinGradientMotion(nrrdData, baselineIndex, bvalue)
    
    # write the QC report
    nExcluded = hardiQCUtils.WriteWithinGradientMotionQCReport(reportfilename, nMotionCorrupted, slice_numbers, baselineIndex)
        
    # construct the corrected sequence     
    correctedData, gradientDirections_new = hardiQCUtils.ConstructWithinGradientMotionCorrectedData(nrrdData, gradientDirections, nExcluded, nMotionCorrupted)
            
    options = hardiIO.updateNrrdOptions(options, gradientDirections_new)
    
    # save as nrrd with the save options as the original nrrd file
    nrrd.write( nrrdfilename_new, correctedData, options)
                
    # fix the nrrd file (thickness and directions are the last dimension)
    hardiIO.fixNRRDfile(nrrdfilename_new)
    
    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_new, niifilename, bvecsfilename, bvalsfilename)
    
    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename, bvecsfilename, btablefilename)
    
    # save src file
    hardiIO.nifti2src(niifilename, btablefilename, srcfilename)
        
    end_time = time.time()
    
    print 'PerformWithinGradientMotionQC: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP TRHEE
---------------------------------------------------------------------------------
Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    

TODO: 
    (0) fix the dimension and thickness of the given nrrd files
    (1) extract the baseline
    (2) convert them to nifti
    (3) save the corresponding bvecs, bvals and btable
    (4) save the src files for visualization in dsi_studio
    (5) use the baseline to extract brain mask from baseline (to be used in motion correction) but don't do the actual masking
"""
def ExtractBaselineAndBrainMask(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    nrrdfilename   = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.nrrd' % (basename) )    
        
    baselineDir                  = os.path.join(prepDir, 'baseline' )
    baselinenrrdfilename         = os.path.join(prepDir, 'baseline/%s_baseline.nrrd' %(basename))
    
    baselinenrrdfilename_masked  = os.path.join(prepDir, 'baseline/%s_baseline_masked.nrrd' %(basename))
    baselineniifilename_masked   = os.path.join(prepDir, 'baseline/%s_baseline_masked.nii' %(basename))
    
    baselineNiifilename          = os.path.join(prepDir, 'baseline/%s_baseline.nii' %(basename))
    tempbvecsfilename            = os.path.join(prepDir, 'baseline/%s_baseline.bvecs' %(basename))
    tempbvalsfilename            = os.path.join(prepDir, 'baseline/%s_baseline.bvals' %(basename))
    
    niiBrainFilename             = os.path.join(prepDir, 'baseline/%s_baseline_brain.nii' %(basename))
    niiBrainMaskFilename         = os.path.join(prepDir, 'baseline/%s_baseline_brain_mask.nii' %(basename))
    
    if os.path.exists(baselineDir) == False:
        os.mkdir(baselineDir)
                
    # save the baseline to nrrd
    hardiIO.extractAndSaveBaselineToNRRD(nrrdfilename, baselinenrrdfilename)
  
    # then convert to nifti                
    hardiIO.convertToNIFTI(baselinenrrdfilename, baselineNiifilename, tempbvecsfilename, tempbvalsfilename)
    
    cmdStr = 'rm -f %s' % (tempbvecsfilename)
    os.system(cmdStr)
    
    cmdStr = 'rm -f %s' % (tempbvalsfilename)
    os.system(cmdStr)
    
    print 'Extract brain region ...'
    # get the mask based on FSL-BET tool - all nrrd files are converted to nifti in a bash script
    brainMask = hardiQCUtils.extractBrainRegion(os.path.abspath(baselineNiifilename),os.path.abspath(niiBrainFilename), os.path.abspath(niiBrainMaskFilename) )
    
    print 'Mask the baseline volume ...'
    nrrdData, options        = nrrd.read(baselinenrrdfilename)
    nrrdDataMasked           = hardiQCUtils.brainMaskingVolume(nrrdData, brainMask)
    
    nrrd.write(baselinenrrdfilename_masked, nrrdDataMasked, options)
    hardiIO.convertToNIFTI(baselinenrrdfilename_masked, baselineniifilename_masked, tempbvecsfilename, tempbvalsfilename)
    
    cmdStr = 'rm -f %s' % (tempbvecsfilename)
    os.system(cmdStr)
    
    cmdStr = 'rm -f %s' % (tempbvalsfilename)
    os.system(cmdStr)
    
    end_time = time.time()
    
    print 'ExtractBaselineAndBrainMask: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP FOUR
---------------------------------------------------------------------------------

OBJECTIVE:
    resample the corrupted slices (detected via DTIPrep and within-gradient motion) Qc
    in the q-space
    
    This is the implementation of the correction strategy based on the below paper
    in order to replace DTIPrep (which excluded gradients suffering from intensity artifacts)
    this correction strategy is trying to correct for within-slice, within-volume and betwee-volumes
    motion artifacts ...
    
    Dubois, Jessica, Sofya Kulikova, Lucie Hertz-Pannier, Jean-François Mangin, Ghislaine Dehaene-Lambertz, 
    and Cyril Poupon. "Correction strategy for diffusion-weighted images corrupted with motion: 
    application to the DTI evaluation of infants' white matter." Magnetic resonance imaging 32, no. 8 (2014): 981-992.
    
Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) baseline and brain mask extraction
    
TODO: 
    (1) get a list of gradients-slices that are to be resampled (identified as corrupted via DTIprep and within gradient motion QC)
    (2) for the corrupted slices, fit Qball on the non-corrupted data (regularized version based on Descoteaux, M., et. al. 2007. Regularized, fast, and robust analytical
        Q-ball imaging.)
    (3) resample corrupted slices based on the fitted dODF
    (4) gradient-based denoising (since DTIprep is supposed to do denoising first which it didn't !!!, we need to retain this anyway after the resampling)
    (5) save as nrrd, nii and other formats
"""

def PerformResampleCorruptedSlicesInQspaceAndGradientDenoise(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    # DTIPrep has problem in calling the denoising filter, so the QCed nrrd is not actually denoised
    
    # filter parameters - gradient-wise denoising
    EstimationRadius_x        = 11 
    EstimationRadius_y        = 11 
    EstimationRadius_z        = 0
    FilteringRadius_x         = 11  
    FilteringRadius_y         = 11  
    FilteringRadius_z         = 0
    NoOfIterations            = 1
    minVoxelsFiltering        = 7
    minVoxelsEstimation       = 7
    histRes                   = 2
    minStd                    = 0
    maxStd                    = 100
    absVal                    = 0 
    
    nrrdfilename_orig            = os.path.join(prepDir, 'DWI_65dir/%s.nrrd' % (basename) )    
    niifilename_orig             = os.path.join(prepDir, 'DWI_65dir/%s.nii' %(basename) )    
    bvecsfilename_orig           = os.path.join(prepDir, 'DWI_65dir/%s.bvecs' %(basename) )    
    bvalsfilename_orig           = os.path.join(prepDir, 'DWI_65dir/%s.bvals' %(basename) )   
    btablefilename_orig          = os.path.join(prepDir, 'DWI_65dir/%s_btable.txt' %(basename) )   
    
    nrrdfilename                 = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.nrrd' % (basename) )    
    niifilename                  = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.nii' %(basename) )    
    bvecsfilename                = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.bvecs' %(basename) )    
    bvalsfilename                = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed.bvals' %(basename) )   
    btablefilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_btable.txt' %(basename) )   
    
    nrrdfilename_rq              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.nrrd' % (basename) )    
    niifilename_rq               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.nii' %(basename) )    
    bvecsfilename_rq             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.bvecs' %(basename) )    
    bvalsfilename_rq             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.bvals' %(basename) )   
    btablefilename_rq            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_btable.txt' %(basename) )   
    srcfilename_rq               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.src.gz' %(basename) )   
        
    #(1) reconstruct the dODF based on regularized Qball based on the sequence from the previous QC steps
    
    diffusionData, bvalue, gradientDirections, baselineIndex, options = hardiIO.readHARDI(nrrdfilename)
    bvals, bvecs = hardiIO.readbtable(btablefilename)
    gtable       = gradient_table(bvals, bvecs)

    qball_model  = drecon.QballModel(gtable, sh_order = 6, smooth = 0.006)
    sphHarmFit   = qball_model.fit(diffusionData)  
    shm_coeff    = sphHarmFit.shm_coeff
    
    # (2) get which slices in which gradients are needed to be resampled
    diffusionData_orig, bvalue_orig, gradientDirections_orig, baselineIndex_orig, options_orig = hardiIO.readHARDI(nrrdfilename_orig)
    gradientDirections_orig = np.squeeze(gradientDirections_orig)
    rows, cols, nSlices, nDirections = diffusionData_orig.shape
    
    # get the gradients and slices which suffer from slice-wise artifacts
    toBeResampled = np.zeros((nDirections, nSlices))
    
    # from DTIPrep - slice-wise intensity artifacts
    excluded = list()
    qcreportfilename = os.path.join(prepDir, 'DWI_65dir/%s_QCReport.txt' % (basename))
    fid = open(qcreportfilename, 'r')
    for line in fid:
        line = line.strip()
        if line.find('Slice-wise Check Artifacts:') >= 0:
            break
    for line in fid:    
        line = line.strip().split('\t')   
        if line[0].find('whole') < 0:
            continue    
        d = int(line[1])
        s = int(line[2])    
        toBeResampled[d,s] = 1 
        excluded.append(d)
    fid.close()    
    excluded = np.unique(np.array(excluded))
    
    # from DTIprep - interlace artifacts
    # see if  there are any other directions that were excluded due to interslice artifacts
    qcreportfilename = os.path.join(prepDir, 'DWI_65dir/%s_QCReport.txt' % (basename))
    dirIndex = dict() # mapping that maps the direction index in dtiprep output to the original index
    fid = open(qcreportfilename,'r')
    included = np.zeros((nDirections,))
    for line in fid:
        if line.find('QCIndex') >= 0 and line.find('Included Gradients:') >= 0:
            line = line.strip().split(' ')
            d    = int(line[2])
            included[d] = 1
            dirIndex[int(line[4])] = int(line[2])
    fid.close()
    included[baselineIndex_orig] = 1
    
    for d in range(nDirections):
        if included[d] == 0:
            if len( np.where(excluded == d)[0]) == 0: # was not excluded due to slice artifacts
                for s in range(nSlices):
                    toBeResampled[d, s] = 1
    
    # from within-gradient motion QC
    qcreportfilename = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCreport.txt' % (basename))
    fid = open(qcreportfilename,'r')
    for line in fid:
        if line.find('Excluded')>= 0:
            line = line.strip().split()
            dirind = int(line[0])
            nslices = int(line[2])
            for i in range(nslices):
                s = int(line[3+i])
                d = dirIndex[dirind]
                toBeResampled[d, s] = 1
    fid.close()
    
    # (3) start resampling
    diffusionData_resampled = copy.deepcopy(diffusionData_orig)
    for kk in range(nSlices):
        if np.sum(toBeResampled[:,kk]) == 0: # no need for resampling this slice along all directions
            continue
        
        for ind, gradDir in enumerate(gradientDirections_orig):
            
            if ind == baselineIndex_orig:
                continue
    
            if toBeResampled[ind,kk] == 0: # only resample the corrupted slices
                continue
            
            for ii in range(rows):
                for jj in range(cols):
                
                    print 'Resampling ii = %d, jj = %d, slice = %d, dir = %d ...' % (ii,jj,kk,ind) 
                
                    S0       = diffusionData_orig[ii,jj,kk,baselineIndex_orig]
                    shCoeffs = shm_coeff[ii,jj,kk,:]                            
                    
                    r, theta, phi = drecon.cart2sphere(gradDir[0], gradDir[1], gradDir[2])
                    
                    shBasis, m, n = drecon.real_sym_sh_basis(6,phi, theta)
                    shBasis       = shBasis.flatten()
                    
                    diffusionData_resampled[ii,jj,kk,ind]  = S0 * np.dot(shBasis, shCoeffs)
                    
    nrrd.write(nrrdfilename_rq, diffusionData_resampled, options_orig)

    # (4) redo the denoising part (Gradient wise) of DTIPrep
    #cmdStr = 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (nrrdfilename_rq, xmlfilename, os.path.join(curDir, 'DWI_65dir'))
    cmdStr = 'DWIRicianLMMSE %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f' % (nrrdfilename_rq, nrrdfilename_rq, FilteringRadius_x, FilteringRadius_y, FilteringRadius_z, EstimationRadius_x, EstimationRadius_y, EstimationRadius_z, NoOfIterations, minVoxelsFiltering, minVoxelsEstimation, histRes, minStd, maxStd, absVal)
    os.system(cmdStr)

    # fix the nrrd file (thickness and directions are the last dimension)
    hardiIO.fixNRRDfile(nrrdfilename_rq)
    
    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_rq, niifilename_rq, bvecsfilename_rq, bvalsfilename_rq)
    
    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_rq, bvecsfilename_rq, btablefilename_rq)
    
    # save src file
    hardiIO.nifti2src(niifilename_rq, btablefilename_rq, srcfilename_rq)

    end_time = time.time()
    
    print 'PerformResampleCorruptedSlicesInQspaceAndGradientDenoise: time elapsed = %f seconds ...' % (end_time - start_time)

"""
---------------------------------------------------------------------------------
STEP FIVE, I will follow HOMOR paper and fit model to all measurements
---------------------------------------------------------------------------------
OBJECTIVE:
    construct the reference volume for each data set using Qball-FMAM
    
Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) baseline and brain mask extraction
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    
TODO: 
    (1) construct the reference volume for each data set using Qball-FMAM
    (2) save nii, nrrd and src
"""
    
def ComputeModelBasedReference_FMAM(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    nrrdfilename              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.nrrd' % (basename) )    
    niifilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.nii' %(basename) )    
    bvecsfilename             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.bvecs' %(basename) )    
    bvalsfilename             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.bvals' %(basename) )   
    btablefilename            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_btable.txt' %(basename) )   
    srcfilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.src.gz' %(basename) )   
    
    nrrdfilename_fmam              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM.nrrd' % (basename) )    
    niifilename_fmam               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM.nii' %(basename) )    
    bvecsfilename_fmam             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM.bvecs' %(basename) )    
    bvalsfilename_fmam             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM.bvals' %(basename) )   
    btablefilename_fmam            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM_btable.txt' %(basename) )   
    srcfilename_fmam               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM.src.gz' %(basename) )   
        
    niiBrainMaskFilename    = os.path.join(prepDir, 'baseline/%s_baseline_brain_mask.nii' %(basename))
    brainMask               = nib.load(niiBrainMaskFilename).get_data()
    
    #(1) reconstruct the dODF based on regularized Qball based on the sequence from the previous QC steps
    # note the DTIPrep involved a denoising step
    
    diffusionData, bvalue, gradientDirections, baselineIndex, options = hardiIO.readHARDI(nrrdfilename)
    gradientDirections = np.squeeze(gradientDirections)
    bvals, bvecs = hardiIO.readbtable(btablefilename)
    gtable       = gradient_table(bvals, bvecs)

    qball_model  = drecon.QballModel(gtable, sh_order = 6, smooth = 0.006)
    sphHarmFit   = qball_model.fit(diffusionData, mask = brainMask)  
    shm_coeff    = sphHarmFit.shm_coeff
    
    rows, cols, nSlices, nDirections = diffusionData.shape
    
    diffusionData_resampled = np.zeros(diffusionData.shape)
    diffusionData_resampled[:,:,:,baselineIndex] = copy.deepcopy(diffusionData[:,:,:,baselineIndex])
    r, theta, phi = drecon.cart2sphere(gradientDirections[:,0], gradientDirections[:,1], gradientDirections[:,2])
    indices = range(nDirections)
    indices.remove(baselineIndex)
    
    shBasis, m, n = drecon.real_sym_sh_basis(6,phi, theta)
    
    for kk in range(nSlices):
        
        print 'Resampling slice = %d ...' % (kk) 
        
        S0 = np.zeros((rows,cols,nDirections))
        for ind in range(nDirections):                         
            S0[:,:,ind]   = diffusionData[:,:,kk,baselineIndex]
        
        shCoeffs = shm_coeff[:,:,kk,:] 
        nCoeffs  = shCoeffs.shape[-1]
        shCoeffs = np.reshape(shCoeffs, (rows*cols,nCoeffs))
        
        S = S0 * np.array(np.matrix(shBasis) * np.matrix(shCoeffs).T  ).T.reshape((rows, cols, nDirections))
        diffusionData_resampled[:,:,kk,indices] = S[:,:,indices]
                    
    nrrd.write(nrrdfilename_fmam, diffusionData_resampled, options)

    hardiIO.fixNRRDfile(nrrdfilename_fmam)
    
    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_fmam, niifilename_fmam, bvecsfilename_fmam, bvalsfilename_fmam)
    
    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_fmam, bvecsfilename_fmam, btablefilename_fmam)
    
    # save src file
    hardiIO.nifti2src(niifilename_fmam, btablefilename_fmam, srcfilename_fmam)
    
    end_time = time.time()
    
    print 'ComputeModelBasedReference_FMAM: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP SIX
---------------------------------------------------------------------------------

OBJECTIVE:
    correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already 
    been carried out, as such we can use standard interpolation

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) baseline and brain mask extraction
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) the reference volume for each data set using Qball-FMAM has been constructed
    
TODO: 
    (0) fix the dimension and thickness of the given nrrd files (if any)
    (1) prepare nifti files for mcflirt
    (2) perform the always-interpolate option to correct for subject motion using mcflirt without using masked sequence
    (3) back to nrrd format, nii and src
"""

def PerformModelBasedReferenceMotionCorrectionMCFLIRT12DOF(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    interpMethod  = 'trilinear'
    useBrainMask  = True # false if we have severe motion (> 5deg), otherwise try to avoid background noise when doing motion correction
    
    nrrdfilename              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.nrrd' % (basename) )    
    niifilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.nii' % (basename) )    
    nrrdfilename_fmam         = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_FMAM.nrrd' % (basename) )    
    
    bvalsfilename              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.bvals' % (basename) )    
    bvecsfilename              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ.bvecs' % (basename) )    
    
    nrrdfilename_MC              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF.nrrd' %(basename) )    
    niifilename_MC               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF.nii' %(basename) )    
    bvecsfilename_MC             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF.bvecs' %(basename) )    
    bvalsfilename_MC             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF.bvals' %(basename) )    
    btablefilename_MC            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_btable.txt' %(basename) )    
    srcfilename_MC               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF.src.gz' %(basename) )    
    reportfilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_QCreport.txt' %(basename) )    
    
    print nrrdfilename
    diffusionData, options = nrrd.read(nrrdfilename)
    fmamData, _            = nrrd.read(nrrdfilename_fmam)
    
    if useBrainMask:
        niiBrainMaskFilename         = os.path.join(prepDir, 'baseline/%s_baseline_brain_mask.nii' %(basename))
        brainMask                    = nib.load(niiBrainMaskFilename).get_data()
        
        diffusionData                = hardiQCUtils.brainMasking(diffusionData, brainMask)
        fmamData                     = hardiQCUtils.brainMasking(fmamData, brainMask)
    
    affine                             = nib.load(niifilename).get_affine()
    rows, cols, nSlices, nDirections   = diffusionData.shape

    mcOutDir                     = os.path.join(prepDir, 'DWI_65dir/MODELBASED_MCFLIRT12DOF_OUT')
    if os.path.exists(mcOutDir):
        shutil.rmtree(mcOutDir)
    os.mkdir(mcOutDir)                 
    
    mcParamFolder             = os.path.join(mcOutDir, '%s.mat' %(basename))
    parfilename               = os.path.join(mcOutDir, '%s.par' %(basename))
    
    if os.path.exists(mcParamFolder):
        shutil.rmtree(mcParamFolder)
    os.mkdir(mcParamFolder)          
                
    parMat                           = np.zeros((nDirections, 12))# to be save in the original par file
    for d in range(nDirections):
        curData = np.zeros((rows, cols, nSlices, 2)) # refvol then the current gradient (including the baseline)
        curData[:,:,:,0] = copy.deepcopy(fmamData[:,:,:,d])
        curData[:,:,:,1] = copy.deepcopy(diffusionData[:,:,:,d])
    
        curOutDir = os.path.join(mcOutDir, 'mcflirt_out_' + str(d).zfill(4) )       
        if os.path.exists(curOutDir):
            shutil.rmtree(curOutDir)
        os.mkdir(curOutDir)
        
        cur_niifilename_for_mcflirt   = os.path.join(curOutDir, '%s_in.nii' %(basename))
        cur_niifilename_corrected     = os.path.join(curOutDir, '%s.nii' %(basename))
        cur_mcParamFolder             = os.path.join(curOutDir, '%s.mat' %(basename))
        cur_parfilename               = os.path.join(curOutDir, '%s.par' %(basename))
        
        hardiIO.save2nii(cur_niifilename_for_mcflirt, curData, affine)
        
        # now go ahead and do motion correction for the current gradient
        hardiQCUtils.RunMCFLIRT_12DOF(cur_niifilename_for_mcflirt, cur_niifilename_corrected, interpMethod)
        
        # copy the current transformation file to the original mat folder
        cmdStr = 'cp -R -v %s %s' % (cur_mcParamFolder + '/MAT_0001', mcParamFolder + '/MAT_' + str(d).zfill(4))
        os.system(cmdStr)
        
        # save the par information too
        ind = -1
        csvfile = csv.reader(open(cur_parfilename),delimiter=' ')
        for row in csvfile:
            ind   = ind + 1
            
            if ind == 1: # the transformation of the second volume
                if len(row) == 1:
                    row = row[0].split()
                    parMat[d,0]  = float(row[0]) # in rad
                    parMat[d,1]    = float(row[1]) # in rad
                    parMat[d,2]    = float(row[2]) # in rad
                    
                    parMat[d,3]  = float(row[3]) # in mm
                    parMat[d,4]    = float(row[4]) # in mm
                    parMat[d,5]    = float(row[5]) # in mm
                    
                    parMat[d,6]  = float(row[6]) # in mm
                    parMat[d,7]    = float(row[7]) # in mm
                    parMat[d,8]    = float(row[8]) # in mm
                    
                    parMat[d,9]  = float(row[9]) # in mm
                    parMat[d,10]    = float(row[10]) # in mm
                    parMat[d,11]    = float(row[11]) # in mm
                else:    
                    parMat[d,0]  = float(row[0]) # in rad
                    parMat[d,1]    = float(row[2]) # in rad
                    parMat[d,2]    = float(row[4]) # in rad
                    
                    parMat[d,3]  = float(row[6]) # in mm
                    parMat[d,4]    = float(row[8]) # in mm
                    parMat[d,5]    = float(row[10]) # in mm
                    
                    parMat[d,6]  = float(row[12]) # in mm
                    parMat[d,7]    = float(row[14]) # in mm
                    parMat[d,8]    = float(row[16]) # in mm
                    
                    parMat[d,9]  = float(row[18]) # in mm
                    parMat[d,10]    = float(row[20]) # in mm
                    parMat[d,11]    = float(row[22]) # in mm
                    
                break
            
        # call twice to make sure it is deleted    
        shutil.rmtree(curOutDir, ignore_errors=True) 
        shutil.rmtree(curOutDir, ignore_errors=True) 
        
        cmdStr = 'rm -rf %s' %(curOutDir)
        os.system(cmdStr)
        print cmdStr
        
    hardiIO.saveMatrixToCSV(parMat, parfilename, delim='\t')
        
    # apply the transformation found by mcflirt on the original data to get an unmasked corrected sequence (for noise removal afterwards too)
    correctedData = hardiQCUtils.ApplyMCFLIRT(niifilename, mcParamFolder, interpMethod, os.path.join(mcOutDir, 'mcflirt'))
    
    # update the btable, get the fmam btable            
    bvals, bvecs    = hardiIO.readbvalsbvecs(bvalsfilename, bvecsfilename)
    
    baselineIndex = np.where(bvals == 0)[0][0]
    bvals_corrected, bvecs_corrected = hardiQCUtils.ReorientBmatrix_mcflirt_12DOF(bvals, bvecs,  mcParamFolder, parfilename, baselineIndex)
    
    motionQuantification = hardiQCUtils.QuantifyMotion_12DOF(parfilename, mcParamFolder, baselineIndex)
    
    # writing out the motion quantification results 
    hardiQCUtils.WriteMotionCorrectionQCReport_12DOF(motionQuantification, reportfilename)
     
    # update the btable in the nrrd options then start the saving cycle 
    options = hardiIO.updateNrrdOptions(options, bvecs_corrected)

    # save as nrrd with the save options as the original nrrd file
    nrrd.write( nrrdfilename_MC, correctedData, options)

    # save to nii
    hardiIO.convertToNIFTI(nrrdfilename_MC, niifilename_MC, bvecsfilename_MC, bvalsfilename_MC)

    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_MC, bvecsfilename_MC, btablefilename_MC)                
    
    # save src file
    hardiIO.nifti2src(niifilename_MC, btablefilename_MC, srcfilename_MC)
    
    
    end_time = time.time()
    
    print 'PerformModelBasedReferenceMotionCorrectionMCFLIRT12DOF: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP SEVEN
---------------------------------------------------------------------------------

OBJECTIVE:
    brain mask the motion corrected datasets

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) brain mask has been extracted
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) motion correction has been performed
    
TODO: 
    (1) brain mask the motion corrected data 
"""   
def BrainMaskModelBasedMotionCorrectedDWI(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    niiBrainMaskFilename         = os.path.join(prepDir, 'baseline/%s_baseline_brain_mask.nii' %(basename))
    nrrdfilename_MC              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF.nrrd' %(basename) )    
    
    nrrdfilename_MC_masked              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked.nrrd' %(basename) )    
    niifilename_MC_masked               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked.nii' %(basename) )    
    bvecsfilename_MC_masked             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked.bvecs' %(basename) )    
    bvalsfilename_MC_masked             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked.bvals' %(basename) )    
    btablefilename_MC_masked            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_btable.txt' %(basename) )    
    srcfilename_MC_masked               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked.src.gz' %(basename) )    
    
    brainMask               = nib.load(niiBrainMaskFilename).get_data()
    correctedData, options  = nrrd.read(nrrdfilename_MC)
    correctedDataMasked     = hardiQCUtils.brainMasking(correctedData, brainMask)
    
    # save as nrrd with the save options as the original nrrd file
    nrrd.write( nrrdfilename_MC_masked, correctedDataMasked, options)
    
    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_MC_masked, niifilename_MC_masked, bvecsfilename_MC_masked, bvalsfilename_MC_masked)
    
    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_MC_masked, bvecsfilename_MC_masked, btablefilename_MC_masked)
        
    # save src file
    hardiIO.nifti2src(niifilename_MC_masked, btablefilename_MC_masked, srcfilename_MC_masked)

    end_time = time.time()
    
    print 'BrainMaskModelBasedMotionCorrectedDWI: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP EIGHT
---------------------------------------------------------------------------------

OBJECTIVE:
    compute the bias field based on the baseline and apply it on the other gradients

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) brain mask has been extracted
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) motion correction has been performed
    (6) brain masking of the motion corrected sequences
    
TODO: 
    (1) compute the bias field
    (2) apply bias field correction on all the gradients, do this on the masked and unmasked MC data
"""
def PerformBiasFieldCorrection(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    niiBrainMaskFilename         = os.path.join(prepDir, 'baseline/%s_baseline_brain_mask.nii' %(basename))
    baselineNiifilename          = os.path.join(prepDir, 'baseline/%s_baseline.nii' %(basename))
    
    
    baselineCorrectedNiifilename = os.path.join(prepDir, 'baseline/%s_baseline_BFC.nii' %(basename))
    biasfieldNiifilename         = os.path.join(prepDir, 'baseline/%s_biasField.nii' %(basename))
    
    nrrdfilename_MC_masked       = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked.nrrd' %(basename) )    
                
    nrrdfilename_MC_masked_BFC              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC.nrrd' %(basename) )    
    niifilename_MC_masked_BFC               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC.nii' %(basename) )    
    bvecsfilename_MC_masked_BFC             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC.bvecs' %(basename) )    
    bvalsfilename_MC_masked_BFC             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC.bvals' %(basename) )    
    btablefilename_MC_masked_BFC            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_btable.txt' %(basename) )    
    srcfilename_MC_masked_BFC               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC.src.gz' %(basename) )    
    
    cmdStr = 'DWIBiasFieldCorrection --image-dimensionality 3 --input-image %s --mask-image %s --shrink-factor 1  --output [%s, %s] --bspline-fitting [1000,3]' % (baselineNiifilename, niiBrainMaskFilename, baselineCorrectedNiifilename, biasfieldNiifilename)
    os.system(cmdStr)
    
    biasfield               = nib.load(biasfieldNiifilename).get_data()
        
    # apply on masked data
    nrrdData, options  = nrrd.read(nrrdfilename_MC_masked)
    nrrdDataCorrected  = hardiQCUtils.ApplyBiasFieldCorrection(nrrdData, biasfield)
    
    # save as nrrd with the save options as the original nrrd file
    nrrd.write( nrrdfilename_MC_masked_BFC, nrrdDataCorrected, options)
    
    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_MC_masked_BFC, niifilename_MC_masked_BFC, bvecsfilename_MC_masked_BFC, bvalsfilename_MC_masked_BFC)
    
    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_MC_masked_BFC, bvecsfilename_MC_masked_BFC, btablefilename_MC_masked_BFC)
        
    # save src file
    hardiIO.nifti2src(niifilename_MC_masked_BFC, btablefilename_MC_masked_BFC, srcfilename_MC_masked_BFC)
        
    end_time = time.time()
    
    print 'PerformBiasFieldCorrection: time elapsed = %f seconds ...' % (end_time - start_time)
    
"""
---------------------------------------------------------------------------------
STEP NINE
---------------------------------------------------------------------------------

OBJECTIVE:
    jointly denoise the bias field corrected data

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) brain mask has been extracted
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) motion correction has been performed
    (6) brain masking of the motion corrected sequences
    (7) bias field correction
    
TODO: 
    (1) jointly denoise the bias field corrected data
"""    
def PerformJointLMMSEDenoising(nrrdfilename, prepDir):
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    
    # filter parameters
    NoOfNeighboringDirections = 6 # when using all, too smooth diffusion
    EstimationRadius_x        = 2 
    EstimationRadius_y        = 2 
    EstimationRadius_z        = 2
    FilteringRadius_x         = 2  
    FilteringRadius_y         = 2  
    FilteringRadius_z         = 2  
    
    nrrdfilename_MC_masked_BFC                     = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC.nrrd' %(basename) )    
                    
    nrrdfilename_MC_masked_BFC_JLMMSE              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.nrrd'%(basename) )    
    niifilename_MC_masked_BFC_JLMMSE               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.nii' %(basename) )    
    bvecsfilename_MC_masked_BFC_JLMMSE             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.bvecs' %(basename) )    
    bvalsfilename_MC_masked_BFC_JLMMSE             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.bvals'%(basename) )    
    btablefilename_MC_masked_BFC_JLMMSE            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE_btable.txt'%(basename) )    
    srcfilename_MC_masked_BFC_JLMMSE               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.src.gz' %(basename) )    
        
    cmdStr = 'DWIJointRicianLMMSE %s %s %d %d %d %d %d %d %d' % (nrrdfilename_MC_masked_BFC, nrrdfilename_MC_masked_BFC_JLMMSE, FilteringRadius_x, FilteringRadius_y, FilteringRadius_z, EstimationRadius_x, EstimationRadius_y, EstimationRadius_z, NoOfNeighboringDirections)
    os.system(cmdStr)

    # apply on masked data
    hardiIO.fixNRRDfile(nrrdfilename_MC_masked_BFC_JLMMSE)
                
    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_MC_masked_BFC_JLMMSE, niifilename_MC_masked_BFC_JLMMSE, bvecsfilename_MC_masked_BFC_JLMMSE, bvalsfilename_MC_masked_BFC_JLMMSE)
    
    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_MC_masked_BFC_JLMMSE, bvecsfilename_MC_masked_BFC_JLMMSE, btablefilename_MC_masked_BFC_JLMMSE)
        
    # save src file
    hardiIO.nifti2src(niifilename_MC_masked_BFC_JLMMSE, btablefilename_MC_masked_BFC_JLMMSE, srcfilename_MC_masked_BFC_JLMMSE)
        
    end_time = time.time()
    
    print 'PerformJointLMMSEDenoising: time elapsed = %f seconds ...' % (end_time - start_time)
 
"""
---------------------------------------------------------------------------------
STEP TEN
---------------------------------------------------------------------------------

OBJECTIVE:
    reconstruction in subject space
    reconstruct the fiber ODF of the clean data 

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) brain mask has been extracted
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) motion correction has been performed
    (6) brain masking of the motion corrected sequences
    (7) bias field correction
    (8) joint denoising has been performed
    
TODO: 
    (1) reconstruct voxel-wise fiber ODF plus fiber orientations (fODF peaks)
    (2) save to fib file for visual debugging using dsi_studio
"""
   
def ReconstructFODFs(nrrdfilename, prepDir):
    
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    from dipy.reconst.peaks import peaks_from_model
    from dipy.data import get_sphere
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    index        = basename.find('_DWI_65dir')
    if index >= 0:
        phan_name = basename[:index]
    else:
        phan_name = basename
    
    nrrdfilename              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.nrrd'%(basename) )    
    niifilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.nii' %(basename) )    
    bvecsfilename             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.bvecs' %(basename) )    
    bvalsfilename             = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.bvals'%(basename) )    
    btablefilename            = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE_btable.txt'%(basename) )    
    srcfilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.src.gz' %(basename) )    
    
    niiBrainMaskFilename      = os.path.join(prepDir, 'baseline/%s_baseline_brain_mask.nii' %(basename))
                
    outDir = os.path.join(prepDir, 'fodfs' )
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.mkdir(outDir)
    
    # (1) read the current dataset
    print "Read the current dataset ..."                
    diffusionData, mask, affine, gtable, parcellation = hardiIO.readDataset(niifilename, niiBrainMaskFilename, btablefilename)

    # (2) compute the single fiber response function for the deconvolution process
    print "Compute the single fiber response function"
    #response, ratio = auto_response(gtable, diffusionData, roi_center=center, roi_radius=radius, fa_thr=0.7)
    response, ratio = hardiQCUtils.single_fiber_response(diffusionData, mask, gtable, fa_thr = 0.7)

    # now start the deconvolution process
    # (3) construct the constrained spherical deconvolution model given the gtable and the fiber response function
    print "Construct the constrained spherical deconvolution model given the gtable and the fiber response function"
    csd_model = ConstrainedSphericalDeconvModel(gtable, response)
    
    sphere    = get_sphere('symmetric724')
    
    print "Fitting the fODF model and finding fiber orientations"
    """
    csd_peaks : PeaksAndMetrics
        An object with ``gfa``, ``peak_directions``, ``peak_values``,
        ``peak_indices``, ``odf``, ``shm_coeffs`` as attributes
    """
    csd_peaks  = peaks_from_model(model=csd_model,
                                 data=diffusionData,
                                 sphere=sphere,
                                 relative_peak_threshold=.4,
                                 min_separation_angle=20,
                                 mask = mask,
                                 sh_order=8,
                                 return_odf=True,
                                 return_sh=True,
                                 gfa_thr = 0,
                                 normalize_peaks=False,
                                 parallel=True)
                                 
    #It is common with CSA ODFs to produce negative values, we can remove those using np.clip
    csd_peaks.odf = np.clip(csd_peaks.odf, 0, np.max(csd_peaks.odf, -1)[..., None])
    
    print "Computing QA ..."
    qa = hardiQCUtils.computeQAFromPeaks(csd_peaks, sphere, mask)
    csd_peaks.qa = copy.deepcopy(qa)
    
    print "saving fib file for visualization ..."

    voxel_size  = np.array([affine[2,2],affine[2,2],affine[2,2]]) 
    dimension   = mask.shape
    fibfilename = os.path.join(outDir, '%s_csd.fib' %(phan_name))
    hardiIO.saveToFIB(fibfilename, dimension, voxel_size, csd_peaks, sphere, diffusionData[:,:,:,0])
    
    print "saving reconstruction results ..."
    filename = os.path.join(outDir, '%s_gfa_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.gfa)
    
    filename = os.path.join(outDir, '%s_qa_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.qa)
    
    options = dict()
    options['encoding'] = 'raw'
    filename = os.path.join(outDir, '%s_odf_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.odf,options)
    
    # taring with relative path
    curWDir = os.getcwd()
    print 'Current directory: %s' % (curWDir)
    print 'Moving to: %s' % (outDir)
    os.chdir(outDir)
    #filename_tgz = os.path.join(outDir, '%s_odf_csd.tar.gz' %(phan_name))
    filename_tgz = '%s_odf_csd.tar.gz' %(phan_name)
    filename     = '%s_odf_csd.nrrd' %(phan_name)
    print filename
    print filename_tgz
    cmdStr = 'GZIP=-9 tar cvzf  %s %s' % (filename_tgz, filename) 
    os.system(cmdStr)
    cmdStr = 'rm -rf %s' % (filename)
    os.system(cmdStr)
    os.chdir(curWDir)
    print 'Back to: %s' % (os.getcwd())
    
    filename = os.path.join(outDir, '%s_peak_dirs_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.peak_dirs)
    
    filename = os.path.join(outDir, '%s_peak_indices_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.peak_indices)
    
    filename = os.path.join(outDir, '%s_peak_values_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.peak_values)
    
    filename = os.path.join(outDir, '%s_shm_coeff_csd.nrrd' %(phan_name))
    nrrd.write(filename, csd_peaks.shm_coeff)
    
    end_time = time.time()
    
    print 'ReconstructFODFs: time elapsed = %f seconds ...' % (end_time - start_time)
 
"""
---------------------------------------------------------------------------------
STEP TEN
---------------------------------------------------------------------------------

OBJECTIVE:
    full brain tractographing in subject space

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) brain mask has been extracted
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) motion correction has been performed
    (6) brain masking of the motion corrected sequences
    (7) bias field correction
    (8) joint denoising has been performed
    (9) reconstruct voxel-wise fiber ODF plus fiber orientations (fODF peaks)
    
TODO: 
    (1) whole brain tractography
    (2) save files for visualization
"""   
def PerformFullBrainTractography(nrrdfilename, prepDir):
    
    from dipy.tracking.eudx import EuDX
    from dipy.data import get_sphere
    
    start_time = time.time()
    
    _ , filename = ntpath.split(nrrdfilename)
    basename     = filename[:-len('.nrrd')]
    index        = basename.find('_DWI_65dir')
    if index >= 0:
        phan_name = basename[:index]
    else:
        phan_name = basename
    
    nrrdfilename              = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.nrrd'%(basename) )    
    niifilename               = os.path.join(prepDir, 'DWI_65dir/%s_QCed_WithinGradientMotionQCed_ResampleQ_MODELBASED_MCFLIRT12DOF_masked_BFC_JLMMSE.nii' %(basename) )    
    
    odfsDir = os.path.join(prepDir, 'fodfs' )

    # sphere used to reconstruct the f/d-odfs
    sphere        = get_sphere('symmetric724')

    
    outDir  = os.path.join(prepDir, 'tractography' )
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.mkdir(outDir)
    
    pvfilename = os.path.join(odfsDir, '%s_peak_indices_csd.nrrd' %(phan_name))    
    pifilename = os.path.join(odfsDir, '%s_peak_values_csd.nrrd' %(phan_name))

    streamlinesfilename         = os.path.join(outDir,  '%s_streamlines_csd.vtk'%(phan_name))
    streamlinesfilename_txt     = os.path.join(outDir,  '%s_streamlines_csd.txt'%(phan_name))
    
    vox_dims = hardiIO.get_vox_dims(niifilename)
    print 'vox_dims = ' + repr(vox_dims)
    
    data_dims = hardiIO.get_data_dims(niifilename)
    print 'data_dims = ' + repr(data_dims)

    affine    = hardiIO.get_affine(niifilename)
    print 'affine = ', affine  
    
    #qa, _           = nrrd.read(qafilename)
    peak_values, _  = nrrd.read(pvfilename)
    peak_indices, _ = nrrd.read(pifilename)    
    
    data                        = dict()
    
    # get the streamlines
    eu = EuDX(peak_values,
              peak_indices,
              seeds=1000000,
              odf_vertices=sphere.vertices,
              ang_thr=30.,
              a_low=0.2, 
              step_sz=0.2,
              affine=affine) # reconstructed streamlines are in the physical coordinates
              
    streamlines = [streamline for streamline in eu]
    data['streamlines']         = copy.deepcopy(streamlines)
    print 'Number of tracks detected: ' + repr(len(streamlines))
                    
    hardiIO.saveRawStreamlinesToVTK(streamlinesfilename, streamlines)
    hardiIO.saveStreamlinesForDSIstudio(streamlinesfilename_txt, streamlines, affine)
                
    end_time = time.time()
    
    print 'PerformFullBrainTractography: time elapsed = %f seconds ...' % (end_time - start_time)
 