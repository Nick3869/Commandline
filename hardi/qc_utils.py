# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 10:38:34 2015

@author: Shireen Elhabian

QC utilities to be called from qc
"""

from __future__ import division
import numpy as np
import os
import time
import shutil
import copy
import ntpath

import nibabel as nib

import hardi.io as hardiIO
import hardi.nrrd as nrrd # I modified this package to fix some bugs

import nipype.interfaces.fsl as fsl
import nibabel as nib
import numpy as np

import csv
"""
OBJECTIVE:
    quantify fast bulk motion within each gradient to exclude those having intra-scan
    motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)
    
    here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded
"""

def DetectWithinGradientMotion(nrrdData, baselineIndex, bvalue):
    
    # settings based on Benner et al
    #baseline_threshold = 100 
    diffusion_coeff    = 1e-2
    baseline_fraction  = 0.7 # f_1 in the paper
    
    rows, cols, nSlices, nDirections = nrrdData.shape
    baseline_threshold = nrrdData[:,:,:,baselineIndex].mean()
        
    # getting the baseline r_score first
    r_baseline = np.zeros((nSlices,))
    for s in range(nSlices):
        Bslice        = copy.deepcopy(nrrdData[:,:,s,baselineIndex]).flatten()
        r_baseline[s] = np.where(Bslice > baseline_threshold)[0].shape[0]
    
    # detect signal drop out
    nMotionCorrupted = np.zeros((nDirections,))
    slice_numbers    = list()
    for v in range(nDirections):                
        if v == baselineIndex:
            continue
      
        curThreshold = baseline_threshold * np.exp(-1 * bvalue * diffusion_coeff)
        
        whichSlices = list()
        for s in range(nSlices):
            Cslice = copy.deepcopy(nrrdData[:,:,s,v]).flatten()
            r      = np.where(Cslice > curThreshold)[0].shape[0]
            
            if r <= 0.05 * (rows * cols): # ignore this slice
                continue
            
            if r < (baseline_fraction * r_baseline[s]):
                nMotionCorrupted[v] = nMotionCorrupted[v] + 1
                whichSlices.append(s)
                
        slice_numbers.append(whichSlices)
        
    return nMotionCorrupted, slice_numbers

def WriteWithinGradientMotionQCReport(reportfilename, nMotionCorrupted, slice_numbers, baselineIndex):
    
    nDirections = nMotionCorrupted.shape[0]
    
    # writing the report
    fid = open(reportfilename, 'w')
    fid.write('\n\nWithin-gradient motion QC Report:\n');
    fid.write('----------------------------------\n');
    fid.write('Gradient Index \t\t Status \t\t number of corrupted slices \t\t slice numbers \n');
    fid.write('-------------- \t\t ------ \t\t -------------------------- \t\t ------------- \n');
    
    ind = -1
    nExcluded = 0
    for v in range(nDirections):
        if v == baselineIndex:
            fid.write('\t%d \t\t Included \t\t\t \t%d \t\t\t\n'%(v, nMotionCorrupted[v]));
            continue

        ind = ind + 1
        whichSlices = slice_numbers[ind]
        
        if nMotionCorrupted[v] == 0:
            fid.write('\t%d \t\t Included \t\t\t \t%d \t\t\t'%(v, nMotionCorrupted[v]));
        else:
            nExcluded = nExcluded + 1
            fid.write('\t%d \t\t Excluded \t\t\t \t%d \t\t\t'%(v,nMotionCorrupted[v]));
        
        for s in whichSlices:
            fid.write(' %d'%(s))
        fid.write('\n')
    fid.close()
    
    return nExcluded
    
def ConstructWithinGradientMotionCorrectedData(nrrdData, gradientDirections, nExcluded, nMotionCorrupted):
    
    rows, cols, nSlices, nDirections = nrrdData.shape
    nDirections_corrected   = nDirections - nExcluded
    correctedData           = np.zeros((rows,cols,nSlices, nDirections_corrected))
    gradientDirections_new  = np.zeros((nDirections_corrected,3))    

    ind = -1            
    for v in range(nDirections):
        if nMotionCorrupted[v] == 0:
            ind = ind+1
            gradientDirections_new[ind,:] = copy.deepcopy(gradientDirections[v,:])
            correctedData[:,:,:,ind]                 = copy.deepcopy(nrrdData[:,:,:,v])
            
    return correctedData, gradientDirections_new


"""
Inter-Gradioent Motion Correction
"""

def RunMCFLIRT(niifilename_for_mcflirt, niifilename_corrected, interpMethod):
    
    # now go ahead and do motion correction
    if interpMethod is 'trilinear':
        cmdStr = 'mcflirt -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 6 -stages 4 -verbose 1 -stats -mats -plots -report'
       
    if interpMethod is 'nn':
        cmdStr = 'mcflirt -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 6 -nn_final -stages 4 -verbose 1 -stats -mats -plots -report'
            
    if interpMethod is 'sinc':
        cmdStr = 'mcflirt -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 6 -sinc_final -stages 4 -verbose 1 -stats -mats -plots -report'
       
    if interpMethod is 'spline':
        cmdStr = 'mcflirt -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 6 -spline_final -stages 4 -verbose 1 -stats -mats -plots -report'
        
    os.system(cmdStr)
    
def RunMCFLIRT_12DOF(niifilename_for_mcflirt, niifilename_corrected, interpMethod):
    
    # this version uses mcflirt2 which dumps the 12 dof in the par file
    # now go ahead and do motion correction
    if interpMethod is 'trilinear':
        cmdStr = 'mcflirt2 -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 12 -stages 4 -verbose 1 -stats -mats -plots -report'
       
    if interpMethod is 'nn':
        cmdStr = 'mcflirt2 -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 12 -nn_final -stages 4 -verbose 1 -stats -mats -plots -report'
            
    if interpMethod is 'sinc':
        cmdStr = 'mcflirt2 -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 12 -sinc_final -stages 4 -verbose 1 -stats -mats -plots -report'
       
    if interpMethod is 'spline':
        cmdStr = 'mcflirt2 -in ' + niifilename_for_mcflirt[0:-4] + ' -out ' + niifilename_corrected[0:-4] + ' -cost normmi -refvol 0 -dof 12 -spline_final -stages 4 -verbose 1 -stats -mats -plots -report'
        
    os.system(cmdStr)

def ApplyMCFLIRT(niifilename, mcParamFolder, interpMethod, out_prefix):
    
    niiData = nib.load(niifilename).get_data()
    affine  = nib.load(niifilename).get_affine()
    
    rows, cols, nSlices, nDirections = niiData.shape
    correctedData = np.zeros((rows, cols, nSlices, nDirections))
    
    for d in range(nDirections):
        cur_niifilename_in  = out_prefix + '_' + str(d).zfill(4) + '_in.nii'
        cur_niifilename_out = out_prefix + '_' + str(d).zfill(4) + '_out.nii'
        matfilename         = os.path.abspath(os.path.join(mcParamFolder, 'MAT_%s' %( str(d).zfill(4))))
        
        cur_niiData     = copy.deepcopy(niiData[:,:,:,d])
        hardiIO.save2nii(cur_niifilename_in, cur_niiData, affine)
        
        cmdStr = 'applyMCFLIRT %s %s %s %s' % (cur_niifilename_in, cur_niifilename_out, matfilename, interpMethod)
        os.system(cmdStr)
        
        correctedData[:,:,:,d] = nib.load(cur_niifilename_out).get_data()
        
        cmdStr = 'rm -rf %s' % (cur_niifilename_in)
        os.system(cmdStr)
        
        cmdStr = 'rm -rf %s' % (cur_niifilename_out)
        os.system(cmdStr)
        
    return correctedData 
    
def ReorientBmatrix_mcflirt(bvals, bvecs, mcParamFolder):
    
    bvals_corrected = copy.deepcopy(bvals)
    bvecs_corrected = copy.deepcopy(bvecs)
    
    nDirections     = bvals.shape[0]
    
    # Reorientation the diffusion gradients and quantify motion just to make sure that mcflirt detected the corrupted ones                                
    for ind in range(nDirections):        
        if bvals[ind] == 0:
            continue
        
        # getting axis and angle from the rotation matrix directly
        matfilename = os.path.abspath(os.path.join(mcParamFolder, 'MAT_%s' %( str(ind).zfill(4))))
        
        fid = open ( matfilename  , 'r')
        table = [row.strip().split('\t') for row in fid]                
        R     = np.zeros((3,3))
        kk    = 0
        for row in table:
            row = np.array(row)
            row = row[0].split()
            
            R[kk,0] = float(row[0])
            R[kk,1] = float(row[1])
            R[kk,2] = float(row[2])
            kk = kk + 1
            if kk == 3:
                break
        fid.close()
        
        R = np.matrix(R)
        
        # re-orient the current direction
        gradDir = np.matrix(bvecs[ind,:])
        gradDir = R * gradDir.transpose()
        gradDir = np.array(gradDir.transpose())
        gradDir = gradDir/np.linalg.norm(gradDir)
        bvecs_corrected[ind,:] = copy.deepcopy(gradDir)
    
    return bvals_corrected, bvecs_corrected


def ReorientBmatrix_mcflirt_12DOF(bvals, bvecs, mcParamFolder, parfilename, baselineIndex):
    
    bvals_corrected = copy.deepcopy(bvals)
    bvecs_corrected = copy.deepcopy(bvecs)
    
    nDirections     = bvals.shape[0]
    
    # first get the scale and skew values which will be used to extract the rotational
    # component from the affine matrix
    
    # scaling
    Sx = list()
    Sy = list()
    Sz = list()
    
    # skewing
    skew_a = list()
    skew_b = list()
    skew_c = list()
    
    
    ind = -1
    csvfile = csv.reader(open(parfilename),delimiter=' ')
    for row in csvfile:
        ind   = ind + 1
        
        if ind == baselineIndex:            
            Sx.append(0.0)
            Sy.append(0.0)
            Sz.append(0.0)
            
            skew_a.append(0.0)
            skew_b.append(0.0)
            skew_c.append(0.0)
            
            continue
        
        if len(row) == 1:
            row = row[0].split()            
            sx    = float(row[6])
            sy    = float(row[7])
            sz    = float(row[8])
            
            sa    = float(row[9])
            sb    = float(row[10])
            sc    = float(row[11])            
        else:            
            sx    = float(row[12])
            sy    = float(row[14])
            sz    = float(row[16])
            
            sa    = float(row[18])
            sb    = float(row[20])
            sc    = float(row[22])
                
        Sx.append(sx)
        Sy.append(sy)
        Sz.append(sz)
        
        skew_a.append(sa)
        skew_b.append(sb)
        skew_c.append(sc)
            
    # Reorientation the diffusion gradients and quantify motion just to make sure that mcflirt detected the corrupted ones                                
    for ind in range(nDirections):        
        if bvals[ind] == 0:
            continue
        
        # getting axis and angle from the rotation matrix directly
        matfilename = os.path.abspath(os.path.join(mcParamFolder, 'MAT_%s' %( str(ind).zfill(4))))
        
        fid = open ( matfilename  , 'r')
        table = [row.strip().split('\t') for row in fid]                
        A     = np.zeros((3,3))
        kk    = 0
        for row in table:
            row = np.array(row)
            row = row[0].split()
            
            A[kk,0] = float(row[0])
            A[kk,1] = float(row[1])
            A[kk,2] = float(row[2])
            kk = kk + 1
            if kk == 3:
                break
        fid.close()
        
        # scale matrix
        S = np.zeros((3,3))
        S[0,0] = Sx[ind]
        S[1,1] = Sy[ind]
        S[2,2] = Sz[ind]
        
        # skew matrix
        Sw = np.eye(3)        
        Sw[0,1] = skew_a[ind]
        Sw[0,2] = skew_b[ind]
        Sw[1,2] = skew_c[ind]
        
        R = np.matrix(A) * np.matrix(np.linalg.inv(S)) * np.matrix(np.linalg.inv(Sw))
        
        # re-orient the current direction
        gradDir = np.matrix(bvecs[ind,:])
        gradDir = R * gradDir.transpose()
        gradDir = np.array(gradDir.transpose())
        gradDir = gradDir/np.linalg.norm(gradDir)
        bvecs_corrected[ind,:] = copy.deepcopy(gradDir)
    
    return bvals_corrected, bvecs_corrected
    
def QuantifyMotion(parfilename, mcParamFolder, baselineIndex = 0):
    
    # Reorientation the diffusion gradients and quantify motion just to make sure that mcflirt detected the corrupted ones                                
    normT         = list()
    rotationAngleFromR = list()
    rotationAxisFromR = list()
    
    ind = -1
    csvfile = csv.reader(open(parfilename),delimiter=' ')
    for row in csvfile:
        ind   = ind + 1
        
        if ind == baselineIndex:
            normT.append(0)
            rotationAxisFromR.append(np.array([0, 0,1]))
            rotationAngleFromR.append(0)
            continue
        
        if len(row) == 1:
            row = row[0].split()
            tx    = float(row[3]) # in mm
            ty    = float(row[4]) # in mm
            tz    = float(row[5]) # in mm
        else:            
            tx    = float(row[6]) # in mm
            ty    = float(row[8]) # in mm
            tz    = float(row[10]) # in mm
        
        normT.append(np.linalg.norm(np.array([tx, ty, tz])))
            
        # getting axis and angle from the rotation matrix directly
        matfilename = os.path.abspath(os.path.join(mcParamFolder, 'MAT_%s' %( str(ind).zfill(4))))
        
        fid = open ( matfilename  , 'r')
        table = [row.strip().split('\t') for row in fid]                
        R     = np.zeros((3,3))
        kk    = 0
        for row in table:
            row = np.array(row)
            row = row[0].split()
            
            R[kk,0] = float(row[0])
            R[kk,1] = float(row[1])
            R[kk,2] = float(row[2])
            kk = kk + 1
            if kk == 3:
                break
        fid.close()
        
        rotationAngleFromR.append(np.rad2deg(np.arccos((np.trace(R)-1)/2)))
        vals, vecs = np.linalg.eig(R)
        rotationAxisFromR.append(np.real(vecs[:,2]))  
        
    normT              = np.array(normT)
    rotationAngleFromR = np.array(rotationAngleFromR)
    rotationAxisFromR  = np.array(rotationAxisFromR)
    
    motionQuantification = dict()
    motionQuantification['normT']              = normT
    motionQuantification['rotationAngleFromR'] = rotationAngleFromR
    motionQuantification['rotationAxisFromR']  = rotationAxisFromR
    
    return motionQuantification
    
def QuantifyMotion_12DOF(parfilename, mcParamFolder, baselineIndex = 0):
    
    # Reorientation the diffusion gradients and quantify motion just to make sure that mcflirt detected the corrupted ones                                
    normT         = list()

    rotationAngleFromR = list()
    rotationAxisFromR = list()
    
    # translation    
    Tx = list()
    Ty = list()
    Tz = list()
    
    # scaling
    Sx = list()
    Sy = list()
    Sz = list()
    
    # skewing
    skew_a = list()
    skew_b = list()
    skew_c = list()
    
    
    ind = -1
    csvfile = csv.reader(open(parfilename),delimiter=' ')
    for row in csvfile:
        ind   = ind + 1
        
        if ind == baselineIndex:
            normT.append(0)
            rotationAxisFromR.append(np.array([0, 0,1]))
            rotationAngleFromR.append(0)
            
            Tx.append(0.0)
            Ty.append(0.0)
            Tz.append(0.0)
            
            Sx.append(1.0)
            Sy.append(1.0)
            Sz.append(1.0)
            
            skew_a.append(0.0)
            skew_b.append(0.0)
            skew_c.append(0.0)
            
            continue
        
        if len(row) == 1:
            row = row[0].split()
            tx    = float(row[3]) # in mm
            ty    = float(row[4]) # in mm
            tz    = float(row[5]) # in mm
            
            sx    = float(row[6])
            sy    = float(row[7])
            sz    = float(row[8])
            
            sa    = float(row[9])
            sb    = float(row[10])
            sc    = float(row[11])
            
        else:            
            tx    = float(row[6]) # in mm
            ty    = float(row[8]) # in mm
            tz    = float(row[10]) # in mm
            
            sx    = float(row[12])
            sy    = float(row[14])
            sz    = float(row[16])
            
            sa    = float(row[18])
            sb    = float(row[20])
            sc    = float(row[22])
        
        normT.append(np.linalg.norm(np.array([tx, ty, tz])))
        
        Tx.append(tx)
        Ty.append(ty)
        Tz.append(tz)
        
        Sx.append(sx)
        Sy.append(sy)
        Sz.append(sz)
        
        skew_a.append(sa)
        skew_b.append(sb)
        skew_c.append(sc)
            
        # getting axis and angle from the rotation matrix directly
        matfilename = os.path.abspath(os.path.join(mcParamFolder, 'MAT_%s' %( str(ind).zfill(4))))
        
        fid = open ( matfilename  , 'r')
        table = [row.strip().split('\t') for row in fid]                
        A     = np.zeros((3,3))
        kk    = 0
        for row in table:
            row = np.array(row)
            row = row[0].split()
            
            A[kk,0] = float(row[0])
            A[kk,1] = float(row[1])
            A[kk,2] = float(row[2])
            kk = kk + 1
            if kk == 3:
                break
        fid.close()
        
        # scale matrix
        S = np.zeros((3,3))
        S[0,0] = Sx[ind]
        S[1,1] = Sy[ind]
        S[2,2] = Sz[ind]
        
        # skew matrix
        Sw = np.eye(3)        
        Sw[0,1] = skew_a[ind]
        Sw[0,2] = skew_b[ind]
        Sw[1,2] = skew_c[ind]
        
        R = np.matrix(A) * np.matrix(np.linalg.inv(S)) * np.matrix(np.linalg.inv(Sw))
        
        tr = (np.trace(R)-1)/2
        if tr < -1:
            tr = -1
        if tr > 1:
            tr = 1
            
        rotationAngleFromR.append(np.rad2deg(np.arccos(tr)))
        vals, vecs = np.linalg.eig(R)
        rotationAxisFromR.append(np.real(vecs[:,2]))  
        
    normT              = np.array(normT)
    rotationAngleFromR = np.array(rotationAngleFromR)
    rotationAxisFromR  = np.array(rotationAxisFromR)
    
    Tx = np.array(Tx)
    Ty = np.array(Ty)
    Tz = np.array(Tz)
    
    Sx = np.array(Sx)
    Sy = np.array(Sy)
    Sz = np.array(Sz)
    
    skew_a = np.array(skew_a)
    skew_b = np.array(skew_b)
    skew_c = np.array(skew_c)
    
    
    motionQuantification = dict()
    motionQuantification['normT']              = normT
    motionQuantification['rotationAngleFromR'] = rotationAngleFromR
    motionQuantification['rotationAxisFromR']  = rotationAxisFromR
    
    motionQuantification['Tx']  = Tx
    motionQuantification['Ty']  = Ty
    motionQuantification['Tz']  = Tz
    
    motionQuantification['Sx']  = Sx
    motionQuantification['Sy']  = Sy
    motionQuantification['Sz']  = Sz
    
    motionQuantification['skew_a']  = skew_a
    motionQuantification['skew_b']  = skew_b
    motionQuantification['skew_c']  = skew_c
    
    
    return motionQuantification

def WriteMotionCorrectionQCReport(motionQuantification, reportfilename, iter_no = None):
    
    normT              = motionQuantification['normT']              
    rotationAngleFromR = motionQuantification['rotationAngleFromR'] 
    rotationAxisFromR  = motionQuantification['rotationAxisFromR']  
    
    # writing the report
    if iter_no == 0 or iter_no is None:
        fid = open(reportfilename, 'w')
        fid.write('\n\nBetween-gradient motion QC Report (Based on FSL-MCFLIRT):\n');
        fid.write('------------------------------------------------------------\n');
        fid.write('------------------------------------------------------------\n');
    else:
        fid = open(reportfilename, 'a')
        
    if iter_no is not None:
        if type(iter_no) is str:
            fid.write('\n\n------------------------------------------------------ Iteration Number : %s ------------------------------------------------------\n' % (iter_no));
        else:
            fid.write('\n\n------------------------------------------------------ Iteration Number : %d ------------------------------------------------------\n' % (iter_no+1));
        fid.write('----------------------------------------------------------------------------------------------------------------------------------\n\n\n');
        
    fid.write('--------------------------------------------------------- Average Translation = %f +/- %f mm --------------------------------------------------\n' %(np.mean(normT),np.std(normT)));
    fid.write('--------------------------------------------------------- Average Rotation    = %f +/- %f degrees ---------------------------------------------\n\n' %(np.mean(rotationAngleFromR), np.std(rotationAngleFromR)));
    
    fid.write('Gradient Index \t\t Translation Magnitude (mm) \t\t Rotation Angle (degrees) \t\t Rotation Axis \n');
    fid.write('-------------- \t\t -------------------------- \t\t ------------------------ \t\t ------------- \n');
    fid.write('-------------- \t\t -------------------------- \t\t ------------------------ \t\t ------------- \n');
    
    nDirections = normT.shape[0]
    
    for v in range(nDirections):
        fid.write('\t%d \t\t\t %f \t\t\t\t %f \t\t\t [ %f, \t%f, \t%f]\n'%(v, normT[v], rotationAngleFromR[v], rotationAxisFromR[v,0], rotationAxisFromR[v,1], rotationAxisFromR[v,2]));
    fid.close()
    
    
def WriteMotionCorrectionQCReport_12DOF(motionQuantification, reportfilename, iter_no = None):
    
    normT              = motionQuantification['normT']              
    rotationAngleFromR = motionQuantification['rotationAngleFromR'] 
    rotationAxisFromR  = motionQuantification['rotationAxisFromR']  
    
    Tx = motionQuantification['Tx']  
    Ty = motionQuantification['Ty']  
    Tz = motionQuantification['Tz']  
    
    Sx = motionQuantification['Sx']
    Sy = motionQuantification['Sy']  
    Sz = motionQuantification['Sz']  
    
    skew_a = motionQuantification['skew_a']
    skew_b = motionQuantification['skew_b']  
    skew_c = motionQuantification['skew_c']  
    
    # writing the report
    if iter_no == 0 or iter_no is None:
        fid = open(reportfilename, 'w')
        fid.write('\n\nBetween-gradient motion QC Report (Based on FSL-MCFLIRT):\n');
        fid.write('------------------------------------------------------------\n');
        fid.write('------------------------------------------------------------\n');
    else:
        fid = open(reportfilename, 'a')
        
    if iter_no is not None:
        if type(iter_no) is str:
            fid.write('\n\n------------------------------------------------------ Iteration Number : %s ------------------------------------------------------\n' % (iter_no));
        else:
            fid.write('\n\n------------------------------------------------------ Iteration Number : %d ------------------------------------------------------\n' % (iter_no+1));
        fid.write('----------------------------------------------------------------------------------------------------------------------------------\n\n\n');
        
    fid.write('--------------------------------------------------------- Average Translation = %f +/- %f mm --------------------------------------------------\n' %(np.mean(normT),np.std(normT)));
    fid.write('--------------------------------------------------------- Average Rotation    = %f +/- %f degrees ---------------------------------------------\n\n' %(np.mean(rotationAngleFromR), np.std(rotationAngleFromR)));
    
    fid.write('--------------------------------------------------------- Average Scale in x direction = %f +/- %f  --------------------------------------------------\n' %(np.mean(Sx), np.std(Sx)));
    fid.write('--------------------------------------------------------- Average Scale in y direction = %f +/- %f  --------------------------------------------------\n' %(np.mean(Sy), np.std(Sy)));
    fid.write('--------------------------------------------------------- Average Scale in z direction = %f +/- %f  --------------------------------------------------\n\n' %(np.mean(Sz), np.std(Sz)));
    
    fid.write('--------------------------------------------------------- Average Skew A = %f +/- %f  --------------------------------------------------\n' %(np.mean(skew_a), np.std(skew_a)));
    fid.write('--------------------------------------------------------- Average Skew B = %f +/- %f  --------------------------------------------------\n' %(np.mean(skew_b), np.std(skew_b)));
    fid.write('--------------------------------------------------------- Average Skew C = %f +/- %f  --------------------------------------------------\n\n\n' %(np.mean(skew_c), np.std(skew_c)));
    
    fid.write('Gradient Index \t\t Translation Magnitude (mm) \t\t Rotation Angle (degrees) \t\t\t\t Rotation Axis \t\t\t Tx (mm) \t Ty  \t\t Tz  \t\t Sx  \t\t Sy  \t\t Sz  \t\t Sa \t\t Sb \t\t Sc \n');
    fid.write('-------------- \t\t -------------------------- \t\t ------------------------ \t\t\t\t ------------- \t\t\t ------- \t --- \t\t --- \t\t --- \t\t --- \t\t --- \t\t ---\t\t ---\t\t ---\t \n');
    fid.write('-------------- \t\t -------------------------- \t\t ------------------------ \t\t\t\t ------------- \t\t\t ------- \t --- \t\t --- \t\t --- \t\t --- \t\t --- \t\t ---\t\t ---\t\t ---\t \n');
    
    nDirections = normT.shape[0]
    
    for v in range(nDirections):
        fid.write('\t%d \t\t\t %f \t\t\t\t %f \t\t\t [ %f, \t%f, \t%f] \t%f \t%f \t%f \t%f \t%f \t%f \t%f \t%f \t%f \n'%(v, normT[v], rotationAngleFromR[v], rotationAxisFromR[v,0], rotationAxisFromR[v,1], rotationAxisFromR[v,2], Tx[v], Ty[v] , Tz[v], Sx[v], Sy[v], Sz[v], skew_a[v], skew_b[v], skew_c[v] ));
    fid.close()

def ApplyBiasFieldCorrection(nrrdData, biasfield):
    
    rows,cols,nSlices,nDirections = nrrdData.shape
    nrrdDataCorrected = np.zeros((rows,cols,nSlices,nDirections))
    
    for d in range(nDirections):
        nrrdDataCorrected[:,:,:,d] = nrrdData[:,:,:,d] / (biasfield + 1e-10)
        
    return nrrdDataCorrected

"""
Given the baseline of a hardi sequence, we want to extract the brain region
"""
def extractBrainRegion(baselineNiifilename, niiBrainFilename='', niiBrainMaskFilename=''):
    # start skull stripping to get the brain region using FSL's bet tool
    if niiBrainFilename == '':
        f = baselineNiifilename.find('.')
        niiBrainFilename = baselineNiifilename[0:f] + '_brain.nii.gz'
        niiBrainMaskFilename = baselineNiifilename[0:f] + '_brain_mask.nii.gz'
    
    fslBet                 = fsl.BET()
    fslBet.inputs.in_file  = baselineNiifilename
    fslBet.inputs.out_file = niiBrainFilename
    fslBet.inputs.mask              = True
    fslBet.inputs.mesh              = True
    fslBet.inputs.outline           = True
    #fslBet.inputs.reduce_bias       = True # takes alot of time to clean the neck 
    fslBet.inputs.vertical_gradient = 0.1 # vertical gradient in fractional intensity threshold (-1->1); default=0; positive values give larger brain outline at bottom, smaller at top 
    fslBet.inputs.frac              = 0.2 # fractional intensity threshold (0->1); default=0.5; smaller values give larger brain outline estimates 
    betInterface                    = fslBet.run()
    #cmd: bet CURRENT_RAW_2012.nii.gz mask.nii -o -m -f 0.2 -g 0.1 -R -S -B 
    
    niiMask   = nib.load(niiBrainMaskFilename)
    brainMask = niiMask.get_data()
    
    return brainMask; 

def brainMasking(nrrdData_, brainMask):
    
    import copy
    nrrdData = copy.deepcopy(nrrdData_)
        
    rows, cols, nSlices, nDirections  = nrrdData.shape
    nrrdDataMasked                    = np.zeros((rows, cols, nSlices, nDirections))
    
    for d in range(0,nDirections):
        for s in range(0,nSlices):
            #print 'Masking: direction no. ' + repr(d) + ' , slice no. ' + repr(s) + ' ...'
            nrrdDataMasked[:,:,s,d] = nrrdData[:,:,s,d] * brainMask[:,:,s]
                        
    return nrrdDataMasked;
    
def brainMaskingVolume(nrrdData_, brainMask):
    
    import copy
    nrrdData = copy.deepcopy(nrrdData_)
        
    rows, cols, nSlices  = nrrdData.shape
    nrrdDataMasked                    = np.zeros((rows, cols, nSlices))
    
    for s in range(0,nSlices):
        #print 'Masking: direction no. ' + repr(d) + ' , slice no. ' + repr(s) + ' ...'
        nrrdDataMasked[:,:,s] = nrrdData[:,:,s] * brainMask[:,:,s]
                    
    return nrrdDataMasked;
    

def single_fiber_response(diffusionData, mask, gtable, fa_thr = 0.7):
    from dipy.reconst.dti import TensorModel, fractional_anisotropy
    
    ten    = TensorModel(gtable)
    tenfit = ten.fit(diffusionData, mask=mask)
    
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    
    indices = np.where(FA > fa_thr)
    lambdas = tenfit.evals[indices][:, :2]
    
    S0s     = diffusionData[indices][:, np.nonzero(gtable.b0s_mask)[0]]
    S0      = np.mean(S0s)
    l01     = np.mean(lambdas, axis=0)
    evals   = np.array([l01[0], l01[1], l01[1]])
    
    response = (evals, S0)
    ratio    = evals[1]/evals[0]
    
    return response, ratio
    
def computeQAFromPeaks(csd_peaks, sphere, mask):
     
    maxFibers  = csd_peaks.peak_indices.shape[-1]
    nvertices  = sphere.vertices.shape[0]
    global_max = csd_peaks.peak_values.max()
    
    rows, cols, nSlices, nVertices = csd_peaks.odf.shape
    QA_field  = np.zeros((rows,cols, nSlices,maxFibers))
    
    for ii in range(rows):
        for jj in range(cols):
            for kk in range(nSlices):
                minODF = csd_peaks.odf[ii,jj,kk,:].min()
                for f in range(maxFibers):
                    if mask[ii,jj,kk] > 0:
                        QA_field[ii,jj,kk,f] = (csd_peaks.peak_values[ii,jj,kk,f] - minODF) / (global_max+1e-10)
    return QA_field
