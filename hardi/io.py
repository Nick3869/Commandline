# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 09:54:45 2015

@author: Shireen Elhabian

objective:
    utitlity functions for io to be used in the processing pipeline

"""
import sys
import nrrd
import numpy as np
import csv
import pyevtk.hl as evtk
import pickle
import fnmatch
import os
import scipy.io as sio
import copy

import nibabel as nib
from dipy.core.gradients import gradient_table

"""
Given the nrrdfilename, this function reads nrrd data plus 
the gradient directions, indicating which direction correspond to b-value =0, 
i.e. baseline, if any...
"""
def readHARDI(nrrdfilename):
    
    nrrdData, options = nrrd.read(nrrdfilename)
    
    if options['kinds'][0] == 'list': # as saved by slicer
        nrrdData = np.transpose(nrrdData, (1,2,3,0))
        options['kinds'] = ['space','space','space','list']
        
    """
    Getting the gradient directions
    """
    keyValuePairs = options['keyvaluepairs']
    bvalue        = float(keyValuePairs['DWMRI_b-value'])
    
    # now lets go over each direction and get its value
    gradientDirections = list() #np.zeros(shape=(nDirs,3), dtype = 'float')
    baselineIndex = -1
    for d in range(0,max(nrrdData.shape)):
        keyname = 'DWMRI_gradient_' + str(d).zfill(4)
        #print keyname
        if keyValuePairs.has_key(keyname):
            keyval  = keyValuePairs[keyname]
            keyval  = keyval.split()    
            gradDir = np.zeros((1,3))
            gradDir[0,0] = np.float(keyval[0])
            gradDir[0,1] = np.float(keyval[1])
            gradDir[0,2] = np.float(keyval[2])    
            if np.sum(gradDir) == 0:
                baselineIndex = d
            else:
                gradDir = gradDir/np.linalg.norm(gradDir) # normalize the gradient direction
            gradientDirections.append(gradDir)
        else:
            break
        
    gradientDirections = np.array(gradientDirections)
    
    return nrrdData, bvalue, gradientDirections, baselineIndex, options

def readDataset(niifilename, niiBrainMaskFilename, btablefilename,  parcellationfilename = None):  
     
    # load the masked diffusion dataset
    diffusionData = nib.load(niifilename).get_data()
    affine        = nib.load(niifilename).get_affine()
    
    # load the brain mask
    mask    = nib.load(niiBrainMaskFilename).get_data()
    
    rows, cols, nSlices, nDirections = diffusionData.shape
    
    bvals, bvecs = readbtable(btablefilename)
    gtable       = gradient_table(bvals, bvecs)
    
    if parcellationfilename != None:
        #parcellation = nib.load(parcellationfilename).get_data()
        parcellation,_ = nrrd.read(parcellationfilename)
    
        if parcellation.shape[2] != nSlices:  # for the second phantom (unc_res)
            parcellation = parcellation[:,:,parcellation.shape[2]-nSlices:]        
        parcellation = np.squeeze(parcellation)
    else:
        parcellation = None
    
    return diffusionData, mask, affine, gtable, parcellation
    

def convertToNIFTI(nrrdfilename, niifilename, bvecsfilename, bvalsfilename):
    cmdStr = 'DWIConvert --inputVolume %s --conversionMode NrrdToFSL --outputVolume %s --outputBVectors %s --outputBValues %s' % (nrrdfilename, niifilename, bvecsfilename, bvalsfilename)
    os.system(cmdStr)
    
def save2nii(niifilename, data, affine=None):

    import nibabel as nib

    if affine == None:
        niiData = nib.Nifti1Image(data, np.eye(4))
    else:
        niiData = nib.Nifti1Image(data, affine)
    
    print niiData.get_header()['dim']
    print niiData.get_data_dtype()
    
    nib.save(niiData, niifilename)

def readbvalsbvecs(bvalsfilename, bvecsfilename):
    
    bvals = list()
    fid = open(bvalsfilename, 'r')
    for line in fid:
        bvals.append(float(line.strip()))
    fid.close()
    
    nDirections = len(bvals)
    bvals       = np.array(bvals)
    
    bvecs = np.zeros((nDirections, 3))
    fid = open(bvecsfilename, 'r')
    ii = 0
    for line in fid:
        line = line.strip()
        line = line.split()
        bvecs[ii,0] = float(line[0])
        bvecs[ii,1] = float(line[1])
        bvecs[ii,2] = float(line[2])
        ii = ii + 1
    fid.close()
    
    return bvals, bvecs
    
def bvecsbvals2btable(bvalsfilename, bvecsfilename, btablefilename):
    bvals, bvecs = readbvalsbvecs(bvalsfilename, bvecsfilename)
    writebtable(btablefilename, bvals, bvecs)

def readbtable(btablefilename):
    
    bvals = list()
    fid = open(btablefilename, 'r')
    for line in fid:
        line = line.strip()
        line = line.split()
        bvals.append(float(line[0]))
    fid.close()
    
    nDirections = len(bvals)
    bvals       = np.array(bvals)
    
    bvecs = np.zeros((nDirections, 3))
    fid = open(btablefilename, 'r')
    ii = 0
    for line in fid:
        line = line.strip()
        line = line.split()
        bvecs[ii,0] = float(line[1])
        bvecs[ii,1] = float(line[2])
        bvecs[ii,2] = float(line[3])
        ii = ii + 1
    fid.close()
    
    return bvals, bvecs
    
def writebtable(btablefilename, bvals, bvecs):
    
    nDirections = bvecs.shape[0]
    
    fid = open(btablefilename, 'w')
    for ii in range(nDirections):
        fid.write('%f\t%f\t%f\t%f\n' % (bvals[ii], bvecs[ii,0], bvecs[ii,1], bvecs[ii,2]))
    fid.close()
    
def nifti2src(niifilename, btablefilename, srcfilename): 
    cmdStr = 'dsi_studio --action=src --source=%s --b_table=%s --output=%s' % (niifilename, btablefilename, srcfilename)
    os.system(cmdStr)


def fixNRRDfile(nrrdfilename, encoding='gzip'):
    # data not saved by slicer ITK (from Clement) would have NAN in the thickness
    # note: slicer save number of directions as the first dimension
    
    # save to nrrd
    nrrdData, options = nrrd.read(nrrdfilename)
        
    if options['kinds'][0] == 'list' or options['kinds'][0] == 'vector': # as saved by slicer
        nrrdData = np.transpose(nrrdData, (1,2,3,0))
        options['kinds'] = ['space','space','space','list']
        
        if type(options['space directions'][0]) is str:
            options['space directions'] = [options['space directions'][1], options['space directions'][2], options['space directions'][3], 'none']
        else:
            options['space directions'] = [options['space directions'][0], options['space directions'][1], options['space directions'][2], 'none']
    
    options['thicknesses'] = [abs(options['space directions'][0][0]), abs(options['space directions'][1][1]), abs(options['space directions'][2][2]), 'NaN']
    options['sizes']       = list(nrrdData.shape)
    
    options['encoding'] = encoding
    nrrd.write( nrrdfilename, nrrdData, options)
    
def updateNrrdOptions(options, new_bvecs):
    
    options['sizes'][-1] = new_bvecs.shape[0]
    bval                 = options['keyvaluepairs']['DWMRI_b-value']
    
    keyValuePairs = dict()
    keyValuePairs['DWMRI_b-value'] = bval
    keyValuePairs['modality'] = 'DWMRI'
    
    nDirections = new_bvecs.shape[0]    
    for d in range(nDirections):
        keyname = 'DWMRI_gradient_' + str(d).zfill(4)
        
        gradDir = np.zeros((1,3))
        gradDir[0,0] = new_bvecs[d,0]
        gradDir[0,1] = new_bvecs[d,1]
        gradDir[0,2] = new_bvecs[d,2]
        if np.linalg.norm(gradDir) > 0:
            gradDir = gradDir/np.linalg.norm(gradDir) # normalize the gradient direction
                
        keyValuePairs[keyname] = '%f %f %f' % (gradDir[0,0], gradDir[0,1], gradDir[0,2])
    
    options['keyvaluepairs'] = keyValuePairs
    return options
    
def extractAndSaveBaselineToNRRD(nrrdfilename, baselinenrrdfilename):
    
    nrrdData, bvalue, gradientDirections, baselineIndex,_ = readHARDI(nrrdfilename)
                
    baseline = nrrdData[:,:,:,baselineIndex]

    # save to nrrd
    _, options = nrrd.read(nrrdfilename)
    
    options['keyvaluepairs']    = []
    
    if options.has_key('centerings'):
        options['centerings']       = [options['centerings'][0], options['centerings'][1], options['centerings'][2]]
    
    options['dimension']        = 3
    
    options['kinds']            = ['space', 'space', 'space']
    
    if options.has_key('thicknesses'):
        options['thicknesses']    = []        
        
    if options.has_key('space directions'):    
        options['space directions'] = [options['space directions'][0], options['space directions'][1], options['space directions'][2]]
        options['thicknesses'] = [abs(options['space directions'][0][0]), abs(options['space directions'][1][1]), abs(options['space directions'][2][2])]
        
    #if options.has_key('thicknesses'):        
    #    options['thicknesses']      = [options['thicknesses'][0], options['thicknesses'][1], options['thicknesses'][2]]
    
    options['sizes']            = list(baseline.shape)
    nrrd.write( baselinenrrdfilename, baseline, options)
    
def saveMatrixToCSV(A,filename, delim=', '):
    fid = open(filename, 'w')
    for row in A:
        fid.write(delim.join(map(str,row)) + '\n')
    fid.close()
    
def saveToFIB(fibfilename, dimension, voxel_size, csd_peaks, sphere, baseline):
    
    rows, cols, nSlices = dimension
    maxFibers           = csd_peaks.peak_indices.shape[-1]
    
    outData = dict()
    outData['dimension']  = np.array([rows, cols, nSlices]).reshape((1,3))
    outData['voxel_size'] = voxel_size.reshape((1,3))
    outData['gfa']        = np.ravel(csd_peaks.gfa.T)
    outData['baseline']   = np.ravel(baseline.T)
    
    for f in range(maxFibers):
        fa    = csd_peaks.qa[:,:,:,f]
        index = csd_peaks.peak_indices[:,:,:,f]
        
        outData['fa%d'%(f)]    = np.ravel(fa.T)
        outData['index%d'%(f)] = np.ravel(index.T)
        
    outData['odf_vertices'] = sphere.vertices.T
    outData['odf_faces']    = sphere.faces.T
    
    N          = 0
    for kk in range(nSlices):
        for jj in range(cols):
            for ii in range(rows):
                if csd_peaks.qa[ii,jj,kk,0] > 0.0:
                    N = N + 1
                    
    nSamples   = sphere.vertices.shape[0]
    odf        = np.zeros((nSamples/2, N ))
    ind        = -1                        
    for kk in range(nSlices):
        for jj in range(cols):
            for ii in range(rows):
                if csd_peaks.qa[ii,jj,kk,0] > 0.0:  
                    ind = ind+ 1
                    curODF = csd_peaks.odf[ii,jj,kk,0:nSamples/2] 
                    odf[:,ind] = curODF
    outData['odfs']    = odf 
    
    sio.savemat(fibfilename, mdict=outData, format='4')
    os.system(('mv %s.mat %s' %(fibfilename,fibfilename)))

def get_vox_dims(volume):
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.get_header()
    voxdims = hdr.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]

def get_data_dims(volume):
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.get_header()
    datadims = hdr.get_data_shape()
    return [int(datadims[0]), int(datadims[1]), int(datadims[2])]

def get_affine(volume):
    import nibabel as nb
    nii = nb.load(volume)
    return nii.get_affine()

def saveRawStreamlinesToVTK(vtkfilename, streamlines):
    # streamlines is a list of np.array, number of entries in the list is
    # the number of tracts while each entry is a matrix Nx3 holding the 3d locations of the points 
    # belonging to a specific tracts
    
    # getting the 3d points and the track label for each point
    x           = []
    y           = []
    z           = []
    nLines      = 0

    for curStream in streamlines:
        nLines = nLines + curStream.shape[0] - 1
        
        for curPt in curStream:
            x.append(curPt[0])
            y.append(curPt[1])
            z.append(curPt[2])
    
    x           = np.array(x)
    y           = np.array(y)
    z           = np.array(z)
    nPoints     = x.shape[0]

    # opening the file for writing ...    
    fid = open(vtkfilename, 'w')
    print fid
    
    # heading info ...   
    fid.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
    
    # writing the points    
    fid.write("POINTS %d float\n" % (nPoints))
    for ii in range(nPoints):
        fid.write("%f %f %f\n" % (x[ii],y[ii],z[ii]))
    
    # writing the lines
    fid.write("\nLINES %d %d \n" % (nLines, 3*nLines))
    
    lineID  = 2
    pointID = -1
    for curStream in streamlines:
        #lineID     = lineID + 1
        curNPoints = curStream.shape[0]
        for ii in range(curNPoints-1):
            pointID = pointID + 1
            fid.write("%d %d %d\n" % (lineID, pointID, pointID+1))
        pointID = pointID + 1
        
    fid.close()

def saveStreamlinesForDSIstudio(txtfilename, streamlines, affine):
    
    fid = open(txtfilename ,'w')
    for xyz in streamlines:
        for pt in xyz:
            pt[0] = (pt[0] + affine[0,3])/2.0
            pt[1] = (pt[1] + affine[1,3])/2.0
            pt[2] = (pt[2] - affine[2,3])/2.0
            fid.write('%f %f %f ' % (pt[0],pt[1], pt[2]))
        fid.write('\n')
    fid.close()