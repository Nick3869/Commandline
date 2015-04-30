# -*- coding: utf-8 -*-

"""

@author: Nicolas Fanjat
 Scientific Computing and Imaging Institute
 University of Utah
 02/17/2015
 
"""

import sys
import argparse
from lxml import etree
import hardi.qc as qc
import hardi.io as io
import hardi.nrrd
import os

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the directory (or single file) that contains your data")
parser.add_argument("-p","--protocol", type=str, help="Path to the XML Protocol you want to use", default = ".HARDIPrepDefault.xml")
parser.add_argument("-j", type=int, help="Number of cores you want to use")
args = parser.parse_args()

if os.path.isdir(args.path):
    files_list=[]
    for root, dirs, files in os.walk(args.path):  
        for i in files:  
            files_list.append(os.path.join(root, i))
else:
    files_list = [args.path]  


xmlFile = args.protocol
stepTree = etree.parse(xmlFile)
groot = stepTree.getroot()
step0 = groot[0]
step1 = groot[1]


for nrrdfilename in files_list:
    outDir, _ = os.path.split(nrrdfilename)
    """ STEP 0 """
    prepDir = qc.PrepareQCsession(nrrdfilename, outDir)
    
    """ STEP 1 """
    xmlfilename = step1[0].text
    hardiQC.RunDTIPrepStage(nrrdfilename, prepDir, xmlfilename)
    
    """ STEP 2 """
    hardiQC.PerformWithinGradientMotionQC(nrrdfilename, prepDir)
    
    """ STEP 3 """
    hardiQC.ExtractBaselineAndBrainMask(nrrdfilename, prepDir)
    
    """ STEP 4 """
    optionsf4 = list()
    optionsf4.append(qc.num(step1[3].text))
    optionsf4.append(qc.num(step1[4].text))
    optionsf4.append(qc.num(step1[5].text))
    optionsf4.append(qc.num(step1[6].text))
    optionsf4.append(qc.num(step1[7].text))
    optionsf4.append(qc.num(step1[8].text))
    optionsf4.append(qc.num(step1[9].text))
    optionsf4.append(qc.num(step1[10].text))
    optionsf4.append(qc.num(step1[11].text))
    optionsf4.append(qc.num(step1[12].text))
    optionsf4.append(qc.num(step1[13].text))
    optionsf4.append(qc.num(step1[14].text))
    optionsf4.append(qc.num(step1[15].text))
    hardiQC.PerformResampleCorruptedSlicesInQspaceAndGradientDenoise(nrrdfilename, prepDir, optionsf4)
    
    """ STEP 5 """
    hardiQC.ComputeModelBasedReference_FMAM(nrrdfilename, prepDir)
    
    """ STEP 6 """
    optionsmc = list()
    optionsmc.append(step1[1].text)
    optionsmc.append(step1[2].text)
    hardiQC.PerformModelBasedReferenceMotionCorrectionMCFLIRT12DOF(nrrdfilename, prepDir, optionsmc)
    
    """ STEP 7 """
    hardiQC.BrainMaskModelBasedMotionCorrectedDWI(nrrdfilename, prepDir)
    
    """ STEP 8 """
    optionsb = list()
    optionsb.append(qc.num(step1[16].text))
    optionsb.append(qc.num(step1[17].text))
    optionsb.append(step1[18].text)
    hardiQC.PerformBiasFieldCorrection(nrrdfilename, prepDir, optionsb)
    
    """ STEP 9 """
    optionsf9 = list()
    optionsf9.append(qc.num(step1[19].text))
    optionsf9.append(qc.num(step1[20].text))
    optionsf9.append(qc.num(step1[21].text))
    optionsf9.append(qc.num(step1[22].text))
    optionsf9.append(qc.num(step1[23].text))
    optionsf9.append(qc.num(step1[24].text))
    optionsf9.append(qc.num(step1[25].text))
    hardiQC.PerformJointLMMSEDenoising(nrrdfilename, prepDir, optionsf9)
    
    """ STEP 10 """
    hardiQC.ReconstructFODFs(nrrdfilename, prepDir)
    
    """ STEP 11 """
    hardiQC.PerformFullBrainTractography(nrrdfilename, prepDir)

