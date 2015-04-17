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
parser.add_argument("-p","--protocol", type=str, help="Path to the XML Protocol you want to use", default = "PROTOCOLS/.default_HARDIPrep.xml")
parser.add_argument("-j", type=int, help="Number of cores you want to use")
args = parser.parse_args()

if os.path.isdir(args.path):
    files_list=[]
    for root, dirs, files in os.walk(args.path):  
        for i in files:  
            files_list.append(os.path.join(root, i))
else:
    files_list = [args.path]  

if args.protocol:
    xmlFile = args.protocol
    stepTree = etree.parse(xmlFile)
    groot = stepTree.getroot()
    step0 = groot[0]

    
for nrrdfilename in files_list:
    outDir, _ = os.path.split(nrrdfilename)
    """ STEP 0 """
    prepDir = qc.PrepareQCsession(nrrdfilename, outDir)

