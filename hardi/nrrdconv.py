# Author : Nicolas Fanjat
# Scientific Computing ang Imaging Institute
# University of Utah
# 12/04/2014

import numpy as np
import nrrd
import os
import nibabel as nib
import sys

name = sys.argv[1]



filename = name
ext = os.path.splitext(name)[1]


if ((ext == '.nrrd') or (ext == '.nhdr')):
    context.write("Processing")
    data, header = nrrd.read(filename)
    
    basename = os.path.basename(x)
    context.write(basename)
    basename = os.path.splitext(basename)[0]+".nii.gz"

    filename = os.path.join(self.output_directory.name, basename)
    
    header['space directions'].remove('none')
    
    origin = np.asarray(header['space origin'])
    origin = np.reshape(origin,(3,1))
    
    directions = header['space directions']
    directions = np.asarray(directions)
    directions = np.reshape(directions,(3,3))
    
    affine = np.hstack((directions,origin))
    affine = np.vstack((affine,np.array([0.,0.,0.,1.])))
    
    context.write("Writing")
    context.write(filename)
    new_image = nib.Nifti1Image(data,affine)
    nib.save(new_image, filename)
    context.write('')
    
context.write('All done !')
        
        
    
    
