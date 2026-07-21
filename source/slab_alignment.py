"""
This script is intended to be used at the command line and will apply a precomputed registration transform to a set of sequential microscopy (.npz) images. The main output from this script is the 3D reconstruction of the transformed images, saved as a .vtk file.
"""

import numpy as np
import torch
import os
import sys
import argparse
import matplotlib.pyplot as plt

def main():
    """
    Command line arguments
    ----------------------
    dataset : str
        The name of the dataset. This will included in several of the output file names.
    target_files : list of str
        List of low-res files in Anterior to Posterior order containing image data
    -tform_file : str
        The .npz file generated from atlas registration which contains all necessary transformation data
    -atlas_files : str or list of str
        List of 3 allen atlas .vtk files or 1 .tiff file
    -outdir : str
        The directory where all output files should be saved
    -epath : str
        Default - \'\'; Local path to the emlddmm library. If no argument is supplied, the library will be cloned locally from its GitHub repository
    -dslice : float
        Default: 500.0; Thickness of each slab (in um) of this dataset. This should be the same value as was used in registration.
    -dI : float
        Default: 50.0; The resolution (um) of the atlas that was used in registration, even if it differs from the atlases provided here
    -dsample : float
        Default: 50.0; The desired output resolution (um)
    -zm : bool
        Default - False; If True (i.e. the flag is passed), use the zero mean convention when loading input data. If this argument was passed during registration, pass it again here.
    -debug : bool
        Default - False; If True (i.e. the flag is passed), run the script in DEBUG mode
    """

    # Used as an argument to init affine matrix, A
    parser = argparse.ArgumentParser()
    
    # Subparser for a new brain
    parser.add_argument('dataset', type = str, help = 'The name of the dataset. This will included in several of the output file names.')
    parser.add_argument('target_files', type = str, nargs='*', help = 'List of low-res files in Anterior to Posterior order containing image data')
    parser.add_argument('-tform_file', type = str, required=True, help = 'The .npz file generated from atlas registration which contains all necessary transformation data')
    parser.add_argument('-atlas_files', type = str, required=True, nargs='*', help = 'List of 3 allen atlas .vtk files or 1 .tiff file')
    parser.add_argument('-outdir', type = str, required=True, help = 'The directory where all output files should be saved')
    parser.add_argument('-epath', type=str, default='', help = 'Default - \'\'; Local path to the emlddmm library. If no argument is supplied, the library will be cloned locally from its GitHub repository')
    parser.add_argument('-dslice', type=float, default = 500.0, help = 'Default: 500.0; Thickness of each slab (in um) of this dataset. This should be the same value as was used in registration.')
    parser.add_argument('-dI', type=float, default = 50.0, help = 'Default: 50.0; The resolution (um) of the atlas that was used in registration, even if it differs from the atlases provided here')
    parser.add_argument('-dsample', type=float, default = 50.0, help = 'Default: 50.0; The desired output resolution (um)')
    parser.add_argument('-zm', '--zero_mean', action='store_false', help = 'Default - True; If True, use the zero mean convention when loading input data. If this argument was passed during registration, pass it again here.')
    parser.add_argument('-debug', action='store_true', help = 'Default - False; If True, run the script in DEBUG mode')

    args = parser.parse_args()

    dataset = args.dataset
    target_files = args.target_files
    transformation_file = args.tform_file
    atlas_names = args.atlas_files
    outdir = args.outdir
    epath = args.epath
    dslice = args.dslice
    dI = args.dI
    dsample = args.dsample
    zeroMean = args.zero_mean
    debug = args.debug

    # ==========================================
    # ===== (0) Import necessary libraries =====
    # ==========================================
    if epath == '':
        print('Since no argument was supplied for the \'epath\' argument, the emlddmm library is being cloned from GitHub into your current directory . . .')
        os.mkdir("emlddmm")
        os.system("git clone https://github.com/twardlab/emlddmm emlddmm")
        sys.path.append(os.path.join(os.getcwd(),'emlddmm'))
    else:
        sys.path.append(epath)
    import emlddmm

    # ==============================
    # ===== (1) Load the atlas =====
    # ==============================
    xI,I,_,_ = emlddmm.read_data(atlas_names[-1])

    # resampled domain
    xImin = [x[0] for x in xI]
    xImax = [x[-1] for x in xI]
    
    xIr = [np.arange(x,X+dsample,dsample) for x,X in zip(xImin,xImax)]
    XIr = torch.tensor( np.stack( np.meshgrid(*xIr,indexing='ij') , -1) )

    # ===================================
    # ===== (2) Load the input data =====
    # ===================================
    xJs = []
    Js = []
    for f in target_files:
        data = np.load(f,allow_pickle=True)
        Ji = np.copy(data['I'])
        if Ji.ndim == 3:
            Ji = Ji[None]
    
        xJi = np.copy(data['xI'])
        dJi = [x[1] - x[0] for x in xJi]
        
        Js.append(torch.tensor(Ji))
        xJs.append([torch.tensor(x) - np.mean(x)*zeroMean for x in xJi])

    # ==================================================
    # ===== (3) Load the transformation parameters =====
    # ==================================================
    tformdata = np.load(transformation_file,allow_pickle=True)
    xv = [torch.tensor(x) for x in tformdata['xv']]
    v = torch.tensor( tformdata['v'] )
    XV = torch.stack(torch.meshgrid(*xv,indexing='ij'),-1)
    A = torch.tensor( tformdata['A'] )
    Ai = torch.linalg.inv(A)
    TJ = torch.tensor( tformdata['TJ'])
    AJ = torch.tensor(tformdata['AJ'])

    # ==========================================================================
    # ===== (4) Compute the transformed images at the specified resolution =====
    # ==========================================================================
    # initialize the 3D forward transformation
    nt = tformdata['v'].shape[0]
    phi = torch.clone(XV)
    for t in reversed(range(nt)):
        Xs = XV+v[t]/nt
        phi = (emlddmm.interp(xv,(phi-XV).permute(-1,0,1,2),Xs.permute(-1,0,1,2))).permute(1,2,3,0) + Xs
    
    if debug:
        # initialize the 3D inverse transformation 
        phii = torch.clone(XV)
        for t in range(nt):
            Xs = XV-v[t]/nt
            phii = (emlddmm.interp(xv,(phii-XV).permute(-1,0,1,2),Xs.permute(-1,0,1,2))).permute(1,2,3,0) + Xs    

    # Loop over all slices . . .
    
    # initialize empty arrays
    Jrecon = torch.zeros(XIr[...,0][None].shape) # the image to reconstruct
    Precon = torch.zeros(XIr[...,0].shape) # the "projection" what gets sliced out.  This is important when sticking neighoring slices together.
    ntJ = tformdata['vJ'][0].shape[0]
    for i in range(len(target_files)):
        
        # extract some information about sample points
        xJ = xJs[i]
        XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)    
        xvJ = [torch.tensor(x) for x in tformdata['xvJ'][i]]    
        vJ = torch.tensor( tformdata['vJ'][i] )
        XVJ = torch.stack(torch.meshgrid(*xvJ,indexing='ij'),-1) 
       
        # the transform from target to atlas is needed to cut out a slice
        AJi = torch.linalg.inv(AJ[i]) # 2d affine
        TJi = torch.linalg.inv(TJ[i])
        Xs = (AJi[:3,:3]@XJ[...,None])[...,0] + AJi[:3,-1]
        
        # now add the diffeomorphism
        phiiJ = torch.clone(XVJ)
        for t in range(ntJ):
            Xs_ = XVJ-vJ[t]/ntJ
            phiiJ = (emlddmm.interp(xvJ,(phiiJ-XVJ).permute(-1,0,1,2),Xs_.permute(-1,0,1,2))).permute(1,2,3,0) + Xs_    
        
        # where should I insert the diffeomorphism? I think it should be after Ai, before Ti
        Xs = emlddmm.interp(xvJ,(phiiJ-XVJ).permute(-1,0,1,2),Xs.permute(-1,0,1,2)).permute(1,2,3,0) + Xs
        
        # now translate
        Xs = (TJi[:3,:3]@Xs[...,None])[...,0] + TJi[:3,-1]
        
        # now evaluate the projection in this space
        trap_i = i    
        P = trapezoid(Xs[...,0],trap_i,[[]]*len(Js),[dI,dI,dI],dslice)
        
        # keep going with Ai and phii for debug
        if debug:
            Xs = (Ai[:3,:3]@Xs[...,None])[...,0] + Ai[:3,-1]
            Xs = emlddmm.interp(xvJ,(phii-XV).permute(-1,0,1,2),Xs.permute(-1,0,1,2)).permute(1,2,3,0) + Xs
        
            # now transform the atlas
            phiI_ = emlddmm.interp(xI,I*1.0,Xs.permute(-1,0,1,2))
            phiI__ = phiI_*P
    
        # now the transform from atlas to target, to reconstruct data in atlas space
        phiJ = torch.clone(XVJ)
        for t in reversed(range(ntJ)):        
            Xs = XVJ+vJ[t]/ntJ
            phiJ = (emlddmm.interp(xvJ,(phiJ-XVJ).permute(-1,0,1,2),Xs.permute(-1,0,1,2))).permute(1,2,3,0) + Xs
    
        # diffeo                
        Xs = emlddmm.interp(xv,(phi-XV).permute(-1,0,1,2),XIr.permute(-1,0,1,2)).permute(1,2,3,0) + XIr
        # affine
        Xs = (A[:3,:3]@Xs[...,None])[...,0] + A[:3,-1]
        # translation T                
        Xs = (TJ[i][:3,:3]@Xs[...,None])[...,0] + TJ[i][:3,-1]
        # diffeo J
        Xs = emlddmm.interp(xvJ,(phiJ-XVJ).permute(-1,0,1,2),Xs.permute(-1,0,1,2)).permute(1,2,3,0) + Xs
        # affine AJ
        Xs = (AJ[i][:3,:3]@Xs[...,None])[...,0] + AJ[i][:3,-1]
        # reconstruct
        Jrecon += emlddmm.interp(xJs[i],Js[i]*P,Xs.permute(-1,0,1,2),padding_mode='zeros')
        Precon += emlddmm.interp(xJs[i],P[None],Xs.permute(-1,0,1,2),padding_mode='zeros')[0]
    
    Jrecon /= Precon
    Jrecon[Precon[None]==0]=0

    if debug:
        quantile_to_pad = 0.1
        val = np.quantile(Ji.ravel(),quantile_to_pad)
        
        fig,ax = plt.subplots(3,1)
        ax[0].imshow(phiI_[0,phiI_.shape[1]//2]>0)
        ax[1].imshow(phiI_[0,:,phiI_.shape[2]//2]>0)
        ax[2].imshow(phiI_[0,:,:,phiI_.shape[3]//2]>0)
        plt.savefig(os.path.join(outdir, 'fig0.jpg'))
        
        fig,ax = plt.subplots(3,1)
        ax[0].imshow(phiI__[0,phiI_.shape[1]//2]>0)
        ax[1].imshow(phiI__[0,:,phiI_.shape[2]//2]>0)
        ax[2].imshow(phiI__[0,:,:,phiI_.shape[3]//2]>0)
        plt.savefig(os.path.join(outdir, 'fig1.jpg'))
        
        fig,ax = plt.subplots(3,1)
        ax[0].imshow(Js[i][0,phiI_.shape[1]//2],vmin=val)
        ax[1].imshow(Js[i][0,:,phiI_.shape[2]//2],vmin=val)
        ax[2].imshow(Js[i][0,:,:,phiI_.shape[3]//2],vmin=val)
        plt.savefig(os.path.join(outdir, 'fig2.jpg'))

    Ir = emlddmm.interp(xI,torch.tensor(I,dtype=torch.float64),XIr.permute(-1,0,1,2),mode='nearest').to(int)

    # these values likey vary from dataset to dataset
    if debug:
        vmin=1.77
        vmax=1.85
        Jreconshow = (Jrecon-vmin)/(vmax-vmin)
        emlddmm.draw(torch.concatenate((Jreconshow,(Ir%7)/7*0.25)),xIr,vmin=0,vmax=1)

    # =====================================
    # (5) ===== Save relevant outputs =====
    # =====================================
    emlddmm.write_data(os.path.join(outdir,'jrecon.vtk'),xIr,Jrecon,dataset)

    print(f'Successfully saved transformed data and saved outputs in {outdir}')

    

def trapezoid(x,i,J,dI,dslice):
    """Models the cutting of a big piece of tissue into a thin slab. In order to make the operation differentiable, we multiply the image by a trapezoid instead of a rectangle. This function defines the trapezoid by which we will multiply the image. This is an example of a processing step described in projective LDDMM. See "Projective diffeomorphic mapping of molecular digital pathology with tissue MRI" (https://doi.org/10.1038/s44172-022-00044-1)
    
    Parameters
    ----------
    x : torch.tensor
        An nd-array that stores the slice location of each voxel in a volume
    i : int
        The index of the slice to be processed
    J : numpy.ndarray
        An array of all the 3D slice data in Anterior-Posterior order
    dI : numpy.ndarray
        The resolution of the atlas along each of the 3 axes
    dslice : float
        The thickness of the slice to be processed in microns

    Returns
    -------
    out : torch.tensor
        The trapezoid by which we will multiply an image
    """
    d = torch.abs(x - (i - (len(J)-1)/2.0)*dslice)
    out = (d < dslice/2.0 - dI[0])*1.0 + (d >= dslice/2.0 - dI[0])*(d < dslice/2.0+dI[0])*(1.0 - (d-dslice/2.0)/dI[0])*0.5
    return out

if __name__ == '__main__':
    main()