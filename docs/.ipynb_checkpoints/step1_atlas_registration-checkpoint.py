"""Low resolution mouse brain image to allen mouse brain atlas registration

This script is intended to be used at the command line and will save the initial, intermediate, final registration results for aligning a single 3D slice (.npz) or multiple consecutive slices ([.npz,...]) to the Allen Mouse Brain Atlas. This script performs step 1 of 3 in the registration pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
import sys

from step1_help import str_to_2D_mat, draw_stack, trapezoid, gA_from_gid

import argparse

def main():
    """
    Command line arguments
    ----------------------
    dataset : str
        The name of the dataset to be registered. This will be included in the names of the majority of the output files.
    orientation : char
        The portion of the brain captured by this dataset (W - whole, L - left hemisphere, R - right hemisphere)
    -low_paths : str or list of str
        1 or more file paths pointing to the location(s) of the low resolution mouse brain images (.npz) in Anterior to Posterior order
    -outdir : str
        The path to the directory where all outputs will be stored
    -atlas_paths : list of 3 str
        The paths pointing to the 3 Allen mouse brain atlases used for registration. In order, these files are named (1) ara_nissl_50.vtk, (2) average_template_50.vtk, and (3) annotation_50.vtk
    -to_flip : int or list of int
        A list of the indeces of the slices from '-low' that need to flipped so that they are all the same orientation
    -device : str
        The device for torch computation (Options: 'cpu', 'cuda:0', 'cuda:1', 'mps')
    -A : str
        The 4x4 affine matrix used to estimate an initial guess for the registration algorithm. Should be of the form [[x,x,x,x],[x,x,x,x],[x,x,x,x],[x,x,x,x]]; If not supplied, a sensible default will be produced based on orientation to reproduce previous work with the Yang lab\'s MORF project
    -AJ : list of str
        A list of 4x4 affine matrices used to estimate an initial guess for each slice to be used in the registration algorithm. Should be a list of the same length as the number of slices and each element must be of the form [[x,x,x,x],[x,x,x,x],[x,x,x,x],[x,x,x,x]]; If not supplied, a sensible default will be produced based on orientation to reproduce previous work with the Yang lab\'s MORF project
    -dslice : int
        Default - 500; The thickness of the slab(s) within the dataset, in units of um
    -e_path : str
        The location of the custom Python library 'emlddmm', which can be cloned from GitHub at https://github.com/twardlab/emlddmm
    -d_path : str
        The location of the custom Python library 'donglab_workflows', which be cloned from GitHub at https://github.com/twardlab/donglab_workflows
        
    -save_fig0 : bool
        Default - False; If present, save histogram of voxel values in outdir
    -save_fig1 : bool
        Default - False; If present, save figure showing MIP of every slice in outdir
    -save_fig2 : bool
        Default - False; If present, save figure showing where data is missing from low-res images in outdir
    -save_fig3 : bool
        Default - False; If present, save figure displaying the trapezoid function in outdir
    -save_fig4 : bool
        Default - False; If present, save figure displaying a map of the edges of the atlas in outdir
    -save_fig5 : bool
        Default - False; If present, save figure displaying the inner product which allows us to perform gradient descent in outdir
    -save_fig6 : bool
        Default - False; If present, save figure displaying the 12x12 matrices gid and gJid in outdir

    --normalizeData : bool
        Default - True; If True, normalize input data to [0,1]
    --zeroMean : bool
        Default - True; If True, use the zero mean convention when loading all input data
    --largerRGBfig : bool
        Default - False; If True, Include 8 panels for each orientation in the figRGB outputs during regstration, otherwise include 5 panels
        
    Raises
    ------
    Exception
        if any of the files provided to the '-low' argument cannot be accessed
    Exception
        if the number of arguments passed to 'AJ' does not equal the number of arguments passed to 'low'
    ValueError
        if the argument 'A' is not of the correct form
    ValueError
        if any elements from the list argument 'AJ' are not of the correct form

    """

    # Used as an argument to init affine matrix, A
    dtype = torch.float64
    
    parser = argparse.ArgumentParser()
    
    # Subparser for a new brain
    parser.add_argument('dataset', type=str,help = 'Name of dataset to be processed')
    parser.add_argument('orientation', type = str, choices = ['W','L','R'], help = 'The portion of the brain shown in the dataset (Whole, Left hemisphere, Right hemisphere)')
    parser.add_argument('-low_paths', type = str, nargs='*', required = True, help = 'List of low-res files in Anterior to Posterior order')
    parser.add_argument('-outdir', type = str, required = True, help = 'Output directory for all files generated from this script')
    parser.add_argument('-atlas_paths', type = str, nargs = 3, required = True, help = 'List of 3 allen atlas .vtk files')
    parser.add_argument('-to_flip', nargs = '*', type=int, default = -1, help = 'Indices of which slices to flip from low_res_files')
    parser.add_argument('-device', type=str, choices = ['cpu', 'cuda:0', 'cuda:1', 'mps'], required = True, help = 'Device for torch computations')
    parser.add_argument('-A', type = str, help = 'Initial guess - [[x,x,x,x],[x,x,x,x],[x,x,x,x],[x,x,x,x]]; If not supplied, a sensible default will be produced based on orientation to reproduce previous work with the Yang lab\'s MORF project')
    parser.add_argument('-AJ', type = str, nargs = '*', help = 'A list of initial shifts for each slice of the form [[x,x,x,x],[x,x,x,x],[x,x,x,x],[x,x,x,x]]; If not supplied, a sensible default will be produced based on orientation to reproduce previous work with the Yang lab\'s MORF project')
    parser.add_argument('-dslice', type = int, default = 500, help = 'Default - 500; Thickness of slab(s) in dataset, in units of um')
    parser.add_argument('-e_path', type = str, required = True, help = 'The directory containing the emlddmm library from Github')
    parser.add_argument('-d_path', type = str, required = True, help = 'The directory containing the donglab_workflows library from Github')
    
    parser.add_argument('-save_fig0', action = 'store_true', help = 'Default - False; If present, save histogram of voxel values in outdir')
    parser.add_argument('-save_fig1', action = 'store_true', help = 'Default - False; If present, save figure showing MIP of every slice in outdir')
    parser.add_argument('-save_fig2', action = 'store_true', help = 'Default - False; If True, save figure showing where data is missing from low-res images in outdir')
    parser.add_argument('-save_fig3', action = 'store_true', help = 'Default - False; If present, save figure displaying the trapezoid function in outdir')
    parser.add_argument('-save_fig4', action = 'store_true', help = 'Default - False; If present, save figure displaying a map of the edges of the atlas in outdir')
    parser.add_argument('-save_fig5', action = 'store_true', help = 'Default - False; If present, save figure displaying the inner product which allows us to perform gradient descent in outdir')
    parser.add_argument('-save_fig6', action = 'store_true', help = 'Default - False; If present, save figure displaying the 12x12 matrices gid and gJid in outdir')
    
    parser.add_argument('-norm','--normalizeData', action = 'store_false', help = 'Default - True; If True, normalize input data to [0,1]')
    parser.add_argument('-zm','--zeroMean', action = 'store_false', help = 'Default - True; If True, use the zero mean convention when loading input data')
    parser.add_argument('--largerRGBfig', action = 'store_true', help = 'Default - False; If True, Include 8 panels for each orientation in the figRGB outputs during regstration, otherwise include 5 panels')
    
    args = parser.parse_args()

    brain = args.dataset
    orientation = args.orientation
    target_files = args.low_paths
    outdir = args.outdir
    atlas_names = args.atlas_paths
    dslice = args.dslice
    device = args.device

    # If no input is provided, then define to_flip as an empty list. Otherwise, use the input
    if args.to_flip == -1:
        to_flip = []
    else:
        to_flip = args.to_flip


    if args.A != None:
        A = args.A
        if A.count(',') != 15 or A.count(']') != 5 or A.count('[') != 5:
            val0 = A.count('[')
            val1 = A.count(']')
            val2 = A.count(',')
            raise ValueError(f'Incorrect format for A. Expected 5 \'[\', 5 \']\', and 15 \',\'. Instead, {val0} \'[\', {val1} \']\', and {val2} \',\' were passed.')
        A = str_to_2D_mat(A)
        A = torch.tensor(A, device = device, dtype = dtype)

    if args.AJ != None:
        AJ = args.AJ
        if len(AJ) != len(target_files):
            val0 = len(AJ)
            val1 = len(target_files)
            raise Exception(f'Number of matrices in AJ ({val0} supplied) must equal the number of files in low ({val1} supplied)')
        AJ_out = []
        for i, AJ_i in enumerate(AJ):
            if AJ_i.count(',') != 15 or AJ_i.count(']') != 5 or AJ_i.count('[') != 5:
                val0 = AJ_i.count('[')
                val1 = AJ_i.count(']')
                val2 = AJ_i.count(',')
                raise ValueError(f'Incorrect format for AJ[{i}]. Expected 5 \'[\', 5 \']\', and 15 \',\'. Instead, {val0} \'[\', {val1} \']\', and {val2} \',\' were passed.')
            AJ_i = str_to_2D_mat(AJ_i)
            AJ_out.append(torch.tensor(AJ_i, device = device, dtype = dtype))
        AJ = AJ_out

    saveFig0 = args.save_fig0
    saveFig1 = args.save_fig1
    saveFig2 = args.save_fig2
    saveFig3 = args.save_fig3
    saveFig4 = args.save_fig4
    saveFig5 = args.save_fig5
    saveFig6 = args.save_fig6
    normalizeData = args.normalizeData
    zeroMean = args.zeroMean
    largerRGBfig = args.largerRGBfig


    # Import custom libraries from local device (Repo should have already been cloned from Github)
    sys.path.append(args.d_path)
    import donglab_workflows as dw
    sys.path.append(args.e_path)
    import emlddmm
    
    seg_name = atlas_names[2]
    atlas_names = atlas_names[:2]

    # Perform checks on input data
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    
    # =================================================================
    # ===== (1) Load the atlas + truncate hemisphere if necessary =====
    # =================================================================
    I = []
    for atlas_name in atlas_names:
        xI,I_,title,names = emlddmm.read_data(atlas_name)
        I_ = I_.astype(np.float32)
        I_ /= np.mean(np.abs(I_))
    
        # Will zero-out atlas image for the hemisphere we are NOT aligning
        if orientation == 'R': # For images only containing a RIGHT  hemisphere
            I_[:,:,:int(np.shape(I_)[2]/2),:] = 0
        elif orientation == 'L': # For images only containing a LEFT hemisphere (+1 brings back fig[1,2])
            I_[:,:,int(np.shape(I_)[2]/2 + 1):,:] = 0
        else:
            I_ = I_
            
        I.append(I_)

    I = np.concatenate(I)
    I[0] = I[0]**0.25
    I[0] /= np.mean(np.abs(I_[0]))
    dI = np.array([x[1] - x[0] for x in xI])
    XI = np.stack(np.meshgrid(*xI,indexing='ij'),-1)
    
    # =================================
    # ===== (2) Load the 10x data =====
    # =================================
    Js = []
    xJs = []
    Ws = []
    dJs = []
    DJs = []
    for fname in target_files:
        if not os.path.exists(fname):
            raise Exception(f'Invalid file in target_files - {fname}')
            
        data = np.load(fname,allow_pickle=True)
        if data['I'].ndim == 3:
            Js.append(data['I'][None])
        else: # data['I'].ndim == 4
            Js.append(data['I'])
        
        # We decided to ignore voxel positions and use zero-mean convention b/c it is more numerically stable
        # data['xI'] is a list of 3 1d arrays with z,y,x locations
        if zeroMean:
            xJs.append([d-np.mean(d) for d in data['xI']])
        else:
            xJs.append([x for x in data['xI']])

    
        if 'W' in data or 'w' in data:
            if 'W' in data:
                key = 'W'
            else:
                key = 'w'
            W_ = data[key]
            if W_.ndim == 4:
                W_ = W_[0]
            Ws.append(W_)
        else:
            Ws.append(np.ones_like(Js[-1][0]))

        if normalizeData:
            # we need to normalize to [0,1]
            mymin = np.quantile(Js[-1][Ws[-1][None]>0.99],0.05)
            mymax = np.quantile(Js[-1][Ws[-1][None]>0.99],0.99)
            Js[-1] = (Js[-1] - mymin)/(mymax - mymin)
            
            # I'd like to squash the bright signal
            signJ = np.sign(Js[-1])
            Js[-1] = signJ*abs(Js[-1])**0.25
        
            # we need to normalize to [0,1] again!
            mymin = np.quantile(Js[-1][Ws[-1][None]>0.99],0.05)
            mymax = np.quantile(Js[-1][Ws[-1][None]>0.99],0.99)
        
            Js[-1] = (Js[-1] - mymin)/(mymax - mymin)
            Js[-1][Js[-1]<0.0] = 0.0 # after scaling I'm going to zero out the low signals (new feb 2 2023)
            Ws[-1][Ws[-1]<0.999] = 0.0 # also I think the weights are not strong enough
        
        dJs.append([x[1]-x[0] for x in xJs[-1]])
        DJs.append(np.prod(dJs[-1]) )
    
        # Make shape of xJs == shape of Js
        for s in [-1,-2,-3]:
            if len(xJs[-1][s]) != Js[-1].shape[s]:
                n = Js[-1].shape[s] - len(xJs[-1][s]) 
                d = xJs[-1][s][1] - xJs[-1][s][0]
                xJs[-1][s] = np.concatenate((xJs[-1][s],[xJs[-1][s][-1]+d ]))
    
    if saveFig0:
        f,ax = plt.subplots()
        ax.hist(Js[-1][Ws[-1][None]>0.99].ravel(),100)
        f.savefig(os.path.join(outdir,f'{brain}_fig0'))
    
    # ==========================================
    # ===== (3) Augment 10x data (Padding) =====
    # ==========================================
   
    for i in range(len(Js)):
        Ji = Js[i]
        Wi = Ws[i]
        xJi = xJs[i]
        dJi = dJs[i]
        
        # find the 10% quantile
        if normalizeData:
            if brain == 'hTME19-2':
                val = np.quantile(Ji[Wi[None]>=0.99],0.25)
            else:
                val = np.quantile(Ji[Wi[None]>=0.99],0.1)
        else:
            val = np.quantile(Ji[Wi[None]>=0.99],0.05)
            
        npad = 2
        Ji = np.pad(Ji,((0,0),(npad,npad),(0,0),(0,0)), constant_values=val)
        Wi = np.pad(Wi,((npad,npad),(0,0),(0,0)), constant_values=1.0)
        for j in range(npad):
            # Grows the Xs, so that we know the location of the padded pixesl too
            xJi[0] = np.concatenate((xJi[0][0][None]-dJi[0], xJi[0], xJi[0][-1][None]+dJi[0]))
        
        Js[i] = Ji
        Ws[i] = Wi
        xJs[i] = xJi
    
    vminJ = 0.0
    vmaxJ = 1.0
    
    if saveFig1:
        fig,ax = draw_stack(Js,xJs,vmin=vminJ,vmax=vmaxJ)
        fig.savefig(os.path.join(outdir,f'{brain}_fig1'))
    if saveFig2:
        fig,ax = draw_stack([w[None] for w in Ws],xJs,vmin=0,vmax=1)
        fig.savefig(os.path.join(outdir,f'{brain}_fig2'))
    
    # ======================================================
    # ===== (4) Convert 10x data into torch.tensor() =====
    # ======================================================
    Inp = np.copy(I)
    xInp = [np.copy(x) for x in xI]
    Jnp = [np.copy(Ji) for Ji in Js]
    xJnp = [[np.copy(x) for x in xJ] for xJ in xJs]
    Wnp = [np.copy(Wi) for Wi in Ws]
    
    dtype = torch.float64
    I = torch.tensor(Inp,device=device,dtype=dtype)
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xInp]
    J = [torch.tensor(Ji,device=device,dtype=dtype) for Ji in Jnp]
    W = [torch.tensor(Wi,device=device,dtype=dtype) for Wi in Wnp]
    xJ = [[torch.tensor(x,device=device,dtype=dtype) for x in xJi] for xJi in xJnp]
    XJ = [ torch.stack(torch.meshgrid(x[0],x[1],x[2],indexing='ij'),-1) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI,indexing='ij'),-1)
    
    if saveFig3:
        t = torch.linspace(-3000,3000,1000)
        f,ax = plt.subplots()
        ax.plot(t,trapezoid(t,2,J,dI,dslice))
        f.savefig(os.path.join(outdir,f'{brain}_fig3'))
    
    # ========================================================================
    # ===== (5) Compute other matrices and scalars for later computation =====
    # ========================================================================
    DI = torch.prod(torch.tensor(dI,device=device,dtype=dtype))
    W_ = (I[0] > 0).to(dtype)
    DW = torch.stack(torch.gradient(W_,spacing=dI.tolist(),dim=(0,1,2)),-1)
    
    if saveFig4:
        fig,axs = emlddmm.draw(DW.permute(-1,0,1,2))
        fig.savefig(os.path.join(outdir,f'{brain}_fig4'))
    
    # Builds the 12x12 matrix using the derivative of the image
    gid = torch.zeros(12,12,device=device,dtype=dtype)
    count0 = 0
    for i0 in range(3):
        for j0 in range(4):
            E0 = ((torch.arange(4,device=device,dtype=dtype)==i0)[:,None]*(torch.arange(4,device=device,dtype=dtype)==j0)[None,:]).to(dtype)
            if j0 == 3:
                tosum0 = DW[...,i0]
            else:
                tosum0 = DW[...,i0]*XI[...,j0]
            count1 = 0
            for i1 in range(3):
                for j1 in range(4):
                    if count0 > count1:
                        count1 += 1
                        continue                                
                    E1 = ((torch.arange(4,device=device,dtype=dtype)==i1)[:,None]*(torch.arange(4,device=device,dtype=dtype)==j1)[None,:]).to(dtype)                
                    if j1 == 3:
                        tosum1 = DW[...,i1]
                    else:
                        tosum1 = DW[...,i1]*XI[...,j1]
                                                                
                    gid[count0,count1] = torch.sum(tosum0*tosum1)*DI
                    gid[count1,count0] = gid[count0,count1]                                
                    
                    count1 += 1
            count0 += 1
    
    # How to use the inner product for the 2D affine
    Jind = len(XJ)//2
    W_ = torch.ones_like(XJ[Jind][...,0])
    border = 2
    W_[:border] = 0.0
    W_[-border:] = 0.0
    W_[:,:border] = 0.0
    W_[:,-border:] = 0.0
    W_[:,:,:border] = 0.0
    W_[:,:,-border:] = 0.0
    DW = torch.stack(torch.gradient(W_,spacing=dI.tolist(),dim=(0,1,2)),-1)
    
    if saveFig5:
        fig, axs = emlddmm.draw(DW.permute(-1,0,1,2))
        fig.savefig(os.path.join(outdir,f'{brain}_fig5'))
    
    gJid = torch.zeros(12,12,device=device,dtype=dtype)
    count0 = 0
    for i0 in range(3):
        for j0 in range(4):
            E0 = ((torch.arange(4,device=device,dtype=dtype)==i0)[:,None]*(torch.arange(4,device=device,dtype=dtype)==j0)[None,:]).to(dtype)
            if j0 == 3:
                tosum0 = DW[...,i0]
            else:
                tosum0 = DW[...,i0]*XJ[Jind][...,j0]
                
            count1 = 0
            for i1 in range(3):
                for j1 in range(4):
                    if count0 > count1:
                        count1 += 1
                        continue                                
                    E1 = ((torch.arange(4,device=device,dtype=dtype)==i1)[:,None]*(torch.arange(4,device=device,dtype=dtype)==j1)[None,:]).to(dtype)                
                    if j1 == 3:
                        tosum1 = DW[...,i1]
                    else:
                        tosum1 = DW[...,i1]*XJ[Jind][...,j1]
                                                                
                    gJid[count0,count1] = torch.sum(tosum0*tosum1)*DJs[Jind]
                    gJid[count1,count0] = gJid[count0,count1]                                
                    
                    count1 += 1
            count0 += 1
    
    if saveFig6:
        f,ax = plt.subplots(1,2)
        ax[0].imshow(gid.cpu())
        ax[1].imshow(gJid.cpu())
        f.savefig(os.path.join(outdir,f'{brain}_fig6'))
    
    
    # Set up kernels / Defines the spatial scale, following a form commonly used in the literature
    dv = torch.tensor([1.0,1.0,1.0],device=device,dtype=dtype)*500.0
    DV = torch.prod(dv)
    dI = torch.tensor(dI,device=device,dtype=dtype)
    v_expand_factor = 0.1
    a = 500.0
    p = 2.0
    aprefactor = 0.25
    nt = 5
    
    x0v = [x[0] - (x[-1]-x[0])*v_expand_factor for x in xI]
    x1v = [x[-1] + (x[-1]-x[0])*v_expand_factor for x in xI]
    xv = [torch.arange(x0,x1,d,device=device,dtype=dtype) for x0,x1,d in zip(x0v,x1v,dv)]
    nv = torch.tensor([len(x) for x in xv],device=device,dtype=dtype)
    XV = torch.stack(torch.meshgrid(xv),-1)
    
    # build energy operator for velocity
    fv = [torch.arange(n,device=device,dtype=dtype)/d/n for n,d in zip(nv,dv)]
    FV = torch.stack(torch.meshgrid(fv))
    
    LL = (1.0 - 2.0*a**2 * 
              ( (torch.cos(2.0*np.pi*FV[0]*dv[0]) - 1)/dv[0]**2  
            + (torch.cos(2.0*np.pi*FV[1]*dv[1]) - 1)/dv[1]**2  
            + (torch.cos(2.0*np.pi*FV[2]*dv[2]) - 1)/dv[2]**2   ) )**(p*2)
    K = 1.0/LL
    
    LLpre = (1.0 - 2.0*(aprefactor*torch.max(dI))**2 * 
             ( (torch.cos(2.0*np.pi*FV[0]*dv[0]) - 1)/dv[0]**2  
             + (torch.cos(2.0*np.pi*FV[1]*dv[1]) - 1)/dv[1]**2  
             + (torch.cos(2.0*np.pi*FV[2]*dv[2]) - 1)/dv[2]**2   ) )**(p*2)
    Kpre = 1.0/LLpre
    KK = K*Kpre
    
    # set up kernels for blocks
    dvJ = torch.tensor([1.0,1.0,1.0],device=device,dtype=dtype)*100.0
    DVJ = torch.prod(dvJ)
    v_expand_factor = 0.1
    aJ = 50.0
    pJ = 2.0
    aprefactorJ = 0.25
    ntJ = 3
    
    LLJ = []
    KJ = []
    KJpre = []
    XVJ = []
    xvJ = []
    for i in range(len(J)):
        x0vJ = [x[0] - (x[-1]-x[0])*v_expand_factor for x in xJs[i]]
        x1vJ = [x[-1] + (x[-1]-x[0])*v_expand_factor for x in xJs[i]]    
        xvJi = [torch.arange(x0,x1,d,device=device,dtype=dtype) for x0,x1,d in zip(x0vJ,x1vJ,dvJ)]
        nvJi = torch.tensor([len(x) for x in xvJi],device=device,dtype=dtype)
        XVJi = torch.stack(torch.meshgrid(xvJi),-1)
    
        # build energy operator for velocity
        fvJi = [torch.arange(n,device=device,dtype=dtype)/d/n for n,d in zip(nvJi,dvJ)]
        FVJi = torch.stack(torch.meshgrid(fvJi))
    
        LLJi = (1.0 - 2.0*aJ**2 * 
                ( (torch.cos(2.0*np.pi*FVJi[0]*dvJ[0]) - 1)/dvJ[0]**2  
                + (torch.cos(2.0*np.pi*FVJi[1]*dvJ[1]) - 1)/dvJ[1]**2  
                + (torch.cos(2.0*np.pi*FVJi[2]*dvJ[2]) - 1)/dvJ[2]**2   ) )**(pJ*2)
        KJi = 1.0/LLJi
    
        LLpreJi = (1.0 - 2.0*(aprefactorJ*torch.max(dI))**2 * 
                 ( (torch.cos(2.0*np.pi*FVJi[0]*dvJ[0]) - 1)/dvJ[0]**2  
                 + (torch.cos(2.0*np.pi*FVJi[1]*dvJ[1]) - 1)/dvJ[1]**2  
                 + (torch.cos(2.0*np.pi*FVJi[2]*dvJ[2]) - 1)/dvJ[2]**2   ) )**(pJ*2)
        KpreJi = 1.0/LLpreJi
        KKJi = KJi*KpreJi
    
        LLJ.append(LLJi)
        KJ.append(KJi)
        KJpre.append(KpreJi)
        XVJ.append(XVJi)
        xvJ.append(xvJi)
    
    ### The values used to initialize A, TJ, and AJ were used to successfully register datasets associated with the MORF project in collaboration with the Yang Lab (TME, hTME)
    # Initialize affine matrix, A, to define image orientations (Only if A is not supplied)
    if args.A == None:
        if orientation == 'W': # TME
            A = torch.tensor([[-1.0,0.0,0.0,500.0],
                             [0.0,0.0,-1.0,0.0],
                             [0.0,1.0,0.0,0.0],
                             [0.0,0.0,0.0,1.0]],device=device,dtype=dtype) 
        if orientation == 'R': # */Q140_MORF/*
            A = torch.tensor([[-1.0,0.0,0.0,1000.0],
                         [0.0,0.0,-1.0,200.0],
                         [0.0,1.0,0.0,-1800.0],
                         [0.0,0.0,0.0,1.0]],device=device,dtype=dtype)
        if orientation == 'L': # */KO-Het/*
            A = torch.tensor([[-1.0,0.0,0.0,1000.0],
                         [0.0,0.0,-1.0,200.0],
                         [0.0,1.0,0.0,1800.0],
                         [0.0,0.0,0.0,1.0]],device=device,dtype=dtype)
    
    # For a few unique cases where the dataset has already been registered using additional slab location info (nelder_mead estimation of AP shift)
    if 'TME08-1' in brain:
        A[0,-1] = 675.0
    if 'TME10-1' in brain:
        A[0,-1] = 406.0
    if 'TME10-3' in brain:
        A[0,-1] = 495.0
    if 'TME07-1' in brain:
        A[0,-1] = 487.0
    if 'TME09-1' in brain:
        A[0,-1] = 408.0
    if 'hTME15-1' in brain:
        A[0,-1] = 368.0
    if 'hTME15-2' in brain:
        A[0,-1] = 413.0
    if 'hTME18-1' in brain:
        A[0,-1] = 372.0
    if 'hTME19-2' in brain:
        A[0,-1] = 537.0
    
    # Define affine and transform matrices TJ and AJ
    TJ = [torch.tensor([[1.0,0.0,0.0,-(i - (len(J)-1)/2.0)*dslice],
                       [0.0,1.0,0.0,0.0],
                       [0.0,0.0,1.0,0.0],
                       [0.0,0.0,0.0,1.0]],device=device,dtype=dtype) for i in range(len(J))]

    if args.AJ == None:
        AJ = [torch.tensor([[0.75,0.0,0.0,np.mean(xJs[i][0])], # 0.75 shrinks the slices, ensuring the edges don't exceed the volume boundaries
                       [0.0,1.0,0.0,np.mean(xJs[i][1])],
                       [0.0,0.0,1.0,np.mean(xJs[i][2])],
                       [0.0,0.0,0.0,1.0]],device=device,dtype=dtype) for i in range(len(J))]
    
    AJ = torch.stack(AJ)
    
    # Perform AP and LR flips on the relevant slices in 'brain'
    if len(to_flip) > 0:
        for i in range(AJ.shape[0]):
            if i in to_flip:
                AJ[i][0,0] = AJ[i][0,0]*(-1) # anterior-posterior flip
                AJ[i][2,2] = AJ[i][2,2]*(-1) # left right flip
    
    v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2],XV.shape[3]),device=device,dtype=dtype,requires_grad=True)
    vJ = [torch.zeros((ntJ,KJ[i].shape[0],KJ[i].shape[1],KJ[i].shape[2],3),device=device,dtype=dtype,requires_grad=True) for i in range(len(J))]

    
    # =================================================
    # ===== (6) Starts the optimization procedure =====
    # =================================================
    A.requires_grad = True
    AJ.requires_grad = True
    v.requires_grad = True
    for vJi in vJ: vJi.requires_grad = True
    
    niter = 40000
    sigmaR = 1e6
    sigmaR = 1e4
    sigmaRJ = 1e3
    
    vstart = 2000
    vJstart = 5000
    
    # J: Each slice (Non-J: The whole brain)
    eA = 2e-1
    eAJ = eA*5e-2
    ev = 1e-3
    evJ = 2e-5 # with sigmaRJ smaller, we need evJ smaller
    
    # Level of regularization one each slice
    sigmaRJ = 2e2
    
    # this means I will draw on 0, 10, and log spacing (or sqrt)
    ndraw = 1
    its = np.arange(niter)
    test = np.log(its+1)
    test = (its+1)**0.5
    #test = (its+1)**0.75
    test /= test[ndraw]
    draws = np.diff(np.ceil(test)) > 0
    draws = np.concatenate((draws,[False]))
    draws[0] = True
    
    figv = plt.figure()
    figPhiI = plt.figure()
    figFPhiI = plt.figure()
    figErr = plt.figure()
    figJ = plt.figure()
    figJRecon = plt.figure()
    figRGB = plt.figure()
    figE,axE = plt.subplots(1,3)
    axE = axE.ravel()
    figP,axP = plt.subplots(2,3)
    axP = axP.ravel()
    Esave = []
    Lsave = []
    Tsave = []
    LJsave = []
    TJsave = []
    maxvsave = []
    maxvJsave = []
    
    for it in range(niter):
        start = time.time()
        # take the atlas and map it onto the slices
        phiI = []
        fphiI = []
        coeffs = []
        err = []
        if draws[it]:#not it%ndraw:
            Jrecon = torch.zeros_like(I[0][None])
            Precon = torch.zeros_like(I[0])
            
        Ai = torch.linalg.inv(A)
        EM = 0.0
        E = 0.0
        drawtime = 0.0
        
        # generate the single deformation
        phii = torch.clone(XV)
        for t in range(nt):
            Xs = XV-v[t]/nt
            phii = (emlddmm.interp(xv,(phii-XV).permute(-1,0,1,2),Xs.permute(-1,0,1,2))).permute(1,2,3,0) + Xs
        # the energy
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2,3)))**2,dim=(0,-1))*LL)/np.prod(v.shape[1:4])*DV/2.0/sigmaR**2/nt
        E += ER
        ERJ = 0.0
        for i in range(len(J)):
            # 2d affine
            AJi = torch.linalg.inv(AJ[i])
            TJi = torch.linalg.inv(TJ[i])
            TAJi = TJi@AJi
            Xs = (AJi[:3,:3]@XJ[i][...,None])[...,0] + AJi[:3,-1]
            
            # now add the diffeomorphism
            phiiJ = torch.clone(XVJ[i])
            for t in range(ntJ):
                Xs_ = XVJ[i]-vJ[i][t]/ntJ
                phiiJ = (emlddmm.interp(xvJ[i],(phiiJ-XVJ[i]).permute(-1,0,1,2),Xs_.permute(-1,0,1,2))).permute(1,2,3,0) + Xs_
            ERJ_ = torch.sum(torch.sum(torch.abs(torch.fft.fftn(vJ[i],dim=(1,2,3)))**2,dim=(0,-1))*LLJ[i])/np.prod(vJ[i].shape[1:4])*DVJ/2.0/sigmaRJ**2/ntJ
            ERJ += ERJ_
            E += ERJ_
            
            # where should I insert the diffeomorphism? I think it should be after Ai, before Ti
            Xs = emlddmm.interp(xvJ[i],(phiiJ-XVJ[i]).permute(-1,0,1,2),Xs.permute(-1,0,1,2)).permute(1,2,3,0) + Xs
            # now translate
            Xs = (TJi[:3,:3]@Xs[...,None])[...,0] + TJi[:3,-1]
            
            # now evaluate the projection in this space
            trap_i = i
            P = trapezoid(Xs[...,0],trap_i,J,dI,dslice)
            
            # now 3D back to atlas
            Xs = (Ai[:3,:3]@Xs[...,None])[...,0] + Ai[:3,-1]
            # shape
            Xs = emlddmm.interp(xv,(phii-XV).permute(-1,0,1,2),Xs.permute(-1,0,1,2)).permute(1,2,3,0)+Xs
            # now interpolate
            phiI_ = emlddmm.interp(xI,I,Xs.permute(-1,0,1,2))
            phiI_ = phiI_*P
            
    
            # now we map contrast
            phiI_r = phiI_.reshape(phiI_.shape[0],-1)
            B = torch.cat( (torch.ones_like(phiI_r[0])[None],  phiI_r))
            Ji = J[i]
            Ji_r = Ji.reshape((Ji.shape[0],-1))        
    
            # we also want to map them back, use P as a weight, or for now just add them
            coeffs_ = torch.linalg.solve(B@(B*W[i].reshape(1,-1)).T + torch.eye(B.shape[0],device=device,dtype=dtype)*0.1,
                                         (Ji_r@(B*W[i].reshape(1,-1)).T).T).T
            coeffs.append(coeffs_)
            fphiI_ = (coeffs_@B).reshape(Ji.shape)
            err_ = (fphiI_ - J[i])
            
            EM += torch.sum(err_**2*W[i])*DJs[i]
            E += EM
        
            # now let's contribute to Jrecon if I'm going to draw
            if draws[it]:#not it%ndraw:
                
                with torch.no_grad():
                    start1 = time.time()
                    phiI.append(phiI_)
                    fphiI.append(fphiI_)
                    err.append(err_)
                
                    # first calc phi
                    if i == 0:
                        phi = torch.clone(XV)
                        for t in reversed(range(nt)):
                            Xs = XV+v[t]/nt
                            phi = (emlddmm.interp(xv,(phi-XV).permute(-1,0,1,2),Xs.permute(-1,0,1,2))).permute(1,2,3,0) + Xs
                        
                    # calculate phiJ
                    phiJ = torch.clone(XVJ[i])
                    for t in reversed(range(ntJ)):
                        Xs = XVJ[i]+vJ[i][t]/ntJ
                        phiJ = (emlddmm.interp(xvJ[i],(phiJ-XVJ[i]).permute(-1,0,1,2),Xs.permute(-1,0,1,2))).permute(1,2,3,0) + Xs
                    
                    
                    # diffeo                
                    Xs = emlddmm.interp(xv,(phi-XV).permute(-1,0,1,2),XI.permute(-1,0,1,2)).permute(1,2,3,0) + XI
                    # affine
                    Xs = (A[:3,:3]@Xs[...,None])[...,0] + A[:3,-1]
                    # translation T                
                    Xs = (TJ[i][:3,:3]@Xs[...,None])[...,0] + TJ[i][:3,-1]
                    # diffeo J
                    Xs = emlddmm.interp(xvJ[i],(phiJ-XVJ[i]).permute(-1,0,1,2),Xs.permute(-1,0,1,2)).permute(1,2,3,0) + Xs
                    # affine AJ
                    Xs = (AJ[i][:3,:3]@Xs[...,None])[...,0] + AJ[i][:3,-1]
                    # reconstruct
                    Jrecon += emlddmm.interp(xJ[i],J[i]*P,Xs.permute(-1,0,1,2),padding_mode='zeros')
                    Precon += emlddmm.interp(xJ[i],P[None],Xs.permute(-1,0,1,2),padding_mode='zeros')[0]
                    drawtime += time.time()-start1
            
        # other saved variables
        Lsave.append( A[:3,:3].clone().detach().cpu().ravel().numpy() )
        Tsave.append( A[:3,-1].clone().detach().cpu().ravel().numpy() )
        LJsave.append(AJ[:,:3,:3].clone().detach().cpu().ravel().numpy())
        TJsave.append(AJ[:,:3,-1].clone().detach().cpu().ravel().numpy())
        
        # gradient and update
        Esave.append([E.item(),(E-ER).item(),ER.item(),ERJ.item()])
        E.backward()
        maxvsave.append(torch.max(torch.sqrt(torch.sum(v**2,-1))).item())
        maxvJsave.append([torch.max(torch.sqrt(torch.sum(vJ[i]**2,-1))).item() for i in range(len(J))])
        
        with torch.no_grad():
            gA = gA_from_gid(gid,A)
            Agrad = torch.linalg.solve(gA,A.grad[:3].reshape(-1)).reshape(3,4)
            A[:3] -= eA*Agrad                        
            A.grad.zero_()
            AJgrad = AJ.grad[:,:3]
            
            for i in range(len(J)):
                gJA = gA_from_gid(gJid,AJ[i])
                AJgradi = torch.linalg.solve(gJA,AJgrad[i].reshape(-1)).reshape(3,4)
                
                AJ[i,:3] -= eAJ*AJgradi
            AJ.grad.zero_()
            
            # vgrad
            if it >= vstart:
                vgrad = v.grad
                # we need to blur it (times Kpre?)        
                vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2,3))*((K*Kpre)[...,None]),dim=(1,2,3)).real
                
                v -= vgrad*ev
            v.grad.zero_()
            
            for i in range(len(J)):
                if it >= vJstart:
                    vJgrad = vJ[i].grad
                    # we need to blur it (times Kpre?)        
                    vJgrad = torch.fft.ifftn(torch.fft.fftn(vJgrad,dim=(1,2,3))*((KJ[i]*KJpre[i])[...,None]),dim=(1,2,3)).real
                    
                    vJ[i] -= vJgrad*evJ
                vJ[i].grad.zero_()
            
        with torch.no_grad():
            start1 = time.time()
            if draws[it]:#not it%ndraw:        
                Jrecon /= Precon
                Jrecon[Precon[None]==0]=0
                
                toshow = [_.clone().detach().cpu() for _ in phiI]
                _ = draw_stack(toshow,xJs,vmin=0.0,fig=figPhiI)
                figPhiI.suptitle('phiI')
                toshow = [_.clone().detach().cpu() for _ in fphiI]
                _ = draw_stack(toshow,xJs,vmin=vminJ,vmax=vmaxJ,fig=figFPhiI)
                figFPhiI.suptitle('fphiI')
    
                toshow = [_.clone().detach().cpu() for _ in err]
                _ = draw_stack(toshow,xJs,vmin=-vmaxJ*0.5,vmax=vmaxJ*0.5,fig=figErr)
                figErr.suptitle('err')
                if it == 0:
                    toshow = [_.clone().detach().cpu() for _ in J]
                    _ = draw_stack(toshow,xJs,vmin=vminJ,vmax=vmaxJ,fig=figJ)
                    figJ.suptitle('J')
    
                _ = emlddmm.draw(Jrecon,xI,vmin=vminJ,vmax=vmaxJ,fig=figJRecon)
                figJRecon.suptitle('Jrecon')
                if brain == 'TisR21':
                    emlddmm.draw(torch.cat((I/torch.amax(I,dim=(1,2,3),keepdims=True)*0.5,(Jrecon-vminJ)/(vmaxJ-vminJ))),[x/1000 for x in xI], vmin=0.0, vmax=1.0, fig=figRGB, n_slices=8, slices_start_end=[(I.shape[1]//9*3,I.shape[1]//9*5),(0,int(I.shape[2]*1.2)//2),None])
                else:
                    _ = emlddmm.draw(torch.cat((I/torch.amax(I,dim=(1,2,3),keepdims=True)*0.5,(Jrecon-vminJ)/(vmaxJ-vminJ))),xI,vmin=0.0,vmax=1.0,fig=figRGB)
                
                figRGB.suptitle('Jrecon and atlas')
                
                _ = emlddmm.draw(v[0].permute(-1,0,1,2),xv,fig=figv)
                figv.suptitle('v0')
                
                axE[0].cla()
                axE[0].plot(Esave)
                axE[0].set_title('Energy')
                axE[0].legend(['E','EM','ER','ERJ'])
                axE[1].cla()
                axE[1].plot([e[-2:] for e in Esave])
                axE[1].set_title('Regularization only')
                
                axP[0].cla()
                axP[0].plot(Lsave)
                axP[0].set_title('L')
                axP[1].cla()
                axP[1].plot(Tsave)
                axP[1].set_title('T')
                axP[2].cla()
                axP[2].plot(LJsave)
                axP[2].set_title('LJ')
                axP[3].cla()
                axP[3].plot(TJsave)
                axP[3].set_title('TJ')
                axP[4].cla()
                axP[4].plot(maxvsave)
                axP[4].set_title('max v')
                axP[5].cla()
                axP[5].plot(maxvJsave)
                axP[5].set_title('max vJ')
                
                _ = figPhiI.canvas.draw()
                _ = figFPhiI.canvas.draw()
                _ = figErr.canvas.draw()
                _ = figJ.canvas.draw()
                _ = figJRecon.canvas.draw()
                _ = figRGB.canvas.draw()
                _ = figE.canvas.draw()
                _ = figP.canvas.draw()
                _ = figv.canvas.draw()
                
                # save in the outdir
                figRGB.savefig(os.path.join(outdir,f'rgb_it_{it:05d}.jpg'), dpi = 500)
                figJRecon.savefig(os.path.join(outdir,f'recon_it_{it:05d}.jpg'), dpi = 500)
                drawtime += time.time() - start1
    
        # Save all relevant figures at the first and final iterations
        if it == range(niter)[-1] or it == range(niter)[0]:
            figPhiI.savefig(os.path.join(outdir,f'figPhiI_it_{it}.jpg'), dpi = 500)
            figFPhiI.savefig(os.path.join(outdir,f'figFPhiI_it_{it}.jpg'), dpi = 500)
            figErr.savefig(os.path.join(outdir,f'figErr_it_{it}.jpg'), dpi = 500)
            figJ.savefig(os.path.join(outdir,f'figJ_it_{it}.jpg'), dpi = 500)
            figJRecon.savefig(os.path.join(outdir,f'figJrecon_it_{it}.jpg'), dpi = 500)
            figRGB.savefig(os.path.join(outdir,f'figRGB_it_{it}.jpg'), dpi = 500)
            figE.savefig(os.path.join(outdir,f'figE_it_{it}.jpg'), dpi = 500)
            figP.savefig(os.path.join(outdir,f'figP_it_{it}.jpg'), dpi = 500)
            figv.savefig(os.path.join(outdir,f'figv_it_{it}.jpg'), dpi = 500)

        if it % 100 == 0:
            print(f'it {it}')
    
    # =================================================================================
    # ===== (7) Save reconstructed brain (.vtk) and the computed transform (.npz) =====
    # =================================================================================
    emlddmm.write_data(os.path.join(outdir,f'{brain}_reconstructed.vtk'),xI,Jrecon,'reconstructed')
    
    vJ_ = np.empty(len(vJ),object)
    vJ_[:] = [v_.clone().detach().cpu().numpy() for v_ in vJ]
    
    xv_ = np.empty(len(xv),object)
    xv_[:] = [x_.clone().detach().cpu().numpy() for x_ in xv]
    
    xvJ_ = np.empty(len(xvJ),object)
    xvJ_[:] = [[x.clone().detach().cpu().numpy() for x in xvJi] for xvJi in xvJ]
    
    np.savez(os.path.join(outdir,'transformation_outputs.npz'), 
            xv=xv_,
            v=v.clone().detach().cpu().numpy(),
            xvJ=xvJ_,
            vJ=vJ_, # why can't I save this (ok fixed)
            A=A.clone().detach().cpu().numpy(),
            AJ=AJ.clone().detach().cpu().numpy(),
            TJ=[t.clone().detach().cpu().numpy() for t in TJ],
            dtype=object
            )
    
    print(f'Saved {brain}_reconstructed.vtk and transformation_outputs.npz in {outdir}')

if __name__ == '__main__':
    main()