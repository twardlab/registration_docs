"""High resolution mouse brain image to low resolution mouse brain image registration

This script is intended to be used at the command line and will save the initial and final registration results for aligning 2 3D mouse brain slices of the same area at different resoluions. This script performs step 2 of 3 in the registration pipeline.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
import sys
from scipy.interpolate import interpn

def main():
    """
    Command line arguments
    ----------------------
    dataset : str
        The name of the dataset to be registered. This will be included in the names of the majority of the output files.
    -low_path : str
        The location of the low resolution .npz file to be aligned to the high resolution .npz file
    -high_path : str
        The location of the high resolution .npz file
    -high_num : str
        The slice ID number of the high resolution image
    -outdir : str
        The path to the directory where all outputs will be stored
    -shifts : list of 3 integers
        Shifts in the Anterior/Posterior, Left/Right, and Dorsal/Ventral directions to be applied to the low resolution image
    -e_path : str
        The location of the custom Python library 'emlddmm', which can be cloned from GitHub at https://github.com/twardlab/emlddmm
    -d_path : str
        The location of the custom Python library 'donglab_workflows', which be cloned from GitHub at https://github.com/twardlab/donglab_workflows
    -gamma : bool
        Default - False; If present, apply a gamma correction to the file located at high_path.
    -checkInit : bool
        Default - False; If present, script will save initial guesses and terminate before registration begins.
    -zeroMean : bool    
        Default - True; If present, use the zero mean convention when loading input data.
    -useRigidTransform : bool
        Default - False; If present, use a rigid transform instead of an affine transform for registration.
        
    Raises
    ------
    Exception
        If the only objects in the .npz file located at '-high' are not [I, I2, xI, w]
    Exception
        If the only objects in the .npz file located at '-low' are not [I, I2, xI, w]
    """

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset', type = str, help = 'Name of dataset to be processed')
    parser.add_argument('-low_path', type = str, required = True, help = 'location of low res .npz file')
    parser.add_argument('-high_path', type = str, required = True, help = 'location of high res .npz file')
    parser.add_argument('-high_num', type = str, required = True, help = 'The slice number of the high res image')
    parser.add_argument('-outdir', type = str, required = True, help = 'Directory for all output figures to be saved')
    parser.add_argument('-shifts', type = int, nargs = 3, required = True, help = 'Shifts in the Anterior/Posterior, Left/Right, and Dorsal/Ventral directions to be applied to the low res image')
    parser.add_argument('-e_path', type = str, required = True, help = 'The directory containing the emlddmm library from Github')
    parser.add_argument('-d_path', type = str, required = True, help = 'The directory containing the donglab_workflows library from Github')
    
    parser.add_argument('-gamma', action = 'store_true', help = 'If present, apply a gamma correction to the file located at high_path')
    parser.add_argument('-checkInit', action = 'store_true', help = 'If present, script will save initial guesses and terminate before registration begins')
    parser.add_argument('-zeroMean', action = 'store_false', help = 'If present, use the zero mean convention when loading input data')
    parser.add_argument('-useRigidTransform', action = 'store_true', help = 'If present, use a rigid transform instead of an affine transform for registration')
    
    args = parser.parse_args()
    
    brain = args.dataset
    atlas_name = args.low_path
    target_name = args.high_path
    target_key = args.high_num
    outdir = args.outdir
    shifts = args.shifts
    performGamma = args.gamma
    checkInit = args.checkInit
    zeroMean = args.zeroMean
    useRigidTransform = args.useRigidTransform
    
    # Import custom libraries from local device (Repo should have already been cloned from Github)
    sys.path.append(args.d_path)
    import donglab_workflows as dw
    sys.path.append(args.e_path)
    import emlddmm
    
    # (0) Perform checks on accepted arguments
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # =============================================
    # ===== (1) Load the high resolution data =====
    # =============================================
    target_data = np.load(target_name,allow_pickle=True)
    
    # Check that expected objects are in target_data
    target_data_keys = [k for k in target_data]
    if 'I' not in target_data_keys or 'I2' not in target_data_keys or 'xI' not in target_data_keys or 'w' not in target_data_keys:
        raise Exception(f'Missing object in file location at {target_data}. Expected [\'I\', \'I2\', \'xI\', \'w\'], but read {target_data_keys}')
    if len(target_data_keys) != 4:
        raise Exception(f'Incorrect number of objects in file located at {target_data}. Expected 4, but there were {len(target_data_keys)}')
    
    
    J = target_data['I'][None]
    J = J.astype(np.float32)
    W = target_data['w']
    
    # Normalization, so that most of the data is [0,1]
    J -= np.quantile(J[W[None]>0.9],0.1)
    J /= np.quantile(J[W[None]>0.9],0.999)
    J[W[None]<=0.90] = 0.0 # there should be nothing less than 0
        
    J[J<0] = 0.0
    if zeroMean:
        xJ = [d - np.mean(d) for d in target_data['xI']]
    else:
        xJ = target_data['xI']
        
    dJ = [x[1] - x[0] for x in xJ]
    J0 = np.copy(J)
    
    # For target, only pad in the slice direction
    npad = 8
    J = np.pad(J,  ((0,0),(npad,npad),(0,0),(0,0)) )
    W = np.pad(W,  ((npad,npad),(0,0),(0,0)) , mode='constant',constant_values=1.0) 
    for i in range(npad):    
        xJ = [ np.concatenate( (xJ[0][0][None]-dJ[0], xJ[0],xJ[0][-1][None]+dJ[0])), xJ[1], xJ[2]  ]
    
    # fig,ax = plt.subplots()
    # ax.hist(J.ravel(),bins=100)
    
    # fig,ax = emlddmm.draw(J,xJ,vmin=0,vmax=1)
    # fig.suptitle('Downsampled lightsheet data')
    # # figopts = {'dpi':300,'quality':90}
    # figopts = {'dpi':300}
    # fig.savefig(outdir + '_down.jpg',**figopts)
    # fig.canvas.draw()
    
    
    # ============================================
    # ===== (2) Load the low resolution data =====
    # ============================================
    I = []
    atlas_data = np.load(atlas_name,allow_pickle=True)
    
    # Check that expected objects are in atlas_data
    atlas_data_keys = [k for k in atlas_data]
    if 'I' not in atlas_data_keys or 'I2' not in atlas_data_keys or 'xI' not in atlas_data_keys or 'w' not in atlas_data_keys:
        raise Exception(f'Missing object in file location at {atlas_data}. Expected [\'I\', \'I2\', \'xI\', \'w\'], but read {atlas_data_keys}')
    if len(atlas_data_keys) != 4:
        raise Exception(f'Incorrect number of objects in file located at {atlas_data}. Expected 4, but there were {len(atlas_data_keys)}')
    
    if zeroMean:
        xI = [d - np.mean(d) for d in atlas_data['xI']]
    else:
        xI = atlas_data['xI']
    
    I_ = atlas_data['I'].astype(np.float32)    
    
    if 'TME07-1' in brain:
        I_ -= np.quantile(I_[atlas_data['W']>0.9],0.1)
        I_ /= np.quantile(I_[atlas_data['W']>0.9],0.999)      
        I_[atlas_data['W']<=0.9] = 0.0
    else:
        I_ -= np.quantile(I_[atlas_data['w']>0.9],0.1)
        I_ /= np.quantile(I_[atlas_data['w']>0.9],0.999)
        I_[atlas_data['w']<=0.9] = 0.0
        
    I.append(I_)
        
    I = np.concatenate(I)    
    if I.ndim == 3:
        I = I[None]
    dI = [x[1] - x[0] for x in xI]
    XI = np.meshgrid(*xI,indexing='ij')
    
    # Pad the low resolution image
    npad = 4
    I = np.pad(I,  ((0,0),(npad,npad),(npad,npad),(npad,npad)) )
    for i in range(npad):
        xI = [ np.concatenate(   (x[0][None]-d,x,x[-1][None]+d)   ) for d,x in zip(dI,xI)]
    
    
    # fig,ax = emlddmm.draw(I,xI,vmin=0,vmax=1)
    # fig.canvas.draw()
    
    # fig,ax = plt.subplots()
    # ax.hist(I.ravel(),100,log=True)
    # fig.canvas.draw()
    
    
    # ============================================================================
    # ===== (3) Apply gamma correction to high resolution image if necessary =====
    # ===========================================================================
    # Gamma correction for 30x images with nonuniform brightness
    #     J = ((J - np.min(J))/(np.max(J) - np.min(J)))**0.5
    #     J = np.clip(J,0,1)
    if performGamma:
        # for step 2 we will want to do some contrast adjustment on the 30x high res images,
        n = J.shape[1]
        nrow = np.ceil(np.sqrt(n)).astype(int)
        Jmin = np.min(J[W[None]>0.9])
        Jmax = np.max(J[W[None]>0.9])
        J_ = np.copy(J)
        J_ -= Jmin
        J_ /= Jmax-Jmin
        J_ = np.clip(J_,0,1)
        J_ = J_**0.5
        J = J_
        
        # fig,ax = plt.subplots(nrow,nrow)
        # ax = ax.ravel()
        # for i in range(n):
        #     h = ax[i].imshow(J_[i],vmin=0,vmax=1)
        # plt.colorbar(mappable=h,ax=ax)
    
        J = J_    
    
    # fig,ax = emlddmm.draw(J,xJ,vmin=0,vmax=1)
    # fig.canvas.draw()
    
    
    # fig,ax = plt.subplots()
    # ax.hist(J.ravel(),100,log=True)
    # fig.canvas.draw()
    
    
    # =========================================================
    # ===== (4) Define initial guess for affine matrix, A =====
    # =========================================================
    # initial affine
    A0 = np.eye(4)
    A0[0,-1] = np.mean(xJ[0]) - np.mean(xI[0])
    
    # Update the affine matrix with the shifts in each cardinal direction
    A0[:3,-1] = np.array(shifts)
    
    # =======================================================
    # ===== (5) Using A, generate initial guess figures =====
    # =======================================================
    
    XJ = np.meshgrid(*xJ,indexing='ij')
    tform = emlddmm.Transform(A0,direction='b')
    A0I = emlddmm.apply_transform_float(xI,I,tform.apply(XJ))
    
    fig,ax = emlddmm.draw(np.concatenate((A0I,J*0)),xJ,vmin=0)
    fig,ax = emlddmm.draw(np.max(np.concatenate((A0I,J*0)),axis=-3,keepdims=True),xJ,vmin=0) # The MIP
    fig.savefig(os.path.join(outdir,f'{target_key}_10x.jpg'))
    
    fig,ax = emlddmm.draw(np.concatenate((A0I*0,J)),xJ,vmin=0)
    fig.savefig(os.path.join(outdir,f'{target_key}_30x.jpg'))
    
    fig,ax = emlddmm.draw(np.concatenate((A0I,J)),xJ,vmin=0)
    fig.savefig(os.path.join(outdir,f'{target_key}_10x_30x.jpg'))
    
    # ================================================
    # ===== (6) Perform low-to-high registration =====
    # ================================================

    if checkInit:
        print(f'Initial results have been saved in {outdir}, terminating program now')
        return    
    
    # Run with only a linear transform (no deformation)
    start = time.time()
    
    # now we want to register
    # we'll start with an affine registration
    device = 'cpu'
    # device = 'cuda:1'
    config0 = {
        'device':device,
        'n_iter':500, 'downI':[2,2,2], 'downJ':[2,2,2],
         'priors':[0.9,0.05,0.05],'update_priors':False,
         'update_muA':0,'muA':[np.quantile(J,0.99)],
         'update_muB':0,'muB':[0.0],     
         'sigmaM':1.0,'sigmaB':2.0,'sigmaA':5.0,
         'order':1,'n_draw':50,'n_estep':3,'slice_matching':0,'v_start':1000,
         'eA':5e6,'A':A0,'full_outputs':True,'local_contrast':[8,16,16]
       }
    
    I_ = np.concatenate((I,I**2))
    if useRigidTransform:
        out = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, Amode=3, **config0)
    else:
        out = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config0)
        
    print("===========================================")
    print("Finished registration with linear transform")
    print("===========================================")
    
    # second run, with deformation
    config1 = dict(config0)
    
    config1['A'] = out['A']
    config1['eA'] = config0['eA']*0.1
    config1['a'] = 1000.0
    config1['sigmaR'] = 5e4 # 1e4 gave really good results, but try 2e4, also good, I showed this in my slides
    config1['n_iter']= 2000
    config1['v_start'] = 0
    config1['ev'] = 1e0
    config1['v_res_factor'] = config1['a']/dI[0]/2/2 # what is the resolution of v, as a multiple of that in I
    
    if useRigidTransform:
        out1 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, Amode=1, **config1)
    else:
        out1 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config1)
    print("======================================")
    print("Finished registration with deformation")
    print("======================================")
    plt.close('all')
    
    # on the next run we do less downsampling
    config2 = dict(config1)
    config2['A'] = out1['A']
    config2['n_iter'] = 1000
    config2['v'] = out1['v']
    config2['downI'] = [1,1,1]
    config2['downJ'] = [1,1,1]
    config2['ev'] = config1['ev']*0.5
    config2['eA'] = config1['eA']*0.5
    
    if useRigidTransform:
        out2 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, Amode=1, **config2)
    else:
        out2 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config2)
    print("============================================================")
    print("Finished registration with deformation and less downsampling")
    print("============================================================")
    
    # =============================================================
    # ===== (7) Save transformation.npz and final output figs =====
    # =============================================================
   
    # save the outputs
    savename = f'transformation_outputs_{brain}_{target_key}_low_to_high.npz'
    np.savez(os.path.join(outdir,savename),out=out2)
    
    out2['figI'].savefig(os.path.join(outdir,target_key + '_transformed.jpg'))
    out2['figfI'].savefig(os.path.join(outdir,target_key + '_contrast.jpg'))
    out2['figErr'].savefig(os.path.join(outdir,target_key + '_err.jpg'))
    
    fig = out2['figErr']
    axs = fig.get_axes()
    for ax in axs:
        ims = ax.get_images()
        for im in ims:
            im.set_cmap('twilight')
            clim = im.get_clim()
            lim = np.max(np.abs(clim))
            im.set_clim(np.array((-1,1))*lim)
    fig.canvas.draw()
    fig.savefig(os.path.join(outdir,target_key + '_err2.jpg'))
    
    # visualize overlap (Before optimization)
    XJ = np.meshgrid(*xJ,indexing='ij')
    tform = emlddmm.compose_sequence([
        emlddmm.Transform(A0,'b'),
                                      ],XJ)
    AI = emlddmm.apply_transform_float(xI,I,tform)
    fig,ax = emlddmm.draw(np.concatenate((AI,J)),xJ,vmin=0)
    fig.canvas.draw()
    fig.suptitle('error_before')
    fig.savefig(os.path.join(outdir,target_key + '_err3_before.jpg'), dpi=500)
    
    # visualize overlap (After optimization)
    XJ = np.meshgrid(*xJ,indexing='ij')
    tform = emlddmm.compose_sequence([emlddmm.Transform(out2['A'],'b'),
                                      emlddmm.Transform(out2['v'],'b',domain=out2['xv'])],XJ)
    AI = emlddmm.apply_transform_float(xI,I,tform)
    fig,ax = emlddmm.draw(np.concatenate((AI,J)),xJ,vmin=0)
    fig.canvas.draw()
    fig.suptitle('error_after')
    fig.savefig(os.path.join(outdir,target_key + '_err3_after.jpg'), dpi=500)

    print(f'Finished Step 2 registration in {time.time() - start:.2f}s and saved output figures and .npz files in {outdir}')


if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
    
    
