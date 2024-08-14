"""
Contains a set of functions used to assist with the preprocessing and computation steps in step1_atlas_registration.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def str_to_2D_mat(mat):
    """Converts a string of the form [[x,x,x,x],[x,x,x,x],[x,x,x,x],[x,x,x,x]] into a 4x4 matrix, where each x represents a number

    Parameters
    ----------
    mat : str
        A string of the form [[x,x,x,x],[x,x,x,x],[x,x,x,x],[x,x,x,x]], where each x is a scalar

    Returns
    -------
    out_arr : torch.tensor
        A 4x4 matrix whose elements were extracted from 'mat'
    """
    mat = mat.replace('[','')
    mat = mat.split(']')
    mat[1] = mat[1][1:]
    mat[2] = mat[2][1:]
    mat[3] = mat[3][1:]
    mat = mat[:4]

    out_arr = []
    out_arr.append(list(np.fromstring(mat[0], sep=',')))
    out_arr.append(list(np.fromstring(mat[1], sep=',')))
    out_arr.append(list(np.fromstring(mat[2], sep=',')))
    out_arr.append(list(np.fromstring(mat[3], sep=',')))
    out_arr = np.array(out_arr)
    
    return out_arr

def clip(I):
    """Any pixel greater than 1, set to 1 and any pixel less than 0, set to 0

    Parameters
    ----------
    I : numpy.ndarray
        An image whose pixel values are expected to lie in the range [0,1], but may have some pixels slightly out of that range

    Returns
    -------
    I : numpy.ndarray
        An image whose pixel values lie in the range [0,1]
    """
    I = np.copy(I)
    I[I<0] = 0
    I[I>1] = 1
    return I
    
def draw_stack(Js,xJs,fig=None,**kwargs):
    """This function generates a matplotlib.pyplot.fig of size len(xJs) x 3 where the ith
    row shows the coronal, axial, and sagittal planes of the data stored in xJs[i].
    
    Parameters
    ----------
    Js : list
        A list of N 3D images, where the ith image is size N(slices)_i x N(row)_i x N(col)_i
    xJs : list
        A list of the voxel locations for every image in the list Js, where the ith element is a list of 3 elements. The first element of the triple of the triple contains the location of each slice, in a 1D array of length N(slices)_i. The second element of the triple contains the location of each row, in a 1D array of lengt N(rows)_i. The third element of the triple contains the location of each col, in a 1D array of lengt N(cols)_i
    fig : matplotlib.pyplot.fig
        A figure on which to overlay additional visualizations.
    kwargs : list
        Additional arguments to modify how xJs and Js are visualized on fig;
        These arguments are passed on to ax.imshow()

    Returns
    -------
    matplotlib.pyplot.fig
        A figure of size len(xJs) x 3 displaying the coronal, axial, and sagittal 
        planes of the data stored in xJs
    
    """
    
    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = np.min([np.min(np.array(J)) for J in Js])
    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = np.min([np.max(np.array(J)) for J in Js])    
    print(kwargs)
    
    
    n = len(Js)
    if fig is None:
        figsize = (5,10)
        figsize = None
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()
    
    
    for i in range(n):
        if i == 0:
            ax0 = fig.add_subplot(n,3,3*i+1)
            ax = ax0
        else:
            ax0 = None
            ax = fig.add_subplot(n,3,3*i+1,sharex=ax0,sharey=ax0)
        # note I make a copy here
        toshow = np.array(Js[i][:,Js[i].shape[1]//2]).transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = np.concatenate((toshow,toshow,toshow),-1)
        elif toshow.shape[-1] == 2:
            toshow = np.stack((toshow[...,0],toshow[...,1],toshow[...,0]),-1)
        else:
            toshow = toshow[...,:3]
        toshow -= vmin
        toshow /= vmax-vmin
        xJ = xJs[i]
        dJ = [x[1]-x[0] for x in xJ]
        extent = (xJ[-1][0]-dJ[-1]*0.5,xJ[-1][-1]+dJ[-1]*0.5,
                 xJ[-2][-1]+dJ[-2]*0.5,xJ[-2][0]-dJ[-2]*0.5)
        ax.imshow(clip(toshow),extent=extent,**kwargs)
        #if i < n-1:
        #    ax.set_xticks([])
                
        if i == 0:
            ax1 = fig.add_subplot(n,3,3*i+2)
            ax = ax1
        else:
            ax1 = None
            ax = fig.add_subplot(n,3,3*i+2,sharex=ax1,sharey=ax1)
        toshow = np.array(Js[i][:,:,Js[i].shape[2]//2]).transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = np.concatenate((toshow,toshow,toshow),-1)
        elif toshow.shape[-1] == 2:
            toshow = np.stack((toshow[...,0],toshow[...,1],toshow[...,0]),-1)
        else:
            toshow = toshow[...,:3]
        toshow -= vmin
        toshow /= vmax-vmin
        
        
        extent = (xJ[-1][0]-dJ[-1]*0.5,xJ[-1][-1]+dJ[-1]*0.5,
                 xJ[-3][-1]+dJ[-3]*0.5,xJ[-3][0]-dJ[-3]*0.5)        
        ax.imshow(clip(toshow),extent=extent,**kwargs)
        ax.set_aspect('auto')
        #if i < n-1:
        #    ax.set_xticks([])
        
        
        if i == 0:
            ax2 = fig.add_subplot(n,3,3*i+3)
            ax = ax2
        else:
            ax2 = None
            ax = fig.add_subplot(n,3,3*i+3,sharex=ax2,sharey=ax2)
        toshow = np.array(Js[i][:,:,:,Js[i].shape[3]//2]).transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = np.concatenate((toshow,toshow,toshow),-1)
        elif toshow.shape[-1] == 2:
            toshow = np.stack((toshow[...,0],toshow[...,1],toshow[...,0]),-1)            
        else:
            toshow = toshow[...,:3]
        toshow -= vmin
        toshow /= vmax-vmin
        extent = (xJ[-2][0]-dJ[-2]*0.5,xJ[-2][-1]+dJ[-2]*0.5,
                 xJ[-3][-1]+dJ[-3]*0.5,xJ[-3][0]-dJ[-3]*0.5)
        ax.imshow(clip(toshow),extent=extent,**kwargs)
        ax.set_aspect('auto')
        #if i < n-1:
        #    ax.set_xticks([])
        plt.subplots_adjust(wspace=0,hspace=0)
        
    return fig,None # later I will return ax

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

def small_to_big(a):
    '''The composition of two affine transforms A and B as 4x4 matrices with 12 parameters each, can be written as matrix multiplication A@B. i.e., the matrix B can be vectorized into a 12x1 vector, and the matrix A can be matrixized into a 12x12 matrix.  Then the same operation could be written as A_big @ B_vector. This is used when computing the components of a metric tesor for affine transformations, which is a 12x12 matrix.
    
    Parameters
    ----------
    a : torch.tensor
        A 4x4 affine transformation matrix
        
    Returns
    -------
    A : torch.tensor
        A 12x12 affine transformation matrix, written in the alternative format (As described above)
    '''
    
    A = torch.zeros(12,12,dtype=a.dtype,device=a.device)
    device = a.device
    dtype = a.dtype
    count0 = 0
    for i0 in range(3):
        for j0 in range(4):
            E0 = ((torch.arange(4,device=device,dtype=dtype)==i0)[:,None]*(torch.arange(4,device=device,dtype=dtype)==j0)[None,:]).to(dtype)
            
            A[:,count0] = (a@E0)[:3,:4].ravel()
            count0 += 1
    return A

def gA_from_gid(gid,A):
    """This is a helper function that computes a metric tensor on the space of affine transformation matrices at the location, A, as a function of the metric tensor at the location, Identity. This approach allows us to perform natural gradient descent and choose a single stepp size for linear and translation parameters during optimization. See "An optical flow based left-invariant metric for natural gradient descent in affine image registration" (https://doi.org/10.3389/fams.2021.718607)
    
    Parameters
    ----------
    gid : torch.tensor
        A 12x12 matrix representing an inner product between tangent vectors to the affine transformation group at Identity
    A : torch.tensor
        A 4x4 affine transformation matrix describing the point on the affine transformation group that we want to pull back our inner product to

    Returns
    -------
    gA : torch.tensor
        A 12x12 matrix representing an inner product between tangent vectors to the affine transformation group at the point, A
    
    """
    A_ = small_to_big(torch.linalg.inv(A))
    gA = A_.T@gid@A_
    return gA