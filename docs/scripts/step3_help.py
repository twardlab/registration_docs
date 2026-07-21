"""
Contains a set of functions used to assist with the preprocessing and computation steps in step3_atlas_registration.py
"""

import numpy as np
import nibabel as nib
import os
import csv
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from math import dist
import torch
# from scipy.spatial.distance import directed_hausdorff

# here we have a backup unpickler that ignores figures in case there are version issues
import pickle
import subprocess
from numpy.lib.format import read_magic, _check_version, _read_array_header
class ClassHack():
    '''
    This class provides methods that my unpickling requires, but doesn't do anything
    '''
    def __init__(self,*args,**kwargs):
        pass
    def __call__(*args,**kwargs):
        pass
    def _remove_ax(self,*args,**kwargs):
        pass
    def _remove_legend(self,*args,**kwargs):
        pass
    def delaxes(self,*args,**kwargs):
        pass
    def _key_handler(self,*args,**kwargs):
        pass
    def pick(self,*args,**kwargs):
        pass

class Unpickler(pickle.Unpickler):
    '''
    An unpickler which can ignore old matplotlib figures stored in a dictionary
    '''
    def find_class(self, module, name):
        #print(module,name)
        if name == 'CallbackRegistry':
            #print('found callback registry')
            return ClassHack
        elif name == 'AxesStack':
            #print('found axes stack')
            return ClassHack
        elif name == '_picklable_subplot_class_constructor':
            #print('found subplot class constructor')
            return ClassHack
        elif module == 'matplotlib.figure' and name == 'Figure':
            return ClassHack    
        elif module =='matplotlib.backend_bases':
            return ClassHack
        else:
            #print('normal module name')
            return super().find_class(module,name)

def backup_unpickle(reg_file):
    """A function used to load the low-to-high registration transform file
    
    Parameters
    ---------
    reg_file : str
        A file path pointing to the location of a low-to-high registration transform file (.npz)

    Returns
    -------
    out : dict
        A dictionary containing the data stored in reg_file
    """
    output = subprocess.run(['unzip','-o',reg_file],capture_output=True)
    with open('out.npy','rb') as f:
        version = read_magic(f)
        _check_version(version)
        dtype = _read_array_header(f, version)[2]    
        out = Unpickler(f).load().item()
    
    return out

def load_swc(fname,skip=0):
    ''' load an swc file.
    
    Parameters
    ----------
    fname : str
        A string pointing to the file to load
    skip : int
        Number of lines to skip at the beginning. (default 0)
    
    Returns
    -------
    V : numpy array of float
        An array storing vertices
    E : numpy array of int
        An array storing edges.  they are ordered as parent -> this sample
    R : numpy array of float
        An arrow storing radii
   
    '''       
    
    # load an swc
    # recall columns
    # data type, sample number, structure identifier, x,y,z,r,parent
    # first get all the sample numbers, and map them to a count
    sample_numbers = []
    sample_counts = []
    count = 0
    with open(fname,'rt') as f:
        for i,line in enumerate(f):
            if i < skip or line[0] == '#':
                continue
            # print(line)
            data = line.split()
            # may be comma separated
            if len(data)==1:
                data = line.split(',')
            sample_number = int(data[0])
            sample_numbers.append(sample_number)
            sample_counts.append(count)
            count += 1
    number_to_count = {k:v for k,v in zip(sample_numbers,sample_counts)}
    count_to_number = {k:v for k,v in zip(sample_counts,sample_numbers)}
            
            
    v = {}
    e = []
    r = {}
    with open(fname,'rt') as f:
        for i,line in enumerate(f):
            # print(f'{i}: {line}')
            if i < skip or line[0] == '#':
                continue
                
            # print(line)
            data = line.split()
            # may be comma separated
            if len(data)==1:
                data = line.split(',')
            sample_number = int(data[0])
            sample_count = number_to_count[sample_number]
            x = float(data[2])
            y = float(data[3])
            z = float(data[4])
            r_ = float(data[5])
            v[sample_count] = np.array([x,y,z])
            r[sample_count] = np.array(r_)

            par = int(data[-1])            
            if par >= 0:
                par_count = number_to_count[par]
                #e.append([par,sample_number,])
                #e.append([par-1,sample_number-1,]) # move to zero indexing
                e.append([par_count,sample_count])
                pass
    # we want to put v into a numpy array
    maxkey = np.max([k for k in v])
    V = np.ones((maxkey+1,3))*np.nan
    for key in v:                
        #V[key-1] = v[key] # zero indexing
        V[key] = v[key] # zero indexing already done above
    R = np.ones((maxkey+1,))*np.nan
    for key in r:
        #R[key-1] = r[key] # zero indexing
        R[key] = r[key] # zero indexing already done above
    E = np.array(e)
    return V,E,R 

def str_to_list(mat):
    """A function which takes a list formatted as a string and converts it to a list of the correct shape

    Parameters
    ----------
    mat : str
        A string of the form '[[a],...[b,c]]' where each element in the list is a single or tuple consisting of strings containing only alphanumeric characters which correspond to the id of each slice

    Returns
    -------
    mat : list
        A list containing the elements of the input, now formatted as a list of singles and tuples with alphanumeric strings for each element
    """
    mat = mat.split('],')
    mat = [[ch if np.char.isalnum(ch) or ch == ',' else '' for ch in elem] for elem in mat]
    for i, elem in enumerate(mat):
        while '' in elem:
            elem.remove('')
        if ',' in elem:
            mat[i] = ''.join(elem).split(',')
        else:
            mat[i] = [''.join(elem)]
    return mat

def get_xyzoff(fname, brain, neuron_dir, skip=0):
    ''' Compute the offset in soma coordinates using the .swc file located at fname. 
    
    Parameters
    ----------
    fname : str
        A string pointing to the file to load
    brain : str
        The name of the dataset in which the neuron lies; Used to determine how to compute the offset from the file
    neuron_dir : str
        The directory in which fname lies; Used to determine how to compute the offset from the file
    skip : int
        Number of lines to skip at the beginning. (default 0)
    
    Returns
    -------
    xyzoff : Array of ints
        the x, y, z offsets between fname and first line of the .swc file stored at fname
    
    '''

    # load an swc, columns: data type, sample number, structure identifier, x,y,z,r,parent
    with open(fname,'rt') as f:
        for i,line in enumerate(f):
            # print(line)
            if i < skip or line[0] == '#':
                continue
                
            data = line.split()
            # may be comma separated
            if len(data)==1:
                data = line.split(',')
            sample_number = int(data[0])
            x = float(data[2])
            y = float(data[3])
            z = float(data[4])
            break

    if 'hTME' in brain or brain in ['Hpca5-2', 'Sp9-3-2', 'Trank1-2-3', 'Zswim4-1', 'MQC06-2', 'MQC09-3', 'MQC18-3', 'MQC82-2']:
        if 'Unmapped' in neuron_dir: # neuron_dir is a global variable defined in one of the first cells
            print(fname)
            _, fx, fy, fz = ((fname.split('/'))[-1].split('_')[-1]).split('-')
            fz = fz[:-4]
        else:
            _, fx, fy, fz = ((fname.split('/'))[-1].split('_')[-2]).split('-')
        
        fx, fy, fz = float(fx), float(fy), float(fz)
        return np.array([fx-x, fy-y, fz-z])
    else:
        return np.array([0.0, 0.0, 0.0])

# Get ontology from Yongsoo Kim's mouse brain segmented atlas
def get_target_ids(atlas_labels_path):
    """Return a list of the region IDs from Yongsoo Kim's atlas corresponding to all of the subregions within the CP

    Parameters
    ==========
    atlas_labels_path : str
        The file path pointing to the location of a csv file containing all of the region IDs for Yongsoo Kim's mouse brain atlas

    Returns
    =======
    region_ids : list of int
        A list of integers corresponding to the regions within the CP region of the mouse brain
    
    """
    ontology = {}
    with open(atlas_labels_path) as f:
        reader = csv.reader(f)    
        for count,row in enumerate(reader):
            if count == 0:
                headers = row
                id_ind = 0
                name_ind = headers.index('name')
                acronym_ind = headers.index('acronym')
                parent_ind = headers.index('parent_id')
                continue
            id_ = row[id_ind]
            id_ = int(id_)
            name = row[name_ind]
            acronym = row[acronym_ind]
            
            parent = row[parent_ind]
            if parent:
                parent = int(parent)
            else:
                parent = None
            ontology[id_] = dict(id=id_,name=name,acronym=acronym,parent=parent)
            
    # find all labels corresponding to CP + ACB
    region_ids = []
    for id_ in ontology:    
        if 'caudoputamen' in ontology[id_]['name'].lower():
            region_ids.append(id_)
            # r_name = ontology[id_]['name']
            # print(f'Appended: {r_name} ({id_})')
        # if 'accumbens' in ontology[id_]['name'].lower():
        #     region_ids.append(id_)
            # r_name = ontology[id_]['name']
            # print(f'Appended: {r_name} ({id_})')

    return region_ids

def atlas_to_cp_mask(atlas_vis, region_ids):
    """Convert an integer-valued segmentation of the mouse brain into a ternary mask (0:background, 1:brain (-CP), 2: CP)

    Parameters:
    ===========
    atlas_vis : array of ints
        An array of size [X,Y,Z] where each element contains the subregion ID of the corresponding voxel in the mouse brain
    region_ids : array of ints
        An array containing all of the region IDs corresponding to the CP

    Returns:
    ========
    all_non_cp : array of ints
        An array of size [X,Y,Z] where each element corresponds to the type of data at each voxel in the ternarized mouse brain
    """
    all_cp = np.zeros(np.shape(atlas_vis))
    for id in region_ids:
        all_cp = np.logical_or(all_cp,atlas_vis==id)

    all_non_cp = np.zeros(np.shape(atlas_vis))
    all_non_cp[atlas_vis != 0] = 1.0
    all_non_cp[all_cp == 1.0] = 2.0
    
    return all_non_cp

def compute_all_dice_scores(atlas_vis, manual_vis):
    """Compute the Dice score between the registered atlas segmentation and a manual anatomical segmentation for the CP and for the whole brain

    Parameters
    ==========
    atlas_vis : array of ints
        An array of size [X,Y,Z] where each element corresponds to the type of data at each voxel in the ternarized mouse brain; Registered atlas segmentation
    manual_vis : array of ints
        An array of size [X,Y,Z] where each element corresponds to the type of data at each voxel in the ternarized mouse brain; Manual anatomical segmentation

    Returns
    =======
    dice_cp : float
        The dice score when comparing the registered atlas segmentation and a manual anatomical segmentation for the CP
    dice_wb : float
        The dice score when comparing the registered atlas segmentation and a manual anatomical segmentation for the whole brain        
    """
    # Compute Dice Score for CP+ACB
    atlas_vis_cp = atlas_vis.copy()
    atlas_vis_cp[atlas_vis_cp==1.0] = 0
    atlas_vis_cp[atlas_vis_cp==2.0] = 1.0
    
    manual_vis_cp = manual_vis.copy()
    manual_vis_cp[manual_vis_cp==1.0] = 0
    manual_vis_cp[manual_vis_cp==2.0] = 1.0

    dice_cp = compute_dice_score(atlas_vis_cp, manual_vis_cp)

    # Compute Dice Score for Whole Brain 
    atlas_vis_cp = atlas_vis.copy()
    atlas_vis_cp[atlas_vis_cp == 2.0] = 1.0
    
    manual_vis_cp = manual_vis.copy()
    manual_vis_cp[manual_vis_cp==2.0] = 1.0

    dice_wb = compute_dice_score(atlas_vis_cp, manual_vis_cp)

    return dice_cp, dice_wb

def compute_dice_score(atlas_vis, manual_vis):
    """Compute the Dice score between the registered atlas segmentation and a manual anatomical segmentation for the CP and for the whole brain

    Parameters
    ==========
    atlas_vis : array of ints
        An array of size [X,Y,Z] where each element corresponds to the type of data at each voxel in a binarized mouse brain; Registered atlas segmentation
    manual_vis : array of ints
        An array of size [X,Y,Z] where each element corresponds to the type of data at each voxel in a binarized mouse brain; Manual anatomical segmentation

    Returns:
    ========
    out : float
        The dice score computed between atlas_vis and manual_vis
    """
    atlas_area = np.sum(atlas_vis)
    manual_area = np.sum(manual_vis)

    intersection = np.logical_and(atlas_vis, manual_vis)
    int_area = np.sum(intersection)

    return (2*int_area) / (atlas_area + manual_area)
    
def npz_to_Nifti1(npzPath, outPath, affine, key=None, saveImage = False):
    """Convert an (.npz) file into a Nifti1 (.nii.gz) file using the provided affine matrix

    Parameters
    ==========
    npzPath : str
        A file path pointing to an npz file containing image data
    outPath : str
        A file path pointing to the desired output location for the converted Nifti1 file
    affine : List of float
        A 4x4 affine matrix used to initialzie the Nifti1 Image
    key : str
        Default - None; The key corresponding to the image data stored at 'npzPath'. If None, use the first key.
    saveImage : bool
        Default - False; If True, save the Nifti1 file at outPath
    """
    
    I = np.load(npzPath)
    
    if key == None:
        key = [k for k in I][0]

    I = I[key]

    I = nib.Nifti1Image(I, affine = affine)
    I.header['xyzt_units'] = 3 # Sets the unit type to um (Default: 0 == Unknown)

    if saveImage:
        nib.save(I, outPath)
        print(f'Saved image at {outPath}')

def compute_all_hausdorff_scores(atlas_bound_coords, manual_bound_coords):
    """ Generate a NxM matrix of Euclidean distances corresponding to the Euclidean distances between a set of N points and another set of M points, extract the N+M Hausdorff distances, and return several performance metrics corresponding to the similarity of the N points from atlas_bound_coords and manual_bound_coords.

    Parameters
    ==========
    atlas_bound_coords : array of float
       A 3xN array containing the x,y,z coordinates for all of the points defining a regional boundary in the registered mouse brain atlas 
    manual_bound_coords : array of float
       A 3xM array containing the x,y,z coordinates for all of the points defining a regional boundary in the manual anatomical segmentation

    Returns
    =======
    h_dist : float
        The Hausdorff Score; i.e. the maximum of all the Hausdorff distances
    h_dist_50 : float
        The 50th percentile from all of the Hausdorff distances
    h_dist_95 : float
        The 95th percentile from all of the Hausdorff distances
    all_min_dist : array of float
        A [N+M] array of all the Hausdorff distances 
    """

    atlas_bound_coords = atlas_bound_coords.reshape((np.shape(atlas_bound_coords)[1], np.shape(atlas_bound_coords)[0]))
    manual_bound_coords = manual_bound_coords.reshape((np.shape(manual_bound_coords)[1], np.shape(manual_bound_coords)[0]))

    dist_matrix = [[dist(a,m) for a in atlas_bound_coords] for m in manual_bound_coords]    
    min0 = np.min(dist_matrix,axis=0)
    min1 = np.min(dist_matrix,axis=1)
    all_min_dist = np.concatenate([min0,min1])

    h_dist = np.max(all_min_dist)
    h_dist_50 = np.percentile(all_min_dist, 50)  
    h_dist_95 = np.percentile(all_min_dist, 95)
        
    return h_dist, h_dist_50, h_dist_95, all_min_dist

def ternary_to_boundary(ternary_mask):
    """ Convert a 3D ternarized image into the boundaries between regions 0,1+2 and 0+1,2.

    Parameters
    ==========
    ternary_mask : array of int
        An array of size [X,Y,Z] where each element corresponds to the type of data at each voxel in the ternarized mouse brain

    Returns
    =======
    cp_3d_bound : array of int
        An array of size [X,Y,Z] where each element is either 1 if it is along the boundary of the CP and 0 otherwise
    wb_3d_bound : array of int
        An array of size [X,Y,Z] where each element is either 1 if it is along the outer boundary of the cortex and 0 otherwise        
    """

    num_slices = np.shape(ternary_mask)[0]
    
    # Set non-CP to 0 and CP to 1 and compute boundary around CP
    t_copy = ternary_mask.copy()
    t_copy[t_copy == 1.0] = 0
    t_copy[t_copy == 2.0] = 1.0
    cp_3d_bound = []
    for idx in range(num_slices):
        ternary_slice = t_copy[idx,:,:]
        ternary_vis_erode = binary_erosion(ternary_slice)
        ternary_bound = ternary_slice - ternary_vis_erode
        cp_3d_bound.append(ternary_bound)
    cp_3d_bound = np.stack(cp_3d_bound)

    # Set CP to 1 and compute boundary around WB
    t_copy = ternary_mask.copy()
    t_copy[t_copy == 2.0] = 1.0
    wb_3d_bound = []
    for idx in range(num_slices):
        ternary_slice = t_copy[idx,:,:]
        ternary_vis_erode = binary_erosion(ternary_slice)
        ternary_bound = ternary_slice - ternary_vis_erode
        wb_3d_bound.append(ternary_bound)
    wb_3d_bound = np.stack(wb_3d_bound)    
    
    return cp_3d_bound, wb_3d_bound
    
def plot_all_slices(atlas_mask, curr_mask, low_img_path, outdir, outdir_fname):
    """Plot the integer-valued masks from the registered mouse brain atlas and the manual anatomical segmentation over the corresponding microscopy image.

    Parameters
    ==========
    atlas_mask : array of int
        An array of size [X,Y,Z] defining an integer-valued image of the registered mouse brain atlas
    curr_mask : array of int
        An array of size [X,Y,Z] defining an integer-valued image of manual anatomical segmentation
    low_img_path : str
        A file path pointing to the location of the corresponding low resolution image used to generate both masks
    outdir : str
        A file path poining to the directory where the plot should be saved
    outdir_fname : str
        
        
        
    """
    # Convert .npz file at low_img_path to .nii.gz
    npz_dir, npz_fname = os.path.split(low_img_path)
    out_fname = os.path.splitext(npz_fname)[0] + '.nii.gz'
    out_path = os.path.join(outdir,out_fname)

    # If it doesn't already exist in outdir, convert the npz file to a nifti file
    if not os.path.exists(out_path):    
        key = 'I'
        resolution, xshift, yshift, zshift = 50.0, 3975, 5675, -6575 # For 50um Yongsoo atlas
        affine = np.asarray([[-resolution,0,0,xshift],[0,-resolution,0,yshift],[0,0,resolution,zshift],[0,0,0,1]])
        I = npz_to_Nifti1(low_img_path, out_path, affine, key, True)

    img = nib.load(out_path)
    img = img.get_fdata()
    vmin = np.percentile(np.unique(img[0]), 10)
    vmax = np.max(np.unique(img[0]))
    num_slices = np.shape(atlas_mask)[0]
    fig, axs = plt.subplots(num_slices,3,layout='constrained')
    
    if "overlap" in out_fname:
        atlas_vis_cp = atlas_mask.copy()
        atlas_vis_cp[atlas_vis_cp==1.0] = 0
        atlas_vis_cp[atlas_vis_cp==2.0] = 1.0
        manual_vis_cp = curr_mask.copy()
        manual_vis_cp[manual_vis_cp==1.0] = 0
        manual_vis_cp[manual_vis_cp==2.0] = 1.0
        intersection = np.logical_and(atlas_vis_cp, manual_vis_cp)
    
        all_dice_cp = 0
        all_dice_wb = 0
        
        for idx in range(num_slices):
            dice_cp, dice_wb = compute_all_dice_scores(atlas_mask[idx,:,:], curr_mask[idx,:,:])
            all_dice_cp += dice_cp
            all_dice_wb += dice_wb
            
            axs[idx,0].imshow(img[idx,:,:],vmin=vmin,vmax=vmax)
            axs[idx,0].imshow(atlas_mask[idx,:,:], alpha=0.3)
            axs[idx,0].set_title('Atlas')
            axs[idx,1].imshow(img[idx,:,:],vmin=vmin,vmax=vmax)
            axs[idx,1].imshow(curr_mask[idx,:,:], alpha=0.3)
            axs[idx,1].set_title('Manual')
            axs[idx,2].imshow(img[idx,:,:],vmin=vmin,vmax=vmax)
            axs[idx,2].imshow(intersection[idx,:,:], alpha=0.3)
            axs[idx,2].set_title(f'Intersection ({dice_cp:.2f},{dice_wb:.2f})')
    
        fig.suptitle(f'{out_fname}, ({all_dice_cp/num_slices:.2f},{all_dice_wb/num_slices:.2f})')

    # Make boundary figure
    else:
        for idx in range(num_slices):
            axs[idx,0].imshow(img[idx,:,:],vmin=vmin,vmax=vmax, label = "")
            axs[idx,0].imshow(atlas_mask[idx,:,:], alpha=0.5, cmap = 'Greens', label = "Atlas")
            axs[idx,0].set_title('Atlas')
            axs[idx,1].imshow(img[idx,:,:],vmin=vmin,vmax=vmax, label = "")
            axs[idx,1].imshow(curr_mask[idx,:,:], alpha=0.5, cmap = 'Reds', label = "Manual")
            axs[idx,1].set_title('Manual')
            axs[idx,2].imshow(img[idx,:,:],vmin=vmin,vmax=vmax, label = "")
            axs[idx,2].imshow(atlas_mask[idx,:,:], alpha=0.25, cmap = 'Greens', label = "Atlas")
            axs[idx,2].imshow(curr_mask[idx,:,:], alpha=0.25, cmap = 'Reds', label = "Manual")
            axs[idx,2].set_title('Overlap')
        
        fig.suptitle(f'{out_fname}')    
            
    fig.set_size_inches(num_slices,num_slices*2)
    fig.savefig(os.path.join(outdir, outdir_fname), dpi=500)
        