"""Registration of .swc neuronal representations to high resolution mouse brain images

This script is intended to be used at the command line and will save the final registration results after applying the whole-image registrations from steps 1 and 2 to a set of .swc files representing neurons. This script performs step 3 of 3 in the registration pipeline.
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sys
from glob import glob
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.collections import LineCollection
import nibabel as nib
from scipy.interpolate import interpn
from scipy.stats import mode


from step3_help import load_swc, backup_unpickle, Unpickler, ClassHack, str_to_list, get_xyzoff
    
def main():
    """
    Command line arguments
    ----------------------
    dataset : str
        The name of the dataset to be registered. This will be included in the names of the majority of the output files.
    orientation : char
        The portion of the brain captured by this dataset (W - whole, L - left hemisphere, R - right hemisphere)
    -neuron_dir : str
        The path to a directory containing all the .swc neuron representations for this dataset
    -low_to_high_dir : str
        The path to a directory containing all the low-to-high resolution transforms (.npz) computed in Step 2
    -tform : str
        The path to the low-res-to-atlas transform file (.npz) computed in Step 1
    -low_img_paths : list of str
        List of all of the file paths pointing to the location(s) of the low resolution mouse brain images (.npz) in Anterior to Posterior order, for this dataset
    -low_img_ids : list of str
        List of slice id numbers associated with each slab in '-low_img_paths'
    -high_img_paths : list of str
        List of all of the file paths pointing to the location(s) of the low resolution mouse brain images (.npz) in Anterior to Posterior order, for this dataset
    -high_img_ids : str
        List of slice id numbers associated with each high-res image; Must be of the form '[[a],...,[b,c]]' where each element of the list is a single or tuple consisting of alphanumeric strings
    -outdir : str
        The path to the directory where all outputs will be stored
    -atlas_paths : list of 4 str
        List of 3 allen atlas .vtk files, and 1 Yongsoo atlas .csv file. In order, these files are named (1) ara_nissl_50.vtk, (2) average_template_50.vtk, (3) UPenn_labels_reoriented_origin.vtk, and (4) atlas_info_KimRef_FPbasedLabel_v2.7.csv
    -toggle_seg : str
        Default - True; If True, atlas segmentations will be included in the QC output figures
    -toggle_cp : str
        Default - True; If True, atlas segmentation will highlight the subregions of the Caudoputamen
    -toggle_low : str
        Default - True; If True, QC output figures will include the low resolution image
    -toggle_high : str
        Default - True; If True, QC output figures will include the high resolution image
    -toggle_neurons : str
        Default - True; If True, QC output figures will include the registered neurons
    --zeroMean : str    
        Default - 'True'; Choices - ['True','False']; If True, use the zero mean convention when loading input data

    -e_path : str
        The location of the custom Python library 'emlddmm', which can be cloned from GitHub at https://github.com/twardlab/emlddmm
    -d_path : str
        The location of the custom Python library 'donglab_workflows', which be cloned from GitHub at https://github.com/twardlab/donglab_workflows

    Raises
    ------
    Exception
        If the number of low resolution image paths does not equal the number of low resolution image IDs
    Exception
        If high_img_ids is not provided AND the number of high resolution image paths does not equal the amount of low resolution image paths (Whole brain) or double this amount (Hemisphere brain)
    Exception
        If the number of high resolution image paths does not equal the number of high resolution image IDs
    Exception
        If the total number of singles and tuples of the high image IDs does not equal the number of low image IDs
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset', type = str, help = 'Name of dataset to be processed')
    parser.add_argument('orientation', type = str, choices = ['W','L','R'], help = 'The portion of the brain shown in the dataset (Whole, Left hemisphere, Right hemisphere)')
    parser.add_argument('-neuron_dir', type = str, required = True, help = 'The path to a directory containing all the .swc neuron representations for this dataset')
    parser.add_argument('-low_to_high_dir', type = str, required = True, help = 'The path to a directory containing all the low-to-high resolution transforms (.npz)')
    parser.add_argument('-tform', type = str, required = True, help = 'The path to the low-res-to-atlas transform file (.npz)')
    
    parser.add_argument('-low_img_paths', type = str, nargs='*', required = True, help = 'List of paths to the low-res files in Anterior to Posterior order')
    parser.add_argument('-low_img_ids', type = str, nargs = '*', required = True, help = 'List of slice id numbers associated with each slab')
    parser.add_argument('-high_img_paths', type = str, nargs='*', required = True, help = 'List of paths to the high-res files in Anterior to Posterior order')
    parser.add_argument('-high_img_ids', type = str, default = "",  help = 'List of slice id numbers associated with each high-res image; Must be of the form \'[[a],...,[b,c]]\' where each element of the list is a single or tuple consisting of alphanumeric strings')
    
    parser.add_argument('-outdir', type = str, required = True, help = 'Output directory for all files generated from this script')
    parser.add_argument('-atlas_paths', type = str, nargs = 4, required = True, help = 'List of 3 allen atlas .vtk files, and 1 Yongsoo atlas .csv file')
    
    parser.add_argument('-toggle_seg', type = str, default = "True", choices = ["True", "False"], help='Default - "True"; If True, atlas segmentations will be included in the QC output figures')
    parser.add_argument('-toggle_cp', type = str, default = "True", choices = ["True", "False"], help='Default - "True"; If True, atlas segmentation will highlight the subregions of the Caudoputamen')
    parser.add_argument('-toggle_low', type=str, default = "True", choices = ["True", "False"], help='Default - "True"; If True, QC output figures will include the low resolution image')
    parser.add_argument('-toggle_high', type=str, default = "True", choices = ["True", "False"], help='Default - "True"; If True, QC output figures will include the high resolution image')
    parser.add_argument('-toggle_neurons', type=str, default = "True", choices = ["True", "False"], help='Default - "True"; If True, QC output figures will include the registered neurons')
    parser.add_argument('-zeroMean', type=str, default = 'True', choices = ['True', 'False'], help = 'Default - True; If True, use the zero mean convention when loading input data')
    
    parser.add_argument('-e_path', type = str, required = True, help = 'The directory containing the emlddmm library from Github')
    parser.add_argument('-d_path', type = str, required = True, help = 'The directory containing the donglab_workflows library from Github')
    
    args = parser.parse_args()
    
    # Import custom libraries from local device (Repo should have already been cloned from Github)

    sys.path.append(args.d_path)
    import donglab_workflows as dw
    sys.path.append(args.e_path)
    import emlddmm
    
    brain = args.dataset
    orientation = args.orientation
    neuron_dir = args.neuron_dir
    reg_dir = args.low_to_high_dir
    transform_file = args.tform
    
    image_low_files = args.low_img_paths
    low_img_ids = args.low_img_ids
    image_high_files = args.high_img_paths
    high_img_ids = args.high_img_ids
   
    outdir = args.outdir
    atlas_paths = args.atlas_paths

    if args.toggle_seg == "True":
        includeSeg = True
    else:
        includeSeg = False

    if args.toggle_cp == "True":
        includeCP = True
    else:
        includeCP = False

    if args.toggle_low == "True":
        includeLow = True
    else:
        includeLow = False

    if args.toggle_high == "True":
        includeHigh = True
    else:
        includeHigh = False

    if args.toggle_neurons == "True":
        includeNeurons = True
    else:
        includeNeurons = False

    if args.zeroMean == 'True':
        zeroMean = True
    else:
        zeroMean = False

    # ====================================================
    # ===== (0) Perform checks on input data formats =====
    # ====================================================
    # Check that # of low_res ids equals the # of low_res files
    if len(image_low_files) != len(low_img_ids):
        raise Exception(f'Amount of low res files ({len(image_low_files)}) differs from the amount of low res ids ({len(low_img_ids)}) provided')
    
    if high_img_ids == "": # If no input is supplied, use low_img_ids to initialize high_img_ids
        if len(image_low_files) == len(image_high_files): # For brains with 1 high res img for each low res img
            high_img_ids = [[id] for id in low_img_ids]
        elif len(image_low_files)*2 == len(image_high_files): # For brains with 2 high res imgs for each low res img
            high_img_ids = [[i+'A',i+'B'] for i in low_img_ids]
        else: # If not either of the above cases, high_img_ids cannot be estimated and must be provided by the user
            raise Exception('Since no input was provided for the argument \'high_img_ids\', the number of image_high_files should be equal to the number of image_low_files or should be twice the length')        
    else: # Convert the input str into a list of 1-tuples and/or 2-tuples based on format of data
        high_img_ids = str_to_list(high_img_ids)
    
    # Count number of ids in high_img_ids
    count = 0
    for id_tuple in high_img_ids:
        for id in id_tuple:
            count += 1
    if count != len(image_high_files):
        raise Exception(f'Amount of high res files ({len(image_high_files)}) differs from the amount of high res ids ({count}) provided')
    
    # Format image_high_files, so that it can indexed using the low_ind variable during registration
    img_high_files_formatted = []
    idx = 0
    for id_tuple in high_img_ids:
        temp_list = []
        for id in id_tuple:
            temp_list.append(image_high_files[idx])
            idx += 1
        img_high_files_formatted.append(temp_list)
    
    if len(high_img_ids) != len(low_img_ids):
        raise Exception(f'Total number of 1-tuples and 2-tuples provided in \'high_img_ids\' ({len(high_img_ids)}) does not equal the number of low_img_ids ({len(low_img_ids)})')
    
    if includeCP:
        includeSeg = True
    
    # If outdir does not exits, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Separate atlas names into the variables originally used
    atlas_names = atlas_paths[:2]
    seg_name = atlas_paths[2]
    ontology_name = atlas_paths[3]
    
    # =============================
    # ===== (1) Load the data =====
    # =============================
    # The low-to-atlas transform file
    dataat = np.load(transform_file,allow_pickle=True)
    
    # The upenn mouse brain atlas segmentation
    xS,S,title,names = emlddmm.read_data(seg_name, normalize=False)
    
    # Will zero-out allen mouse brain atlas image for the hemisphere we are NOT aligning
    if orientation == 'L':
        S[:,:,int(np.shape(S)[2]/2 + 1):,:] = 0.0
    elif orientation == 'R':
        S[:,:,:int(np.shape(S)[2]/2),:] = 0.0
    else: # Whole brain: Atlas does not need to be modified 
        pass
    
    dS = np.array([x[1]-x[0] for x in xS])
    
    # downsample segmentation
    xSd,Sd = emlddmm.downsample_image_domain(xS,(S>0)*1.0,[5,5,5])
    dSd = [x[1] - x[0] for x in xSd]
    
    # create isosurface of segmentation
    verts,faces,normals,values = measure.marching_cubes((Sd[0]>0)*1.0,level=0.5)
    verts *= dSd # for 100 micron
    verts += np.array((xSd[0][0],xSd[1][0],xSd[2][0]))
    
    # Get ontology from Yongsoo Kim's mouse brain segmented atlas
    ontology = {}
    import csv
    with open(ontology_name) as f:
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
            
    # find all labels corresponding to CP
    cp_ids = []
    for id_ in ontology:    
        if 'caudoputamen' in ontology[id_]['name'].lower():
            cp_ids.append(id_)
    
    # Load the allen mouse brain atlas and zero-out missing hemisphere if necessary
    Iat = []
    for atlas_name in atlas_names:
        xIat,I_,title,names = emlddmm.read_data(atlas_name)
        I_ = I_.astype(np.float32)
        I_ /= np.mean(np.abs(I_))
    
        # Will zero-out atlas image for the hemisphere we are NOT aligning
        if orientation == 'R': # For images only containing a RIGHT  hemisphere
            I_[:,:,:int(np.shape(I_)[2]/2),:] = 0.0
        elif orientation == 'L': # For images only containing a LEFT hemisphere (+1 brings back fig[1,2])
            I_[:,:,int(np.shape(I_)[2]/2 + 1):,:] = 0.0
        else: # Whole brain: Atlas does not need to be modified 
            pass
            
        Iat.append(I_)
        
    Iat = np.concatenate(Iat)   
    Iat[0] = Iat[0]**0.25
    Iat[0] /= np.mean(np.abs(Iat[0]))
    dIat = np.array([x[1] - x[0] for x in xIat])
    XIat = np.stack(np.meshgrid(*xIat,indexing='ij'),0)
    
    # fig,ax = emlddmm.draw(Iat,xIat,vmin=0,n_slices=8)
    # fig.canvas.draw()
    
    
    # ====================================================================================
    # ===== (2) Register all neurons for this dataset + save 1 QC img for each slice =====
    # ====================================================================================
    
    # These variables will store various data points for all slabs
    neurons = []
    neuron_labels = []
    neuron_edges = []
    neuron_labels_noise = []
    images = np.zeros((3,Iat.shape[1],Iat.shape[2],Iat.shape[3]))
    set_v_to_zero = False
    files_all = [] 
    
    # Generate a list of all the .swc neuronal representations from neuron_dir
    files_ = []
    for dirpath, dirnames, filenames in os.walk(neuron_dir):
        for filename in filenames:
            if '.swc' in filename:
                files_.append(os.path.join(dirpath,filename))
    all_slice_idx = range(0,len(image_low_files))
    
    # Begin the registration
    for low_ind in all_slice_idx: 
        # Name of slab to be used for saving later registrations + in the title of the output figures
        image_low_key = low_img_ids[low_ind]
        
        # load the low res image
        image_low_file = image_low_files[low_ind]
        data = np.load(image_low_file,allow_pickle=True)
        I = data['I']
    
        # We decided to ignore voxel positions and use zero-mean convention b/c it is more numerically stable
        # (10/30/23) Defines 2 coordinate systems (xI1 from step 1, and xI2 == xI from step 2)
        if zeroMean:
        # if True
            xI = [d - np.mean(d) for d in data['xI']] # We decided to ignore voxel positions and use zero-mean convention b/c it is more numerically stable
        else: # Compute offset between high_res and low_res
            xI = [np.array(d) for d in data['xI']]
            # xI1 = [np.array(d) for d in data['xI']]
            # xI1[0] -= np.mean(xI1[0])
            # xI1[1] -= np.mean(xI1[1])
            # # xI1[2] -= np.mean(xI1[2]) # (08/07/24) Added, so that zero mean is enforceed for all 3 axes
            # xI2 = [np.array(d) for d in data['xI']]
            # I will use xI2, but keep track of the shift
            # xI = xI2
            # xIshift = [np.mean(x2-x1) for x2,x1 in zip(xI2,xI1)]
            # xshiftA = np.eye(4)
            # xshiftA[:3,-1] = xIshift
            # print(xshiftA)
    
        # check the size
        if len(xI[-1]) != I.shape[-1]:        
            xI[-1] = np.concatenate((xI[-1],xI[-1][-1][None] + (xI[-1][1]-xI[-1][0])))
        
        XI = np.stack(np.meshgrid(*xI,indexing='ij'))
        key = 'W' if 'W' in data else 'w'
        WI = data[key]
        dI = [xi[1] - xi[0] for xi in xI]
        
        # get extent and draw it
        extentlow0 = (xI[2][0]-dI[2],xI[2][-1]+dI[2],xI[1][-1]+dI[1],xI[1][0]-dI[1])
        extentlow1 = (xI[2][0]-dI[2],xI[2][-1]+dI[2],xI[0][-1]+dI[0],xI[0][0]-dI[0])
        extentlow2 = (xI[1][0]-dI[1],xI[1][-1]+dI[1],xI[0][-1]+dI[0],xI[0][0]-dI[0])
        
        I0 = np.max(I,0)
        I1 = np.max(I,1)
        I2 = np.max(I,2)
        fig,ax = plt.subplots()
        # vminI = np.min(I[WI>0.9])
        vminI = np.quantile(I.ravel(),0.4) # Made change on 6/27/23 in order to make each slice output figure show the 10x, 30x, AND neuron
        vmaxI = np.max(I[WI>0.9])
        ax.imshow(I0,extent=extentlow0,vmin=vminI,vmax=vmaxI)
        ax.set_title(os.path.splitext(os.path.split(image_low_file)[-1])[0])
    
        image_high_files = img_high_files_formatted[low_ind]
        image_high_keys = high_img_ids[low_ind]
    
        # we'll extract this slice
        files = list(filter(lambda x: '_'+image_low_key in x, files_))
        files_all.append(files)
        
        # now we start processing
        fig_together = plt.figure()
        JtoI = np.zeros_like(I)
        WJtoI = np.zeros_like(WI)
        
        # note these variables with an "all" suffix will store left and right
        phiiVall = []
        Vall = []
        Eall = []
        Rall = []
        # now we will need to loop through, the files above
        print(f'Starting low_res slice: {image_low_key}\n')
        for image_high_file,image_high_key in zip(image_high_files,image_high_keys):
    
            image_high_key = image_high_key.upper() # Some files have lower case a or b for hemi distinction
    
            # load the image    
            data = np.load(image_high_file,allow_pickle=True)
            J = data['I']
    
            # We decided to ignore voxel positions and use zero-mean convention b/c it is more numerically stable
            if zeroMean:
                xJ = [d - np.mean(d) for d in data['xI']]
            else: # Whole brain
                xJ = [np.array(d) for d in data['xI']]
            
        
            key = 'W' if 'W' in data else 'w'
            WJ = data[key]
            xJ0 = [xi - xi[0] for xi in xJ]
            dJ = [xi[1] - xi[0] for xi in xJ]
            vminJ = np.quantile(J[WJ>0.9],0.1)
            vmaxJ = np.quantile(J[WJ>0.9],0.999)
        
            
    
            extenthigh0 = (xJ[2][0]-dI[2],xJ[2][-1]+dI[2],xJ[1][-1]+dI[1],xJ[1][0]-dI[1])
            extenthigh1 = (xJ[2][0]-dI[2],xJ[2][-1]+dI[2],xJ[0][-1]+dI[0],xJ[0][0]-dI[0])
            extenthigh2 = (xJ[1][0]-dI[1],xJ[1][-1]+dI[1],xJ[0][-1]+dI[0],xJ[0][0]-dI[0])
            J0 = np.max(J,0)
            J1 = np.max(J,1)
            J2 = np.max(J,2)
            fig0,ax0 = plt.subplots()
            ax0.imshow(J0,extent=extenthigh0,vmin=vminJ,vmax=vmaxJ)
            ax0.set_title(os.path.splitext(os.path.split(image_high_file)[-1])[0])
            fig0.canvas.draw()
    
            fig1,ax1 = plt.subplots()
            ax1.imshow(J1,extent=extenthigh1,vmin=vminJ,vmax=vmaxJ)
            ax1.set_title(os.path.splitext(os.path.split(image_high_file)[-1])[0])
            fig1.canvas.draw()
    
            fig2,ax2 = plt.subplots()
            ax2.imshow(J2,extent=extenthigh2,vmin=vminJ,vmax=vmaxJ)
            ax2.set_title(os.path.splitext(os.path.split(image_high_file)[-1])[0])
            fig2.canvas.draw()

            # (08/01/24) Name of swc files differ across some datasets
            files_filter = list(filter(lambda x: f'Str_{image_high_key}' in x, files))
            if len(files_filter) == 0:
                files_filter = list(filter(lambda x: f'{brain}_{image_high_key}' in x, files))
            
            # now we load them
            V = []
            E = []
            R = []
            dJ0 = np.array([0.2,0.2,1.0])
            for file in files_filter:
                v_,e_,r_ = load_swc(file)
                
                # Compares xyz_fname to xyz_soma and returns offset if brain has already been registered and offset is known to exist
                xyzoff = get_xyzoff(file, brain=brain, neuron_dir=neuron_dir) 
                v_ += xyzoff
            
                # I think I need to scale by voxel size
                v_ *= dJ0
                
                # also account for the coordinate offset
                # a zero should correspond to an x[0]
                v_ += np.array([xJ[2][0],xJ[1][0],xJ[0][0]])
                V.append(v_)
                E.append(e_)
                R.append(r_)
    
            for i in range(len(V)):
                v_ = V[i]
                e_ = E[i]
    
                colors = np.ones(e_.shape[0])[...,None]*np.array([1,1,1,1])
                    
                colors[:,-1] = 0.5
                lines0 = LineCollection(v_[e_][...,[0,1]],colors=colors)
                lines1 = LineCollection(v_[e_][...,[0,2]],colors=colors)
                lines2 = LineCollection(v_[e_][...,[1,2]],colors=colors)
    
                ax0.add_collection(lines0)
                ax1.add_collection(lines1)
                ax2.add_collection(lines2)
    
            Vall.append(V)
            Eall.append(E)
            Rall.append(R)
    
            # now transform the volume into the low res image, we'll accumulate a weight and the image,  to do this we have to load the transformations
            # Since reg_file should have been generated by step2_atlas_registration.py, it should be safe to grab the file should be named accordingly
            reg_file = glob(os.path.join(reg_dir, '*' + image_high_key + '*.npz'))[0]
            reg_data = np.load(reg_file,allow_pickle=True)
            try:
                out = reg_data['out'].item()
            except:
                out = backup_unpickle(reg_file)
            
            tform = emlddmm.compose_sequence([emlddmm.Transform(out['v']*(1-set_v_to_zero),domain=out['xv']),emlddmm.Transform(out['A'])],XI)
            phiiJ = emlddmm.apply_transform_float(xJ,J[None],tform,padding_mode='zeros').numpy()
            phiiW = emlddmm.apply_transform_float(xJ,WJ[None],tform,padding_mode='zeros').numpy()    
            JtoI += (phiiJ[0] - vminJ)/(vmaxJ-vminJ)*phiiW[0] #not sure if I should do this for normalization    
            WJtoI += phiiW[0]
            toshow = ((JtoI)/(WJtoI))[None]
            emlddmm.draw(np.stack((toshow[0],(I-vminI)/(vmaxI-vminI),toshow[0])),xI,vmin=0,vmax=1,fig=fig_together)
            fig_together.canvas.draw()
    
            # now we have to map the neurons in, and draw them
            XJ = np.stack(np.meshgrid(*xJ,indexing='ij'))
            tform = emlddmm.compose_sequence([emlddmm.Transform(out['A'],direction='b'),emlddmm.Transform(out['v']*(1-set_v_to_zero),direction='b',domain=out['xv'])],XJ)
            phiiV = []
            for vi in V:
                # note swap xyz to zyx and back
                phiiV.append( emlddmm.interp(xJ,tform,vi.T[::-1,:,None,None].astype(np.float32))[...,0,0].T.numpy()[...,::-1] )
            phiiVall.append(phiiV) # here we add left side and right side
                   
        # now we are out of the left right high res loop, create a max projection, accounting for nans
        toshowmax = np.array(toshow[0,0])
        nanind = np.isnan(toshowmax)
        for i in range(1,toshow.shape[1]):
            toshownew = np.array(toshow[0,i])
            nanindnew = np.isnan(toshownew)
            inds = np.logical_or( toshownew > toshowmax , np.logical_not(nanindnew)*nanind  )
            toshowmax[inds] = toshownew[inds]
            nanind = np.isnan(toshowmax)
    
        # show them with neurons
        fig,ax = plt.subplots()
        ax.imshow(toshowmax,vmin=0,vmax=1,extent=extentlow0)
        idx = 0
        for j in range(len(phiiVall)):
            for i in range(len(phiiVall[j])):
                v_ = phiiVall[j][i]
                e_ = Eall[j][i]
                
                colors = np.ones(e_.shape[0])[...,None]*np.array([1,1,1,1])
                    
                colors[:,-1] = 0.5
                lines0 = LineCollection(v_[e_][...,[0,1]],colors=colors,linewidths=0.5)
                #lines1 = LineCollection(v_[e_][...,[0,2]],colors=colors,linewidths=0.5)
                #lines2 = LineCollection(v_[e_][...,[1,2]],colors=colors,linewidths=0.5)
    
                ax.add_collection(lines0)
                #ax1.add_collection(lines1)
                #ax2.add_collection(lines2)
    
                idx += 1
        fig.canvas.draw()
    
        # ===============================
        # ===== GENERATE QC FIGURES =====
        # ===============================
        
        # okay now we need to map just one slice
        # I'll just use my low res slab, and not worry about high res until later
        # so, first for the images
        # we start with XIat, we map it forward
        # we will need to do some permuting
    
        # Off-by-one issue since we skip slice 01b at 30x (Due to missing data)
        if brain == 'hTME19-2' and low_ind > 0:
            low_ind += 1
    
        # (10/30/23) Include additional shift due to misalignment b/t steps 1 and 2
        # if not zeroMean:
        # if False:
        #     tform_at_to_low = emlddmm.compose_sequence([
        #         emlddmm.Transform(dataat['v'].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xv']),
        #         emlddmm.Transform(dataat['A']),
        #         emlddmm.Transform(dataat['TJ'][low_ind]),
        #         emlddmm.Transform(xshiftA),
        #         emlddmm.Transform(dataat['vJ'][low_ind].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xvJ'][low_ind]),
        #         emlddmm.Transform(dataat['AJ'][low_ind])
        #     ],XIat)
        if True: # Whole brain
            tform_at_to_low = emlddmm.compose_sequence([
                emlddmm.Transform(dataat['v'].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xv']),
                emlddmm.Transform(dataat['A']),
                emlddmm.Transform(dataat['TJ'][low_ind]),
                emlddmm.Transform(dataat['vJ'][low_ind].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xvJ'][low_ind]),
                emlddmm.Transform(dataat['AJ'][low_ind])
            ],XIat)
    
        # v: Deformation for whole brain
        # A: Affine transform for whole brain
        # TJ: ???
        # vJ: Deformation for slice AJ[low_ind]
        # AJ: Affine transform for slice AJ[low_ind]
        
        Ishow_ = (I - vminI)/(vmaxI-vminI)
        toshow_ = np.array(toshow)
        toshow_[np.isnan(toshow_)] = 0.0
    
        outtest = emlddmm.apply_transform_float(xI,np.concatenate((toshow_,Ishow_[None])),tform_at_to_low,padding_mode='zeros') 
        
        Iatshow = np.array(Iat[0][None])
        vlim = np.quantile(Iatshow,[0.001,0.999])
        Iatshow -= vlim[0]
        Iatshow /= vlim[1]-vlim[0]
        emlddmm.draw(np.concatenate((outtest**0.5,Iatshow*0.5)),xIat,vmin=0,vmax=1,n_slices=8)
    
        # save this
        images += np.concatenate((outtest**0.5,Iatshow*0.5))
        
        # okay now we need to map just one slice
        # I'll just use my low res slab, and not worry about high res until later
        # so, first for the images
        # we start with XIat, we map it forward
        # we will need to do some permuting
        
        # did I do this in the right order? yes note that I reversed the order later
        # (10/30/23) Additional shift needed for hemi brains
        # if not zeroMean:
        # if False:
        #     tform_low_to_at = emlddmm.compose_sequence([
        #         emlddmm.Transform(dataat['v'].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xv'],direction='b'),
        #         emlddmm.Transform(dataat['A'],direction='b'),
        #         emlddmm.Transform(dataat['TJ'][low_ind],direction='b'),
        #         emlddmm.Transform(xshiftA,direction='b'),
        #         emlddmm.Transform(dataat['vJ'][low_ind].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xvJ'][low_ind],direction='b'),
        #         emlddmm.Transform(dataat['AJ'][low_ind],direction='b')
        #     ][::-1],XI)
        # else: # Whole brain
        if True:
            tform_low_to_at = emlddmm.compose_sequence([
                emlddmm.Transform(dataat['v'].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xv'],direction='b'),
                emlddmm.Transform(dataat['A'],direction='b'),
                emlddmm.Transform(dataat['TJ'][low_ind],direction='b'),
                emlddmm.Transform(dataat['vJ'][low_ind].transpose(0,-1,1,2,3)*(1-set_v_to_zero),domain=dataat['xvJ'][low_ind],direction='b'),
                emlddmm.Transform(dataat['AJ'][low_ind],direction='b')
            ][::-1],XI)
    
        # ===\/=== (10/5/23) New QC with atlas segmentation superimposed ===\/===
        # Displays 30x MIP + neurons + 10x MIP (show again with images)
        Ishow = (np.max(I,0) - vminI)/(vmaxI-vminI)
        fig,ax = plt.subplots()

        # Background of black pixels, so that any additional images can be overlayed
        ax.imshow(np.zeros_like(Ishow[...,None]**0.5)*np.array([[[1.0,0.0,0.0]]]),vmin=0,vmax=1,extent=extentlow0)

        if includeLow:
            ax.imshow(Ishow[...,None]**0.5*np.array([[[1.0,0.0,0.0]]]),vmin=0,vmax=1,extent=extentlow0)
        if includeHigh:
            ax.imshow(toshowmax[...,None]**0.5*np.array([[[0.0,1.0,0.0]]]),vmin=0,vmax=1,alpha=0.4,cmap='Greens',extent=extentlow0)
    
        if includeNeurons:
            # If only neurons are to be visualized, set the background to black
            idx = 0
            for j in range(len(phiiVall)):
                for i in range(len(phiiVall[j])):
                    v_ = phiiVall[j][i]
                    e_ = Eall[j][i]
                    
                    colors = np.ones(e_.shape[0])[...,None]*np.array([1,1,1,1])
                    colors[:,-1] = 0.5
        
                    lines0 = LineCollection(v_[e_][...,[0,1]],colors=colors,linewidths=0.5)
                    ax.add_collection(lines0)
        
                    idx += 1
    
        # Intialize a mode projection, which will be used to define regional segmentations
        Stformed = interpn(xS,S[0],tform_low_to_at.permute(1,2,3,0),bounds_error=False,fill_value=0,method='nearest')        
        m,c = mode(Stformed,axis=0,keepdims=True)
        m = m[0]        # get segmentation labels

        # If True, include the regional segmentations for all regions in this coronal slice
        if includeSeg:            
            # from m we'll get boundaries
            mb = 1 - (m == np.roll(m,shift=1,axis=0))*(m == np.roll(m,shift=1,axis=1))*(m == np.roll(m,shift=-1,axis=0))*(m == np.roll(m,shift=-1,axis=1))
            mb = mb*1.0
            # and add it to the image
            ax.imshow(np.stack((mb,mb,mb,mb*0.125),-1),interpolation='none',extent=extentlow0,cmap='gray')
            
        # If True, emphasize the regional segmentation for the CP in this coronal slice
        if includeCP:
            mcp = np.zeros_like(m)
            for id_ in cp_ids:
                mcp = np.logical_or(mcp,m==id_)
            mcpb = 1 - (mcp == np.roll(mcp,shift=1,axis=0))*(mcp == np.roll(mcp,shift=1,axis=1))*(mcp == np.roll(mcp,shift=-1,axis=0))*(mcp == np.roll(mcp,shift=-1,axis=1))
            mcpb = mcpb*1.0
            ax.imshow(np.stack((mcpb,mcpb,mcpb,mcpb*0.5),-1),interpolation='none',extent=extentlow0,cmap='gray')
    
        ax.set_title(f'{brain}_{image_low_key}_seg')
        fig.savefig(os.path.join(outdir,f'{brain}_{image_low_key}_seg.jpg'), dpi = 500.0)
    
        # ===/\=== (10/05/23) New QC with atlas segmentation superimposed ===/\===
    
        # now we want to interpolate our neuron points here
        phiiphiiV = []
        for phiiV in phiiVall: # note phiiVall stores left and right for this slab
            for vi in phiiV:
                # note swap xyz to zyx and back
                phiiphiiV.append( emlddmm.interp(xI,tform_low_to_at,vi.T[::-1,:,None,None].astype(np.float32))[...,0,0].T.numpy()[...,::-1] )
    
        if Eall:
            Eall_ = []
            for E_hemi in Eall:
                for val in E_hemi:
                    Eall_.append(val)
        else:
            Eall_ = np.zeros((0,2),dtype=int)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for vi,ei in zip(phiiphiiV,Eall_):
            lc = Line3DCollection(vi[:,::-1][ei],linewidths=0.5) # flip xyz to zyx
            ax.add_collection3d(lc)
        ax.set_xlim(xIat[0][0],xIat[0][-1])
        ax.set_ylim(xIat[1][0],xIat[1][-1])
        ax.set_zlim(xIat[2][0],xIat[2][-1])
        
        # then we want to show them with segmentations
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        pc = Poly3DCollection(verts[faces],alpha=0.1)
        ax.add_collection3d(pc)
    
        for vi,ei in zip(phiiphiiV,Eall_):
            lc = Line3DCollection(vi[:,::-1][ei],linewidths=0.1) # flip xyz to zyx
            ax.add_collection3d(lc)
    
        ax.set_xlim(xIat[0][0],xIat[0][-1])
        ax.set_ylim(xIat[1][0],xIat[1][-1])
        ax.set_zlim(xIat[2][0],xIat[2][-1])
    
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        c = np.array([np.mean(xlim),np.mean(ylim),np.mean(zlim)])
        d = np.array([xlim[1]-c[0], ylim[1]-c[1],zlim[1]-c[2]])
        maxd = np.max(d)
        ax.set_xlim(c[0]-maxd,c[0]+maxd)
        ax.set_ylim(c[1]-maxd,c[1]+maxd)
        ax.set_zlim(c[2]-maxd,c[2]+maxd)
    
        # sample the structures
        VS = []
        for vi in phiiphiiV:
            tmp = emlddmm.interp([x.astype(np.float64) for x in xS],
                           S.astype(np.float64),
                           vi.T[::-1,:,None,None].astype(np.float64),
                           mode='nearest')[...,0,0].T.numpy()[...,::-1] 
            VS.append(tmp)
        
        # NEW in v08
        # add some fuzz
        # since registration was at 50 micron, I'll add gaussian noise with 50 micron std
        VS_noise = []
        Sf = S.astype(np.float64)
        n_noise = 500
        std_noise = 50.0
        for n_noise in range(n_noise):
            noise = np.random.randn(3)*std_noise
            VS_noise_i = []
            for vi in phiiphiiV:
                vi = np.array(vi)  # COPY! important
                vi += noise
    
                tmp = emlddmm.interp([x.astype(np.float64) for x in xS],
                               Sf,
                               vi.T[::-1,:,None,None].astype(np.float64),
                               mode='nearest')[...,0,0].T.numpy()[...,::-1]     
                VS_noise_i.append(tmp)
            VS_noise.append(VS_noise_i)
        
        neurons.append(phiiphiiV)
        neuron_labels.append(VS)
        neuron_labels_noise.append(VS_noise)
        neuron_edges.append(Eall_)

        plt.close('all')
        
    
    # =========================================================
    # ===== (3) Generate + save all registered .swc files =====
    # =========================================================
    images[-1] = Iat[0]/np.quantile(Iat[0],0.99)
    emlddmm.draw(images,xIat,vmin=0,vmax=1,n_slices=8)
    
    affine = np.diag([dIat[0],dIat[1],dIat[2],1.0])
    affine[:3,-1] = np.array([xIat[0][0],xIat[1][0],xIat[2][0]])
    vol = nib.Nifti1Image(images[0],affine=affine)
    nib.save(vol,os.path.join(outdir,'atlas_high.nii'))
    vol = nib.Nifti1Image(images[1],affine=affine)
    nib.save(vol,os.path.join(outdir,'atlas_low.nii'))
    vol = nib.Nifti1Image(images[2],affine=affine)
    nib.save(vol,os.path.join(outdir,'atlas_nissl.nii'))
    
    
    # ========== Misc figure 1 ==========
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pc = Poly3DCollection(verts[faces],alpha=0.1)
    ax.add_collection3d(pc)
    
    for Vi,Ei in zip(neurons,neuron_edges):
        for vi,ei in zip(Vi,Ei):
            lc = Line3DCollection(vi[:,::-1][ei],linewidths=0.1,color=np.random.rand(3)) # flip xyz to zyx
            ax.add_collection3d(lc)
    
    
    ax.set_xlim(xIat[0][0],xIat[0][-1])
    ax.set_ylim(xIat[1][0],xIat[1][-1])
    ax.set_zlim(xIat[2][0],xIat[2][-1])
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    c = np.array([np.mean(xlim),np.mean(ylim),np.mean(zlim)])
    d = np.array([xlim[1]-c[0], ylim[1]-c[1],zlim[1]-c[2]])
    maxd = np.max(d)
    ax.set_xlim(c[0]-maxd,c[0]+maxd)
    ax.set_ylim(c[1]-maxd,c[1]+maxd)
    ax.set_zlim(c[2]-maxd,c[2]+maxd)
    
    # here it is colored by soma label
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pc = Poly3DCollection(verts[faces],alpha=0.1)
    ax.add_collection3d(pc)
    colors = {}
    for Vi,Ei,Li in zip(neurons,neuron_edges,neuron_labels):    
        for vi,ei,li in zip(Vi,Ei,Li):
            labels = np.unique(li)
            for lii in li:
                if lii.item() not in colors:
                    colors[lii.item()] = np.random.rand(3)
            c = colors[li[0].item()] # use the cell body
            lc = Line3DCollection(vi[:,::-1][ei],linewidths=0.1,colors=c) # flip xyz to zyx
            ax.add_collection3d(lc)
    
    
    ax.set_xlim(xIat[0][0],xIat[0][-1])
    ax.set_ylim(xIat[1][0],xIat[1][-1])
    ax.set_zlim(xIat[2][0],xIat[2][-1])
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    c = np.array([np.mean(xlim),np.mean(ylim),np.mean(zlim)])
    d = np.array([xlim[1]-c[0], ylim[1]-c[1],zlim[1]-c[2]])
    maxd = np.max(d)
    ax.set_xlim(c[0]-maxd,c[0]+maxd)
    ax.set_ylim(c[1]-maxd,c[1]+maxd)
    ax.set_zlim(c[2]-maxd,c[2]+maxd)
    
    x0 = np.array([x[0] for x in xS])
    x1 = np.array([x[-1] for x in xS])
    
    verts_ = verts - x0
    
    # ========== Misc figure 2 ==========
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.view_init(90,90) # yes they look totally flat! somewhere they are getting squished!
    pc = Poly3DCollection(verts_[faces],alpha=0.1)
    ax.add_collection3d(pc)
    colors = {}
    for Vi,Ei,Li in zip(neurons,neuron_edges,neuron_labels):    
        for vi,ei,li in zip(Vi,Ei,Li):
            vi_ = vi - x0[::-1]
            labels = np.unique(li)
            for lii in li:
                if lii.item() not in colors:
                    colors[lii.item()] = np.random.rand(3)
            c = colors[li[0].item()] # use the cell body
            lc = Line3DCollection(vi_[:,::-1][ei],linewidths=0.1,colors=c) # flip xyz to zyx
            ax.add_collection3d(lc)
    
    ax.set_xlim(np.min(verts_[:,0],0),np.max(verts_[:,0],0))
    ax.set_ylim(np.min(verts_[:,1],0),np.max(verts_[:,1],0))
    ax.set_zlim(np.min(verts_[:,2],0),np.max(verts_[:,2],0))
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    c = np.array([np.mean(xlim),np.mean(ylim),np.mean(zlim)])
    d = np.array([xlim[1]-c[0], ylim[1]-c[1],zlim[1]-c[2]])
    maxd = np.max(d)
    ax.set_xlim(c[0]-maxd,c[0]+maxd)
    ax.set_ylim(c[1]-maxd,c[1]+maxd)
    ax.set_zlim(c[2]-maxd,c[2]+maxd)
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('x2')
    
    # here I deal with flips
    x0 = np.array([x[0] for x in xS])
    x1 = np.array([x[-1] for x in xS])
    
    verts_ = verts - x0
    verts_[:,0] = (x1[0]-x0[0]) - verts_[:,0]
    verts_[:,1] = (x1[1]-x0[1]) - verts_[:,1]
    
    # ========== Misc figure 3 ==========
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pc = Poly3DCollection(verts_[faces],alpha=0.1)
    ax.add_collection3d(pc)
    colors = {}
    for Vi,Ei,Li in zip(neurons,neuron_edges,neuron_labels):    
        for vi,ei,li in zip(Vi,Ei,Li):
            # offset (note xyz zyx)
            vi_ = vi - x0[::-1]
            # flip the first and last (note xyz,zyx)
            vi_[:,2] = (x1[0]-x0[0]) - vi_[:,2]
            vi_[:,1] = (x1[1]-x0[1]) - vi_[:,1]
        
            labels = np.unique(li)
            for lii in li:
                if lii.item() not in colors:
                    colors[lii.item()] = np.random.rand(3)
            c = colors[li[0].item()] # use the cell body
            lc = Line3DCollection(vi_[:,::-1][ei],linewidths=0.1,colors=c) # flip xyz to zyx
            ax.add_collection3d(lc)
    
    ax.set_xlim(np.min(verts_[:,0],0),np.max(verts_[:,0],0))
    ax.set_ylim(np.min(verts_[:,1],0),np.max(verts_[:,1],0))
    ax.set_zlim(np.min(verts_[:,2],0),np.max(verts_[:,2],0))
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    c = np.array([np.mean(xlim),np.mean(ylim),np.mean(zlim)])
    d = np.array([xlim[1]-c[0], ylim[1]-c[1],zlim[1]-c[2]])
    maxd = np.max(d)
    ax.set_xlim(c[0]-maxd,c[0]+maxd)
    ax.set_ylim(c[1]-maxd,c[1]+maxd)
    ax.set_zlim(c[2]-maxd,c[2]+maxd)
    ax.set_xlabel('x0=z')
    ax.set_ylabel('x1=y')
    ax.set_zlabel('x2=x')
    
    # here I deal with flips
    
    x0 = np.array([x[0] for x in xS])
    x1 = np.array([x[-1] for x in xS])
    
    verts_ = verts - x0
    verts_[:,0] = (x1[0]-x0[0]) - verts_[:,0]
    verts_[:,1] = (x1[1]-x0[1]) - verts_[:,1]
    
    # now permute
    permutation = [2,0,1] # no
    permutation = [1,2,0] # yes! note this permutation is cyclic, so doesn't change handedness
    verts_ = verts_[:,permutation]
    
    # ========== Misc figure 4 ==========
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=30,azim=45,vertical_axis='x')
    pc = Poly3DCollection(verts_[faces],alpha=0.1)
    ax.add_collection3d(pc)
    colors = {}
    for Vi,Ei,Li in zip(neurons,neuron_edges,neuron_labels):    
        for vi,ei,li in zip(Vi,Ei,Li):
            # offset (note xyz zyx)
            vi_ = vi - x0[::-1]
            # flip the first and last (note xyz,zyx)
            vi_[:,2] = (x1[0]-x0[0]) - vi_[:,2]
            vi_[:,1] = (x1[1]-x0[1]) - vi_[:,1]
            
            # use the same permutation as above
            vi_ = vi_[:,::-1][:,permutation][:,::-1]
        
            labels = np.unique(li)
            for lii in li:
                if lii.item() not in colors:
                    colors[lii.item()] = np.random.rand(3)
            c = colors[li[0].item()] # use the cell body
            lc = Line3DCollection(vi_[:,::-1][ei],linewidths=0.1,colors=c) # flip xyz to zyx
            ax.add_collection3d(lc)
    
    ax.set_xlim(np.min(verts_[:,0],0),np.max(verts_[:,0],0))
    ax.set_ylim(np.min(verts_[:,1],0),np.max(verts_[:,1],0))
    ax.set_zlim(np.min(verts_[:,2],0),np.max(verts_[:,2],0))
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    c = np.array([np.mean(xlim),np.mean(ylim),np.mean(zlim)])
    d = np.array([xlim[1]-c[0], ylim[1]-c[1],zlim[1]-c[2]])
    maxd = np.max(d)
    ax.set_xlim(c[0]-maxd,c[0]+maxd)
    ax.set_ylim(c[1]-maxd,c[1]+maxd)
    ax.set_zlim(c[2]-maxd,c[2]+maxd)
    ax.set_xlabel('x0=z')
    ax.set_ylabel('x1=y')
    ax.set_zlabel('x2=x')
    
    ax.invert_xaxis() # this is important to make z look like it is pointing right
    fig.savefig(os.path.join(outdir,'neuron_ccf_space.png'))
    
    # I need to write out all the swc files
    count1 = 0
    count = 0
    with open(os.path.join(outdir,'soma_label_v08.csv'),'wt') as f_soma:
        f_soma.write('name, id\n')
        for Vi,Ei,Li,Filesi in zip(neurons,neuron_edges,neuron_labels,files_all):
            count2 = 0
            for vi,ei,li,filei in zip(Vi,Ei,Li,Filesi):        
                filename_end = os.path.split(filei)[-1]
                
                # I need to write out swc from vertices and edges
                outname = os.path.join(outdir,filename_end.replace('.swc','_mapped.swc'))
                
                f_soma.write(filename_end+','+str(int(li[0].item()))+'\n')
                with open(outname,'wt') as f:
                    #f.write(f'# Generated by Daniel Tward\n')
                    for i in range(len(vi)):
                        # TODO
                        # we need to update the radius
                        radius = 1.0
    
                        parent = ei[np.where(ei[:,1] == i)[0],0]
    
                        if parent.size == 0:
                            parent = -1
                        else: 
                            parent = parent[0] + 1
                        # we print the vertex number, starting at 1
                        # then we print the structure label
                        # then we print the components in xyz order (we reverse the zyx)
                        # then we print the radius, which is just set to 1 
                        # then we print the parent
                        f.write(f'{i+1},{int(li[i].item())},{vi[i,2]},{vi[i,1]},{vi[i,0]},{radius},{parent}\n')
    
                count += 1
                count2 += 1
            count1 += 1
    
    # this cell is a copy of the above, but with the transform
    count1 = 0
    count = 0
    with open(os.path.join(outdir,'soma_label_hard.csv'),'wt') as f_soma:
        f_soma.write('name, id\n')
        for Vi,Ei,Li,Filesi in zip(neurons,neuron_edges,neuron_labels,files_all):
            count2 = 0
            for vi,ei,li,filei in zip(Vi,Ei,Li,Filesi):     
                vi_ = vi - x0[::-1]
                # flip the first and last (note xyz,zyx)
                vi_[:,2] = (x1[0]-x0[0]) - vi_[:,2]
                vi_[:,1] = (x1[1]-x0[1]) - vi_[:,1]
    
                # use the same permutation as above
                vi_ = vi_[:,::-1][:,permutation][:,::-1]
    
    
                filename_end = os.path.split(filei)[-1]
                
                # I need to write out swc from vertices and edges
                outname = os.path.join(outdir,f'image_{count1:06d}_neuron_{count2:06d}.swc')
                outname = os.path.join(outdir,filename_end.replace('.swc','_mapped_ccf_permuted.swc'))
                
                f_soma.write(filename_end+','+str(int(li[0].item()))+'\n')
                with open(outname,'wt') as f:
                    #f.write(f'# Generated by Daniel Tward\n')
                    for i in range(len(vi_)):
                        # TODO
                        # we need to update the radius
                        radius = 1.0
    
                        parent = ei[np.where(ei[:,1] == i)[0],0]
    
                        if parent.size == 0:
                            parent = -1
                        else: 
                            parent = parent[0] + 1
                        # we print the vertex number, starting at 1
                        # then we print the structure label
                        # then we print the components in xyz order (we reverse the zyx)
                        # then we print the radius, which is just set to 1 
                        # then we print the parent
                        f.write(f'{i+1},{int(li[i].item())},{vi_[i,2]},{vi_[i,1]},{vi_[i,0]},{radius},{parent}\n')
    
                count += 1
                count2 += 1
            count1 += 1
    
    labels = np.unique(S)
    with open(os.path.join(outdir,'soma_label_soft_prob.csv'),'wt') as f_soma:
        # write out the label headers
        f_soma.write('name, ', )
        for l in labels:
            f_soma.write(f'{l}, ')
        f_soma.write('\n')
        
        for Li,Filesi in zip(neuron_labels_noise,files_all):        
            # note Li is length 50 and consists of 50 samples
            # I transpose the first to dims below
            for li,filei in zip(zip(*Li),Filesi):        
                filename_end = os.path.split(filei)[-1]
                
                # get the soma (first entry)
                neuron_labels = [li[i][0] for i in range(len(li))]
                neuron_prob = np.zeros(len(labels))
                for l in neuron_labels:
                    neuron_prob += labels==l
                neuron_prob /= np.sum(neuron_prob)
                
                
                f_soma.write(filename_end+', '+ ', '.join([str(p) for p in neuron_prob]) +'\n')
    
if __name__ == '__main__':
    main()