"""
Contains a set of functions used to facilitate the registration process in step1_atlas_registration.py, step2_atlas_registration.py, and step3_atlas_registration.py
"""

import torch
import os
from glob import glob
from os.path import join,split,splitext
import csv
import numpy as np

def isValidBrain(brain):
    if brain in ['TME07 1', 'TME08-1', 'TME09-1', 'TME10-1', 'TME10-3', 'hTME15-1', 'hTME15-2', 'hTME18-1', 'hTME19-2']:
        return True
    return False

def get_step1_args(brain, dtype, user = 'abenneck'):

    # Change to 'L' or 'R' depending on orientation of hemisphere-only images and 'W' if whole brain
    orientation = 'W'

    # Define device for torch computations
    device = 'cuda:0'

    # Define local paths to directory containing all 10x .npz files for 'brain'
    if brain in ['TME07-1', 'TME08-1', 'TME09-1', 'TME10-1', 'TME10-3', 'TME12-1', 'TME20-1']:
        brain_path = brain
    elif brain in ['MQC06-2', 'MQC09-3', 'MQC18-3', 'MQC82-2']: # The original HD brains
        brain_path = f'Q140_MORF/{brain}'
        orientation = 'R' 
    elif '12m' in brain: # The new HD brains
        orientation = 'L'
        if '06-2' in brain or '15-1' in brain or '18-3' in brain:
            brain_path = f'Q140_MORF/Shasha_Q140/Camk-MORF3_{brain}'
        elif '09-3' in brain:
            brain_path = f'Q140_MORF/Shasha_Q140/Camk-MORF3-Q140_{brain}'
        elif '07-5' in brain:
            brain_path = f'Q140_MORF/Shasha_Q140/Camk-MORF3-Q140_{brain}'
            orientation = 'R'
        elif '6-4' in brain:
            brain_path = f'Q140_MORF/Shasha_Q140/Camk-MORF3-Q140_12m_MQC6-4 (prev 6-3)'
            orientation = 'L'
        else:
            raise Exception('Invalid brain')
    elif 'hTME' in brain:
        orientation = 'R'
        if 'hTME15-1' in brain:
            brain_path = f'Q140_MORF_D1/Camk-MORF3-D1Tom_12m_{brain}'
        elif 'hTME18-1' in brain or 'hTME15-2' in brain:
            brain_path = f'Q140_MORF_D1/Camk-MORF3-D1Tom-Q140_12m_{brain}'
        elif 'hTME19-2' in brain:
            brain_path = f'Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME19-2_downsampled_v2'
        else:
            raise Exception('Invalid brain')
    else: # KO-Het_MORF
        orientation = 'L' 
        if 'Hpca' in brain:
            brain_path = f'KO-Het_MORF/HpcaKO-Camk-MORF3-D1tom_Hpca5-2'
        elif 'Sp9' in brain:
            brain_path = f'KO-Het_MORF/Sp9Het-Camk-MORF3-D1tom_Sp9-3-2'
        elif 'Trank' in brain:
            brain_path = f'KO-Het_MORF/Trank1KO-Camk-MORF3-D1tom_Trank1-2-3'
        elif 'Zswim' in brain:
            brain_path = f'KO-Het_MORF/Zswim6Het-Camk-MORF3-D1tom_Zswim4-1'
        else:
            raise Exception('Invalid brain')

    default_channel = 0
    if 'TME07-1' in brain:
        target_dir = f'/home/{user}/nafs/dtward/dong/dragonfly_03_2022'
    elif 'hTME19-2' in brain:
        target_dir = f'/home/{user}/panfs/dong/3D_registration/Yang_MORF_DragonFly/{brain_path}/10x/ch_{default_channel}_pipeline_building'
    else:
        target_dir = f'/home/{user}/panfs/dong/3D_registration/Yang_MORF_DragonFly/{brain_path}_downsampled/10x/ch_{default_channel}_pipeline_building'

    if brain in ['MQC06-2', 'MQC09-3', 'MQC82-3'] or 'hTME' in brain:
        target_pattern = f'*10X*ch_{default_channel}_*.npz'
    elif 'TME07-1' in brain:
        target_pattern = f'*10x*_channel_{default_channel}*.npz'
    else:
        target_pattern = f'*10x*ch_{default_channel}_*.npz'

    target_files = []
    for rootdir,directories,files in os.walk(target_dir):
        matched = glob(join(rootdir,target_pattern))
        if matched:
            target_files.extend(matched)
    target_files.sort()

    # Fnames in strange order, manually sort list here
    if brain == 'Trank1-2-3':
        temp_target_files = []
        temp_target_files.append(target_files[0])
        temp_target_files.append(target_files[1])
        temp_target_files.append(target_files[3])
        temp_target_files.append(target_files[2])
        temp_target_files.append(target_files[4])
        temp_target_files.append(target_files[5])
        temp_target_files.append(target_files[6])
        temp_target_files.append(target_files[7])
        target_files = temp_target_files
    
    outdir = f'/home/{user}/dragonfly_work/dragonfly_outputs_script/{brain}/dragonfly_output_vis'

    # load atlas images
    atlas_names = [
        f'/home/{user}/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/ara_nissl_50.vtk',
        f'/home/{user}/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/average_template_50.vtk',
        f'/home/{user}/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/annotation_50.vtk'
    ]

    # Define affine matrix, A, as an initial guess for the step1 registration
    if orientation == 'W': # TME
        A = torch.tensor([[-1.0,0.0,0.0,500.0],
                         [0.0,0.0,-1.0,0.0],
                         [0.0,1.0,0.0,0.0],
                         [0.0,0.0,0.0,1.0]],device=device,dtype=dtype) 
    elif orientation == 'R': # */Q140_MORF/*
        A = torch.tensor([[-1.0,0.0,0.0,1000.0],
                     [0.0,0.0,-1.0,200.0],
                     [0.0,1.0,0.0,-1800.0],
                     [0.0,0.0,0.0,1.0]],device=device,dtype=dtype)
    else: # */KO-Het/*
        A = torch.tensor([[-1.0,0.0,0.0,1000.0],
                     [0.0,0.0,-1.0,200.0],
                     [0.0,1.0,0.0,1800.0],
                     [0.0,0.0,0.0,1.0]],device=device,dtype=dtype)

    # Define the indeces of slices that need to be flipped
    if 'TME08-1' in brain:
        to_flip = [0,1,3,5,7]
        A[0,-1] = 675.0
    elif 'TME10-1' in brain:
        to_flip = [0,1,2,5,7,8]
        A[0,-1] = 406.0
    elif 'TME10-3' in brain:
        to_flip = [1,3,7]
        A[0,-1] = 495.0
    elif 'TME07-1' in brain:
        to_flip = [1,2,3,4,5]
        A[0,-1] = 487.0
    elif 'TME09-1' in brain:
        to_flip = [0,1]
        A[0,-1] = 408.0
    elif 'hTME15-1' in brain:
        to_flip = [0,1]
        A[0,-1] = 368.0
    elif 'hTME15-2' in brain:
        to_flip = [5]
        A[0,-1] = 413.0
    elif 'hTME18-1' in brain:
        to_flip = []
        A[0,-1] = 372.0
    elif 'hTME19-2' in brain:
        to_flip = []
        A[0,-1] = 537.0
    else:
        print(f'Invalid Brain - {brain}')

    return orientation, target_files, outdir, atlas_names, A, to_flip, device