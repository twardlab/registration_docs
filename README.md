# Dragonfly MORF Mouse Brain Registration

This repository contains the scripts and documentation for the pipeline used to register the 10x images, 30x images, and neuronal reconstructions for over 12 mouse brains to the Allen Common Coordinate Frameowrk (CCF). Please go to our [GitHub Pages](https://twardlab.github.io/registration_docs/) site associated with this repository to see the detailed documentation for all of the custom functions and arguments used within each package.

## Step 1 (Atlas to 10x Registration)
Using the `argparse` library, we created a command-line interface in order to register the Allen Mouse Brain Atlas to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step1_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.

### Step 1 Example
Here is an example list of positional and optional arguments used to register a set of images containing a section of the right hemisphere of a mouse brain imaged at 10x. This code was executed from within a cell on a Jupyter notebook.
~~~
script_path = '/home/abenneck/registration_docs/docs/step1_atlas_registration.py'

brain = 'hTME15-1'
orientation = 'R'
target_files = ['/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_01_2023-05-08_10.12.46_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_02_2023-05-08_10.31.20_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_03_2023-05-08_10.48.01_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_04_2023-05-08_11.24.52_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_05_2023-05-08_11.43.25_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_06_2023-05-08_11.57.43_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_07_2023-05-08_12.19.31_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_08_2023-05-08_12.35.58_FusionStitcher_ch_0_pow_[0.125]_down.npz', '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/Q140_MORF_D1/Camk-MORF3-D1Tom_12m_hTME15-1_downsampled/10x/ch_0_pipeline_building/hTME15-1_Camk-MORF3-D1Tom_12m_10X_09_2023-05-08_12.55.16_FusionStitcher_ch_0_pow_[0.125]_down.npz']
outdir = '/home/abenneck/dragonfly_work/dragonfly_outputs_script/hTME15-1/dragonfly_output_vis'
atlas_names = ['/home/abenneck/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/ara_nissl_50.vtk', '/home/abenneck/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/average_template_50.vtk', '/home/abenneck/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/annotation_50.vtk']
to_flip = [0,1]
device = 'cuda:0'
d_path = '/home/abenneck/Desktop/donglab_workflows'
e_path = '/home/abenneck/Desktop/emlddmm'
A = "[[-1.0,0.0,0.0,1000.0],[0.0,0.0,-1.0,200.0],[0.0,1.0,0.0,-1800.0],[0.0,0.0,0.0,1.0]]" 

cmd_str = f'python3 {script_path} {brain} {orientation} -low_paths {target_files[0]} {target_files[1]} {target_files[2]} {target_files[3]} {target_files[4]} {target_files[5]} {target_files[6]} {target_files[7]} {target_files[8]} -outdir {outdir} -atlas_paths {atlas_names[0]} {atlas_names[1]} {atlas_names[2]} -to_flip {to_flip[0]} {to_flip[1]} -device {device} -d_path {d_path} -e_path {e_path} -A {A}'

# Optional flags to include
save_fig0 = False
save_fig1 = False
save_fig2 = False
save_fig3 = False
save_fig4 = False
save_fig5 = False
save_fig6 = False
normalizeData = True
zeroMean = True
largerRGBfig = False

if save_fig0:
    cmd_str += ' -save_fig0'
if save_fig1:
    cmd_str += ' -save_fig1'
if save_fig2:
    cmd_str += ' -save_fig2'
if save_fig3:
    cmd_str += ' -save_fig3'
if save_fig4:
    cmd_str += ' -save_fig4'
if save_fig5:
    cmd_str += ' -save_fig5'
if save_fig6:
    cmd_str += ' -save_fig6'

if not normalizeData:
    cmd_str += ' -norm'
if not zeroMean:
    cmd_str += ' -zm'
if largerRGBfig:
    cmd_str += ' --largerRGBfig'

os.system(cmd_str)
~~~

## Step 2 (30x to 10x Registration)
Using the `argparse` library, we created another command-line interface in order to register our dataset of 30x mouse brain images to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step2_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.

## Step 3 (Neuron Reconstruction Registration)
Using the `argparse` library, we created a command-line interface in order to register our dataset of neuron reconstructions (defined as a set of .swc files) to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step3_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.
