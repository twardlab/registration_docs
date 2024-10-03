# Dragonfly MORF Mouse Brain Registration

This repository contains the scripts and documentation for the pipeline used to register the 10x images, 30x images, and neuronal reconstructions for over 12 mouse brains to the Allen Common Coordinate Framework (CCF). Please go to our [GitHub Pages](https://twardlab.github.io/registration_docs/) site associated with this repository to see the detailed documentation for all of the custom functions and arguments used within each package.

Below is an overview of the 3 primary steps in our registration pipeline, along with examples of how to use the command-line interface associated with each script.

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

### Step 1 Example Outputs from running the above cell
| Before Registration | After Registration |
| ------------------- | ------------------ |
| ![image](/images/hTME15-1_rgb_it_00000.jpg) |![image](/images/hTME15-1_rgb_it_39761.jpg)|

## Step 2 (30x to 10x Registration)
Using the `argparse` library, we created another command-line interface in order to register our dataset of 30x mouse brain images to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step2_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.

### Step 2 Example
Here is an example list of positional and optional arguments used to register a single 30x image to the corresponding 10x image. This code was executed from within a cell on a Jupyter notebook.
~~~
script_path = '/home/abenneck/registration_docs/docs/step2_atlas_registration.py'

dataset = 'TME08-1'
low_path = '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_1_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_04_ch_1_pow_[0.125]_down.npz'
high_path = '/home/abenneck/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_04A_ch_0_pow_[0.125]_down.npz'
high_num = '04A'
outdir = '/home/abenneck/dragonfly_work/dragonfly_outputs_script/TME08-1/step2_out'
shifts = [0,800,2900]
d_path = '/home/abenneck/Desktop/donglab_workflows'
e_path = '/home/abenneck/Desktop/emlddmm'
gamma = False
checkInit = False
zeroMean = True
useRigidTransform = False

# Example where all input arguments are provided
cmd_str = f'python3 {script_path} {dataset} -low_path {low_path} -high_path {high_path} -high_num {high_num} -outdir {outdir} -shifts {shifts[0]} {shifts[1]} {shifts[2]} -e_path {e_path} -d_path {d_path}'

if gamma:
    cmd_str += ' -gamma'

if checkInit:
    cmd_str += ' -checkInit'

if not zeroMean:
    cmd_str += ' -zeromean'

if useRigidTransform:
    cmd_str += ' -useRigidTransform'

os.system(cmd_str)
~~~

### Step 2 Example Outputs from running the above cell
| Before Registration | After Registration |
| ------------------- | ------------------ |
| ![image](/images/TME08-1_04A_err3_before.jpg) |![image](/images/TME08-1_04A_err3_after.jpg)|

## Step 3 (Neuron Reconstruction Registration)
Using the `argparse` library, we created a command-line interface in order to register our dataset of neuron reconstructions (defined as a set of .swc files) to our dataset of 30x mouse brain images. This program can be used by running:

`python3 step3_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.

### Step 3 Example
Here is an example list of positional and optional arguments used to register a set of neuron reconstructions (.swc) associated with a whole mouse brain. Note that `low_to_high_dir` contains the registrations for every 30x slice, which were generated by running the Step 2 script on every 30x image and that `tform` is the file containing all of the registration information generated for this dataset at Step 1. This code was executed from within a cell on a Jupyter notebook.

~~~
# Example where all required arguments, aside from the high_img_ids argument are provided
# The high_img_ids argument is generated within the script

script_path = '/home/abenneck/registration_docs/docs/step3_atlas_registration.py'

# Define all input variables
dataset = 'TME08-1'
orientation = 'W'
neuron_dir = '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_reconstructions'
low_to_high_dir = '/home/abenneck/mounts/bmaproot/nafs/dtward/andrew_work/test/dragonfly_work/dragonfly_outputs/TME08-1/dragonfly_10_to_30_outputs'
tform = '/home/abenneck/mounts/bmaproot/nafs/dtward/andrew_work/test/dragonfly_work/dragonfly_outputs_weighted/TME08-1/dragonfly_output_vis/transformation_outputs.npz'

low_img_paths = ['/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_01_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_02_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_03_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_04_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_05_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_06_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_07_ch_0_pow_[0.125]_down.npz',
                 '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/10x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_10x_08_ch_0_pow_[0.125]_down.npz'
                ]
low_imd_ids = ['01','02','03','04','05','06','07','08']

high_img_paths = ['/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_01A_ch_0_pow_[0.125]_down.npz', 
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_01B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_02A_ch_0_pow_[0.125]_down.npz', 
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_02B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_03A_ch_0_pow_[0.125]_down.npz', 
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_03B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_04A_ch_0_pow_[0.125]_down.npz', 
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_04B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_05A_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_05B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_06A_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_06B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_07A_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_07B_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_08A_ch_0_pow_[0.125]_down.npz',
                  '/home/abenneck/mounts/bmaproot/panfs/dong/3D_registration/Yang_MORF_DragonFly/TME08-1_downsampled/30x/ch_0_pipeline_building/Camk2a-MORF3-D1Tom_TME08-1_30x_Str_08B_ch_0_pow_[0.125]_down.npz'
                 ]
# high_img_ids = "[['01A','01B'],['02A','02B'],['03A','03B'],['04A','04B'],['05A','05B'],['06A','06B'],['07A','07B'],['08A','08B']]"

outdir = '/home/abenneck/dragonfly_work/dragonfly_outputs_script/TME08-1/step3_out'
atlas_paths = ['/home/abenneck/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/ara_nissl_50.vtk', '/home/abenneck/mounts/bmaproot/nafs/dtward/allen_vtk/allen_vtk/average_template_50.vtk','/home/abenneck/mounts/bmaproot/nafs/dtward/dong/upenn_atlas/UPenn_labels_reoriented_origin.vtk', '/home/abenneck/mounts/bmaproot/nafs/dtward/dong/upenn_atlas/atlas_info_KimRef_FPbasedLabel_v2.7.csv']
d_path = '/home/abenneck/Desktop/donglab_workflows'
e_path = '/home/abenneck/Desktop/emlddmm'

cmd_str = f'python3 {script_path} {dataset} {orientation} -neuron_dir {neuron_dir} -low_to_high_dir {low_to_high_dir} -tform {tform} -low_img_paths {low_img_paths[0]} {low_img_paths[1]} {low_img_paths[2]} {low_img_paths[3]} {low_img_paths[4]} {low_img_paths[5]} {low_img_paths[6]} {low_img_paths[7]} -low_img_ids {low_imd_ids[0]} {low_imd_ids[1]} {low_imd_ids[2]} {low_imd_ids[3]} {low_imd_ids[4]} {low_imd_ids[5]} {low_imd_ids[6]} {low_imd_ids[7]} -high_img_paths {high_img_paths[0]} {high_img_paths[1]} {high_img_paths[2]} {high_img_paths[3]} {high_img_paths[4]} {high_img_paths[5]} {high_img_paths[6]} {high_img_paths[7]} {high_img_paths[8]} {high_img_paths[9]} {high_img_paths[10]} {high_img_paths[11]} {high_img_paths[12]} {high_img_paths[13]} {high_img_paths[14]} {high_img_paths[15]} -outdir {outdir} -atlas_paths {atlas_paths[0]} {atlas_paths[1]} {atlas_paths[2]} {atlas_paths[3]} -e_path {e_path} -d_path {d_path}'

toggle_seg = True
toggle_cp = True
toggle_low = True
toggle_high = True
toggle_neurons = True

if not toggle_seg:
    cmd_str += ' -toggle_seg'
if not toggle_cp:
    cmd_str += ' -toggle_cp'
if not toggle_low:
    cmd_str += ' -toggle_low'
if not toggle_high:
    cmd_str += ' -toggle_high'

if not toggle_neurons:
    cmd_str += ' -toggle_neurons'

os.system(cmd_str)
~~~

### Step 3 Example Output from running the above cell (Note that similar images would also be generated for each of the 10x slabs)
This output includes the registered 10x image, the 2 corresponding 30x images, all of the registered neuronal reconstructions for this slab, regional segmentations, and a highlighted segmentation surrounding the Caudateputamen (CP). The CP was the primary region of interest for the flagship paper where these scripts were used to register 12 mouse brains to the Allen Mouse Brain Atlas. (TODO: Add link once one is available)
![image](/images/TME08-1_04_seg.jpg)

Note that all other Python modules within the `source` folder do not have command-line interfaces and are instead used to define functions which are called in the 3 primary registration scripts or for generating the documentation found on this repository's [GitHub Pages](https://twardlab.github.io/registration_docs/) site
- This includes `step1_help.py` and `step3_help.py`

# registration_script_example.ipynb
This notebook is inteded to be used to show that the scripts within this repository function as intended.

# compute_neuron_probabilities.ipynb
This notebook was used to assign conditional regional probabilities to all of the registered neurons in the Dendritome MORF project.
