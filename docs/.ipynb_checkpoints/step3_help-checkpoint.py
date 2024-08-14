"""
Contains a set of functions used to assist with the preprocessing and computation steps in step3_atlas_registration.py
"""

import numpy as np

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