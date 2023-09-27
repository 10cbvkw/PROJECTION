import pandas as pd
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import cv2 as cv
import imageio
from scipy import spatial
from tqdm import tqdm
import torch
from pytorch3d import transforms
import mrcfile

def readpdb(fi='4cmq.pdb',libfile='lib/mol.csv',iscg=False):
    pdb=open(fi,'r')
    index=[]
    name=[]
    resname=[]
    chain=[]
    resid=[]
    x=[]
    y=[]
    z=[]
    weight=[]
    element=[]
    lines=pdb.readlines()
    weightdict={}
    lib=pd.read_csv(libfile)
    if iscg:
        for i in lib.index:
            weightdict[(lib.residue[i],lib.element[i])]=lib.electron[i]
    else:
        for i in lib.index:
            weightdict[lib.element[i]]=lib.electron[i]
    for line in lines:
        if line[0:4] == 'ATOM':
            index.append(int(line[6:11]))
            n=line[12:16]
            name.append(n.strip())
            resname.append(line[17:20].strip())
            resid.append(int(line[22:26]))
            x.append(float(line[30:38]))
            y.append(float(line[38:46]))
            z.append(float(line[46:54]))
            if len(line)<77 or line[76:78].strip()=='':
                element.append(n.strip()[0])
            else:
                element.append(line[76:78].strip())
            if iscg:
                weight.append(weightdict[(resname[-1],element[-1])])
            else:
                weight.append(weightdict[element[-1]])

    data=pd.DataFrame({
        'series':index,
        'name':name,
        'resname':resname,
        'resid':resid,
        'x':x,
        'y':y,
        'z':z,
        'weight':weight,
        'element':element
    })
    return data

def make_gird_3d(shape):

    x = torch.linspace(-0.5, 0.5, shape[0])
    y = torch.linspace(-0.5, 0.5, shape[1])
    z = torch.linspace(-0.5, 0.5, shape[2])

    X, Y, Z = torch.meshgrid(x, y, z, indexing = 'ij')
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)

    grid_3d = torch.zeros(X.shape[0], 3)
    grid_3d[:, 0] = X
    grid_3d[:, 1] = Y
    grid_3d[:, 2] = Z

    return grid_3d

def make_gird_2d(shape):

    x = torch.linspace(-0.5, 0.5, shape[0])
    y = torch.linspace(-0.5, 0.5, shape[1])

    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    grid_2d = torch.zeros(X.shape[0], 2)
    grid_2d[:, 0] = X
    grid_2d[:, 1] = Y
    
    return grid_2d

def read_pdb_data(data):
    x = data.x
    y = data.y
    z = data.z
    atom_num = x.shape[0]
    atom_weights = data.weight
    atom_coords = np.zeros((atom_num, 3))
    atom_coords[:, 0] = data.x
    atom_coords[:, 1] = data.y
    atom_coords[:, 2] = data.z
    return atom_num, atom_coords, atom_weights

def create_density_map(num, coords, weights, grid_3d, voxel_size , L):

    coords = torch.tensor(coords).cuda()
    weights = torch.tensor(weights).cuda()
    voxel_size = torch.tensor(voxel_size).cuda()
    L = torch.tensor(L).cuda()

    grid_3d = grid_3d.cuda()

    grid_x = grid_3d[:, 0] * voxel_size
    grid_y = grid_3d[:, 1] * voxel_size
    grid_z = grid_3d[:, 2] * voxel_size

    atom_x = coords[:, 0]
    atom_y = coords[:, 1]
    atom_z = coords[:, 2]

    density_map = torch.zeros(grid_3d.shape[0], device='cuda')

    for i in range(num):      
        density_map += weights[i] * torch.exp(-((atom_x[i] - grid_x)**2 + (atom_y[i] - grid_y)**2 + (atom_z[i] - grid_z)**2))
    density_map = density_map.cpu().numpy()
    density_map = density_map.reshape(L, L, L)
    return density_map

def create_projection(num, coords, weights, rotated_grid_3d, voxel_size , L):

    coords = torch.tensor(coords).cuda()
    weights = torch.tensor(weights).cuda()
    voxel_size = torch.tensor(voxel_size).cuda()
    L = torch.tensor(L).cuda()

    grid_3d = rotated_grid_3d.cuda()

    grid_x = grid_3d[:, 0] * voxel_size
    grid_y = grid_3d[:, 1] * voxel_size
    grid_z = grid_3d[:, 2] * voxel_size

    atom_x = coords[:, 0]
    atom_y = coords[:, 1]
    atom_z = coords[:, 2]

    density_map = torch.zeros(grid_3d.shape[0], device='cuda')

    for i in range(num):      
        density_map += weights[i] * torch.exp(-((atom_x[i] - grid_x)**2 + (atom_y[i] - grid_y)**2 + (atom_z[i] - grid_z)**2))
    
    density_map = density_map.reshape(L, L, L)
    projection = torch.sum(density_map, dim=-1)
    projection = projection.cpu().numpy()

    return projection

def add_gaussian_noise(projection, snr):
    
    signal_power = np.mean(projection**2)
    noise_power = signal_power / snr
    noise_stddev = np.sqrt(noise_power)

    noise = np.random.normal(0, noise_stddev, projection.shape)
    noisy_projection = projection + noise

    return noisy_projection

def save_mrc_file(density_map, output_path):
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(density_map)

def compute_ctf(freqs, dfu, dfv, defocus_angle, volt, cs, w, phase_shift=0, b_factor=None):
    """
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 array of 2D spatial frequencies
        dfu (float): DefocusU (Angstrom)
        dfv (float): DefocusV (Angstrom)
        defocus_angle (float): DefocusAngle (degrees)
        volt (float): accelerating voltage (kV)
        cs (float): spherical aberration (mm)
        w (float): amplitude contrast ratio
        phase_shift (float): degrees
        b_factor (float): envelope fcn B-factor (Angstrom^2)
    """
    # convert units
    volt = volt * 1000
    cs = cs * 10 ** 7
    defocus_angle = defocus_angle * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
    x = freqs[:, 0]
    y = freqs[:, 1]
    ang = np.arctan2(y, x)
    s2 = x ** 2 + y ** 2
    df = .5 * (dfu + dfv + (dfu - dfv) * np.cos(2 * (ang - defocus_angle)))
    gamma = 2 * np.pi * (-.5 * df * lam * s2 + .25 * cs * lam ** 3 * s2 ** 2) - phase_shift
    ctf = np.sqrt(1 - w ** 2) * np.sin(gamma) - w * np.cos(gamma)
    if b_factor is not None:
        ctf *= np.exp(-b_factor / 4 * s2)
    return np.require(ctf, dtype=freqs.dtype)

def apply_ctf_2d(projection, ctf_2d):
    # Apply the 2D CTF to the projection in Fourier space
    projection_fft = np.fft.fft2(projection)
    ctf_affected_projection = projection_fft * ctf_2d

    # Take the inverse Fourier transform to get the final image
    ctf_affected_projection = np.fft.ifft2(ctf_affected_projection)

    # Return the real part (amplitude) of the result
    return np.abs(ctf_affected_projection)

L = 128
pdb = 'pdblib/4cmq.pdb'
mol = 'lib/mol.csv'
N_stack_frames = 1
SNR = 1
shape = (L, L, L)
ctf_shape = (L, L)
voxel_size = 256
volume_output_path = 'volumes/4cmq.mrc'
mrcs_shape = (N_stack_frames, L, L)
f_out_mrcs = 'projections/4cmq_'+str(L) +'_snr_'+ str(SNR) +'_n_'+ str(N_stack_frames) +'.mrcs'
translation_var = 4

image_size = L
Apix = 3
defocus_u = 2.0e+05  # Defocus_u
defocus_v = 2.0e+05  # Defocus_v 
defocus_angle = 3.67e+01  # defocus_angle 
voltage = 300.0  # Accelerating voltage in volts
Cs = 2.0  # Spherical aberration coefficient in millimeters
w = 0.1   # (float): amplitude contrast ratio
phase_shift = 0 # (float): degrees
b_factor = None# (float): envelope fcn B-factor (Angstrom^2)

add_ctf = True
add_noise = True

data = readpdb(fi=pdb,libfile=mol,iscg=False)
atom_num, atom_coords, atom_weights = read_pdb_data(data)
atom_coords[:, 0] = atom_coords[:, 0] - atom_coords[:, 0].mean()
atom_coords[:, 1] = atom_coords[:, 1] - atom_coords[:, 1].mean()
atom_coords[:, 2] = atom_coords[:, 2] - atom_coords[:, 2].mean()
rots = transforms.random_rotations(N_stack_frames)
grid_3d = make_gird_3d(shape)

# density_map = create_density_map(num = atom_num, coords = atom_coords, weights = atom_weights, grid_3d = grid_3d, voxel_size = voxel_size, L = L)
# save_mrc_file(density_map, volume_output_path)

mrcs_out = mrcfile.new_mmap(f_out_mrcs, shape=mrcs_shape, mrc_mode=2, overwrite=True)
freqs = make_gird_2d(ctf_shape).numpy()
ctf = compute_ctf(freqs = freqs, dfu = defocus_u, dfv = defocus_v, defocus_angle = defocus_angle, volt = voltage, cs = Cs, w = w, phase_shift=phase_shift, b_factor=b_factor).reshape(ctf_shape)

for i in range(N_stack_frames):
    
    rotated_grid_3d = grid_3d @ rots[i]

    rotated_grid_3d[:, 0] = rotated_grid_3d[:, 0] + np.random.normal(0, translation_var / voxel_size)
    rotated_grid_3d[:, 1] = rotated_grid_3d[:, 1] + np.random.normal(0, translation_var / voxel_size)
    rotated_grid_3d[:, 2] = rotated_grid_3d[:, 2] + np.random.normal(0, translation_var / voxel_size)

    projection = create_projection(num = atom_num, coords = atom_coords, weights = atom_weights, rotated_grid_3d = rotated_grid_3d, voxel_size = voxel_size, L = L)
    if add_ctf:
        projection = apply_ctf_2d(projection = projection, ctf_2d = ctf)
    if add_noise:
        projection = add_gaussian_noise(projection = projection, snr = SNR)
    mrcs_out.data[i] = projection

mrcs_out.flush()
mrcs_out.close()


