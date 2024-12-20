import numpy as np
from time import time as t
np.set_printoptions(precision=1)

# Load the dataset
npdata, mat_file = np.load('data/data_clean.npy'), 'data/data_clean.mat'
# npdata, mat_file = np.load('data/data.npy', allow_pickle=True), 'data/data.mat'

print(npdata.dtype) # show the keys
# [('label', '<U10'), ('shot', '<i4'), ('time', '<f4'), ('data', '<f4', (92,)), ('data_err', '<f4', (92,)), ('target', '<f4', (21,)), ('emiss', '<f4', (110, 110)), ('x_emiss', '<f4', (110,)), ('y_emiss', '<f4', (110,)), ('majr', '<f4'), ('minr', '<f4'), ('b_tor', '<f4', (24,)), ('b_rad', '<f4', (24,)), ('phi_tor', '<f4', (24,))]

label = npdata['label']
shot = npdata['shot']
time = npdata['time']
data = npdata['data']
data_err = npdata['data_err']
target = npdata['target']
emiss = npdata['emiss']
x_emiss = npdata['x_emiss']
y_emiss = npdata['y_emiss']
majr = npdata['majr']
minr = npdata['minr']
b_tor = npdata['b_tor']
b_rad = npdata['b_rad']
phi_tor = npdata['phi_tor']

N = len(label)
print(f'Number of examples: {N}')
assert len(shot) == N
assert len(time) == N
assert len(data) == N
assert len(data_err) == N
assert len(target) == N
assert len(emiss) == N
assert len(x_emiss) == N
assert len(y_emiss) == N
assert len(majr) == N
assert len(minr) == N
assert len(b_tor) == N
assert len(b_rad) == N
assert len(phi_tor) == N

# sxr = npdata['data']
# sxr_horizontal = None

d = {
    'label': str(label),
    'shot': shot.reshape(-1, 1),
    'time': time.reshape(-1, 1),
    'sxr': data,
    'data_err': data_err,
    'bessel_coefss': target,
    'emiss': emiss,
    'x_emiss': x_emiss,
    'y_emiss': y_emiss,
    'majr': majr.reshape(-1, 1),
    'minr': minr.reshape(-1, 1),
    'b_tor': b_tor,
    'b_rad': b_rad,
    'phi_tor': phi_tor
}

# Save the dataset as an npz file
start_time = t()
np.savez('data/data_clean', **d)
print(f'Time to save: {t() - start_time:.2f} s')
del d, npdata, label, shot, time, data, data_err, target, emiss, x_emiss, y_emiss, majr, minr, b_tor, b_rad, phi_tor

# Load the dataset
start_time = t()
d = np.load('data/data_clean.npz')
print(f'Time to load: {t() - start_time:.2f} s')

# print the keys
print(f'Keys: {d.files}')

import matplotlib.pyplot as plt

emiss = d['emiss']

# plot 10 random examples
idxs = np.random.randint(0, N, 10)

fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(emiss[idxs[i]], cmap='inferno')
    ax.axis('off')
plt.show()
