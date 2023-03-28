import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

LNG_DIR = Path('/homedata/nmpnguyen/LNG/NETCDF/')
LNG_LISTFILES = sorted(LNG_DIR.glob('LNG_HSRL_RAW_L1_TEMP_CADDIWA_*.nc'))

key532 = ['raw_HSR_Signal_532', 'Model_Molecular_Backscatter_532', 'Model_Molecular_Extinction_532','Model_Molecular_Transmittance_532', 
        'LNG_Total_Attenuated_Backscatter_532']

key355 = ['raw_HSR_Signal_355', 
        'Model_Molecular_Backscatter_355', 
        'Model_Molecular_Extinction_355',
        'Model_Molecular_Transmittance_355', 
        'LNG_Parallel_Attenuated_Backscatter_355',
        'LNG_Perpendicular_Attenuated_Backscatter_355']
# fig, axs = plt.subplots(figsize=(16,10), nrows=3, ncols=3)
# for filepath, (i,ax) in zip(LNG_LISTFILES, enumerate(axs.flat)):
allsr355 = None
allsr532 = None

for filepath in LNG_LISTFILES:
    data = xr.open_dataset(filepath)
    valid_rate = (data['Validity_rate'] == 1) 
    height, time, Pointing = data['Height'].isel(time=valid_rate).values , data['Time'].isel(time=valid_rate).values, data['LNG_UpDown'].isel(time=valid_rate).values
    height2, time2 = height[np.unique(np.where(Pointing==1)[0]),:], time[np.unique(np.where(Pointing==1)[0])], 
    maskHSR532 = data['Mask_532'].isel(time=valid_rate).values
    # SR 532
    mol = data[key532[1]].isel(time=valid_rate).values * data[key532[3]].isel(time=valid_rate).values
    atb = data[key532[-1]].isel(time=valid_rate).values
    sr = atb/mol
    sr532 = sr[np.unique(np.where((Pointing==1)&(maskHSR532==1))[0]),:] #np.ma.masked_where(Pointing!=1, sr)
    print(sr532.shape)
#     idx = random.randint(0, time2.shape[0])
#     ax.semilogx(sr2[idx,:], height2[idx,:], color='g', label='532')
    # SR 355
    mol = data[key355[1]].isel(time=valid_rate).values * data[key355[3]].isel(time=valid_rate).values
    atb = data[key355[-2]].isel(time=valid_rate).values + data[key355[-1]].isel(time=valid_rate).values
    sr = atb/mol
    sr355 = sr[np.unique(np.where((Pointing==1)&(maskHSR532==1))[0]),:] #np.ma.masked_where(Pointing!=1, sr)
    print(sr355.shape)
#     ax.semilogx(sr2[idx,:], height2[idx,:], color='b', label='355')    
#     ax.axvline(1, color="black", linestyle="--") 
#     ax.set_ylim(-1, 18)
#     ax.set(xlabel='SR', ylabel='Alt [km]', title=f'{filepath.stem.split("_")[6]}, {time2[idx]}')
#     pcm = ax.pcolormesh(time, height.T, sr2.T, vmin=0, vmax=20, cmap='turbo')
#     plt.colorbar(pcm, ax=ax, label='SR 532')
#     ax.set_ylim(-1,18)
#     ax.set(xlabel='SR', ylabel='Alt [km]', title=f'{filepath.stem.split("_")[6]}')
    if (allsr532 is None )| (allsr355 is None):
        allsr355 = sr355.flatten()
        allsr532 = sr532.flatten()
    else:
        allsr355 = np.concatenate((allsr355, sr355.flatten()))
        allsr532 = np.concatenate((allsr532, sr532.flatten()))


def get_params_histogram(srlimite, Xdata, Ydata):
    def remove_NaN_Inf_values(arrayX, arrayY):
        idsX = np.where(~np.isnan(arrayX)&~np.isinf(arrayX))[0]
        idsY = np.where(~np.isnan(arrayY)&~np.isinf(arrayY))[0]
        print(idsX, idsY)
        mask = np.intersect1d(idsX, idsY)
        return mask
    
    from scipy import stats
    from scipy.optimize import curve_fit
    # objective function for best fit
    def objective(x, a, b):
        return a * x + b
    
#     if (~np.isnan(Xdata)|~np.isinf(Xdata)).sum() > (~np.isnan(Ydata)|~np.isinf(Ydata)).sum():
    mask = remove_NaN_Inf_values(Xdata, Ydata)
    print('A')
    H = np.histogram2d(Xdata[mask], Ydata[mask], bins=100, range = srlimite)
    Hprobas = H[0]*100/len(Ydata[mask])
    noNaNpoints = len(Ydata[mask])
    # create the curve fit
    param, param_cov = curve_fit(objective, Xdata[mask], Ydata[mask])     
    print(param, param_cov)

    print(f'nombre de points no-NaN: {noNaNpoints}')
    xedges, yedges = np.meshgrid(H[1], H[2])
#     print(slope, intercept)
#     fitLine = slope * allsr532 + intercept
    return xedges, yedges, Hprobas, noNaNpoints


Xx, Yy, Hcounts, _ = get_params_histogram([[0,50], [0,100]], allsr355, allsr532)

# Histogram SR532/SR355

fig, ax = plt.subplots()
hist = ax.pcolormesh(Xx, Yy, Hcounts.T, norm=LogNorm(vmin=1e-5, vmax=1e-1))
c = plt.colorbar(hist, ax=ax, label='%')
plt.minorticks_on()
ax.set(xlabel='SR355', ylabel='SR532')
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
plt.savefig('/homedata/nmpnguyen/LNG/scatterSR_mask1.png')