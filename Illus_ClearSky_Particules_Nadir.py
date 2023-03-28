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



def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# find_nearest_index(data['Time'].values, 70000)
key532 = ['raw_HSR_Signal_532', 
          'Model_Molecular_Backscatter_532', 
          'Model_Molecular_Extinction_532',
          'Model_Molecular_Transmittance_532', 
          'LNG_Total_Attenuated_Backscatter_532']

key355 = ['raw_HSR_Signal_355', 
        'Model_Molecular_Backscatter_355', 
        'Model_Molecular_Extinction_355',
        'Model_Molecular_Transmittance_355', 
        'LNG_Parallel_Attenuated_Backscatter_355',
        'LNG_Perpendicular_Attenuated_Backscatter_355']

time_to_see = [0, 78000, 36000, 54200, 31200, 72000, 68000, 60000, 37000]


for filepath, i in zip(LNG_LISTFILES[4:5], range(4,5)):
    print(filepath)
    fig, ((ax, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3, figsize=(10,12))
    data = xr.open_dataset(filepath)
    valid_rate = (data['Validity_rate'] == 1) 
    height, time, Pointing = data['Height'].isel(time=valid_rate).values , data['Time'].isel(time=valid_rate).values, data['LNG_UpDown'].isel(time=valid_rate).values
    maskHSR532 = data['Mask_532'].isel(time=valid_rate).values
    idx = find_nearest_index(time, time_to_see[i])
    limitez = np.where((height[idx,:] > 2.5) & (height[idx,:] < 3))[0]
    # 532nm
    mol = data[key532[1]].isel(time=valid_rate) * data[key532[3]].isel(time=valid_rate)
    atb = data[key532[-1]].isel(time=valid_rate)
#     pr2_integ = 0; mol_attn_integ = 0
#     for z in limitez[:-1]:
#         pr2_integ = pr2_integ + atb[idx,z]*(height[idx, z+1] - height[idx, z])
#         mol_attn_integ = mol_attn_integ + mol[idx,z]*(height[idx, z+1] - height[idx, z])

#     const = (mol_attn_integ/pr2_integ).reshape(-1,1)
#     print(const)
    sr = atb/mol
    sr532 = sr.where(((Pointing == 1)& np.isin(maskHSR532, [0,1,2,3])), drop=False)
    sr532_clear = sr.where(((Pointing == 1)& (maskHSR532 == 0)), drop=False)
    atb532 = atb.where(((Pointing == 1)& np.isin(maskHSR532, [0,1,2,3])), drop=False)
    mol532 = mol.where(((Pointing ==1)& np.isin(maskHSR532, [0,1,2,3])), drop=False)
    ax3.semilogx(mol532[idx,:], height[idx,:], '--', color='g')
    # 355nm
    mol = data[key355[1]].isel(time=valid_rate) * data[key355[3]].isel(time=valid_rate)
    atb = data[key355[-2]].isel(time=valid_rate) + data[key355[-1]].isel(time=valid_rate)
#     pr2_integ = 0; mol_attn_integ = 0
#     for z in limitez[:-1]:
#         pr2_integ = pr2_integ + atb[:,z]*(height[:, z+1] - height[:, z])
#         mol_attn_integ = mol_attn_integ + mol[:,z]*(height[:, z+1] - height[:, z])

#     const = (mol_attn_integ/pr2_integ).reshape(-1,1)
#     print(const[idx])
    sr = atb/mol
    sr355 = sr.where(((Pointing == 1)& np.isin(maskHSR532, [0,1,2,3])), drop=False) #np.ma.masked_where(Pointing!=1, sr)
    sr355_clear = sr.where(((Pointing == 1)& (maskHSR532 == 0)), drop=False)
    atb355 = atb.where(((Pointing == 1)& np.isin(maskHSR532, [0,1,2,3])), drop=False)
    mol355 = mol.where(((Pointing ==1)& np.isin(maskHSR532, [0,1,2,3])), drop=False)
    ax3.semilogx(mol355[idx,:], height[idx,:], '--', color='b')
    
    nb_profils = sr532.dropna(how='all', dim='time').shape[0]
    pcm = ax.pcolormesh(time, height.T, sr532.values.T, cmap='turbo', vmin=0, vmax=20)
    ax.set_ylim(-1,18)
    plt.colorbar(pcm, ax=ax, label='SR532')
    ax.set(xlabel='Time, Sec UTC', ylabel='Alt [km]', title=f'{filepath.stem.split("_")[6]}, {nb_profils} profils')
    
    nb_profils = sr355.dropna(how='all', dim='time').shape[0]
    pcm = ax2.pcolormesh(time, height.T, sr355.values.T, cmap='turbo', vmin=0, vmax=20)
    ax2.set_ylim(-1,18)
    plt.colorbar(pcm, ax=ax2, label='SR355')
    ax2.set(xlabel='Time, Sec UTC', ylabel='Alt [km]', title=f'{filepath.stem.split("_")[6]}, {nb_profils} profils')
    

    ax.axvline(time[idx], color="black", linestyle="--")
    ax2.axvline(time[idx], color="black", linestyle="--")
    ax3.semilogx(atb532[idx,:], height[idx,:], color='g', label='532')
    ax3.semilogx(atb355[idx,:], height[idx,:], color='b', label='355')    
    ax3.set_ylim(-1, 12)
    ax3.set_xlim(1e-08, 1e-5)
    ax3.legend(loc='best')
    ax3.set(xlabel='ATB', ylabel='Alt [km]', title=f'{filepath.stem.split("_")[6]}, {time[idx]}')

    ax4.plot((atb532/mol532)[idx,:], height[idx,:], color='g', label='532')
    ax4.plot((atb355/mol355)[idx,:], height[idx,:], color='b', label='355')    
    ax4.axvline(1, color="black", linestyle="--") 
    ax4.set_ylim(-1, 10)
    ax4.set_xlim(0,8)
    ax4.legend(loc = 'best')
    ax4.set(xlabel='SR', ylabel='Alt [km]', title=f'{filepath.stem.split("_")[6]}, {time[idx]}')
    
    Xx, Yy, Hcounts, _ = get_params_histogram([[0,40], [0,80]], sr355.values.flatten(), sr532.values.flatten())
    hist = ax5.pcolormesh(Xx, Yy, Hcounts.T, norm=LogNorm(vmin=1e-5, vmax=1e-0))
    c = plt.colorbar(hist, ax=ax5, label='%')
    plt.minorticks_on()
    ax5.set(xlabel='SR355', ylabel='SR532', title='clear sky + particules')
    ax5.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    
    Xx, Yy, Hcounts, _ = get_params_histogram([[0,40], [0,80]], sr355_clear.values.flatten(), sr532_clear.values.flatten())
    hist = ax6.pcolormesh(Xx, Yy, Hcounts.T, norm=LogNorm(vmin=1e-5, vmax=1e-0))
    c = plt.colorbar(hist, ax=ax6, label='%')
    plt.minorticks_on()
    ax6.set(xlabel='SR355', ylabel='SR532', title='clear sky')
    ax6.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(Path('/homedata/nmpnguyen/LNG/Figs/', f'Illus_ClearSky_Particules_Nadir_{filepath.stem}.png'))