import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def find_nearest(array, value):
#     time_array = pd.to_datetime(time_array)
    idt = (np.abs(array - value)).argmin()
    value = array[idt]
    return idt, value


def get_AMB_APB(data, channel):
    '''
    Extraire les données nécessaires à utiliser et renommer 
        MPAB = Attenuated Molecular Backscatter
        ATB = Attenuated Backscatter
        MMB = Model Molecular Backscatter
        MPB = Model Particular Backscatter
        MME = Model Molecular Extinction
        MPE = Model Particular Extinction 
    '''
    if channel == 355: 
        config_data_vars = {
            f'LNG_Molecular_Parallel_Attenuated_Backscatter_{channel}':'MPAB',
            f'LNG_Particulate_Parallel_Attenuated_Backscatter_{channel}':'PPAB',
            f'LNG_Parallel_Attenuated_Backscatter_{channel}':'PAB',
            f'LNG_Perpendicular_Attenuated_Backscatter_{channel}':'PerpAB',
        }
    else:
        config_data_vars = {
            f'LNG_Total_Attenuated_Backscatter_{channel}':'ATB',
        }
    config_data_vars.update({
        f'Model_Molecular_Backscatter_{channel}':'MMB',
        f'Model_Molecular_Extinction_{channel}':'MME',
        f'Model_Molecular_Transmittance_{channel}':'T2mol',
        'Height' : 'Height', 
        'Time' : 'Time'
    })
#     config_data_vars = [f'LNG_Molecular_Parallel_Attenuated_Backscatter_{channel}', 
#                        f'LNG_Particulate_Parallel_Attenuated_Backscatter_{channel}',
#                        f'Model_Molecular_Backscatter_{channel}',
#                        'Height']
#     print(config_data_vars.keys())
    data = data.where((data['Validity_rate']==1) & (data['LNG_UpDown']==1), drop=False)
    tmp_data = data[list(config_data_vars.keys())]
    tmp_data = tmp_data.rename_vars(config_data_vars)
    return tmp_data

listfiles = sorted(Path('/homedata/nmpnguyen/LNG/NETCDF/').glob('*.nc'))
from sklearn.metrics import mean_absolute_error, r2_score
maes = []
r2s = []
for i in range(len(listfiles))[2:3]:
    d = xr.open_dataset(listfiles[i],decode_cf=False)

    print('--------Extraire les données nécessaires---------')
    new_dd = get_AMB_APB(d, 532)
    new_d = get_AMB_APB(d, 355)
    idt = np.where(~np.isnan(new_d['Time']).all(axis=1))[0]

    print('--------Trouver Alt sol et Alt Aircraft de chaque profil--------')
    sol_ind = [find_nearest(d['Height'][r,:].values, 0.0) for r in range(d['Height'].shape[0])]
    alt_aircraft_ind = [find_nearest(d['Height'][r,:].values, d['Aircraft_Altitude'].values[r]-0.2) for r in range(d['Height'].shape[0])]

    print('--------Calculer Particulaire Extinction et Backscatter en 355nm----------')
    C = (-np.log(new_d['MPAB']/new_d['MMB'])/2).values # --> transmittance 2 way
    A = np.full(new_d['Height'].shape, np.nan) 
    PartExt355 = np.full(new_d['Height'].shape, np.nan) 

    for t in tqdm(idt): #range(d['Time'].values.shape[0])
        depart = alt_aircraft_ind[t][0]
        fin = sol_ind[t][0]
        A[t, depart-1] = 0.0
        # for k in range(depart, fin): #d['Height'].shape[1]
        #     delta_z = np.abs(d['Height'][t, k-1] - d['Height'][t, k]).values*1e3
        #     A[t, k] = ((C[t, k] - C[t, k-1])/(delta_z))
        if fin==0:
            delta_z = np.abs(d['Height'][t, depart-1:fin-1] - d['Height'][t, depart:]).values*1e3
            A[t, depart:] = ((C[t, depart:] - C[t, depart-1:fin-1])/(delta_z))
        else:
            delta_z = np.abs(d['Height'][t, depart-1:fin-1] - d['Height'][t, depart:fin]).values*1e3
            A[t, depart:fin] = ((C[t, depart:fin] - C[t, depart-1:fin-1])/(delta_z))

    PartExt355 = A - new_d.MME.values
    PartBack355 = (new_d['PPAB']*new_d['MMB'])/new_d['MPAB']

    print('--------Reconstruire Equation d Artem---------')
    alpha532 = new_dd['MME'].values
    alpha532_integ = np.full(alpha532.shape, np.nan)
    T2_532 = np.full(alpha532.shape, np.nan)
    for t in tqdm(idt): #range(d['Time'].values.shape[0])
        depart = alt_aircraft_ind[t][0]
        fin = sol_ind[t][0]
        alpha532_integ[t, depart-1] = 0.0
        # for k in range(depart, fin): 
        #     delta_z = np.abs(d['Height'][t, k] - d['Height'][t, k-1]).values*1e3
        #     alpha532_integ[t, k] = np.nansum(alpha532_integ[t, k-1] + delta_z*alpha532[t, k])
        #     T2_532[t, k] = np.exp(-2 * alpha532_integ[t, k])
        if fin==0:
            delta_z = np.abs(d['Height'][t, depart-1:fin-1] - d['Height'][t, depart:]).values*1e3
            alpha532_integ[t, depart:] = delta_z*alpha532[t, depart:]
            alpha532_integ[t, depart:] = alpha532_integ[t, depart:] + alpha532_integ[t, depart-1:fin-1]
            T2_532[t, depart:] = np.exp(-2 * alpha532_integ[t, depart:])
        else:
            delta_z = np.abs(d['Height'][t, depart-1:fin-1] - d['Height'][t, depart:fin]).values*1e3
            alpha532_integ[t, depart:fin] = delta_z*alpha532[t, depart:fin]
            alpha532_integ[t, depart:fin] = alpha532_integ[t, depart:fin] + alpha532_integ[t, depart-1:fin-1]
            T2_532[t, depart:fin] = np.exp(-2 * alpha532_integ[t, depart:fin])

    etotal = np.nansum(np.dstack((alpha532, 0.9*PartExt355)),2)
    etotal_interg = np.full(alpha532.shape, np.nan)
    T2_total = np.full(alpha532.shape, np.nan)
    for t in tqdm(idt): #range(d['Time'].values.shape[0])
        depart = alt_aircraft_ind[t][0]
        fin = sol_ind[t][0]
        etotal_interg[t, depart-1] = 0.0
        # for k in range(depart, fin): 
        #     delta_z = np.abs(d['Height'][t, k] - d['Height'][t, k-1]).values*1e3
        #     etotal_interg[t, k] = etotal_interg[t, k-1] + delta_z*etotal[t, k]
        #     T2_total[t, k] = np.exp(-2 * etotal_interg[t, k])
        if fin==0:
            delta_z = np.abs(d['Height'][t, depart-1:fin-1] - d['Height'][t, depart:]).values*1e3
            etotal_interg[t, depart:] = delta_z*etotal[t, depart:]
            etotal_interg[t, depart:] = etotal_interg[t, depart:] + etotal_interg[t, depart-1:fin-1]
            T2_total[t, depart:] = np.exp(-2 * etotal_interg[t, depart:])
        else:
            delta_z = np.abs(d['Height'][t, depart-1:fin-1] - d['Height'][t, depart:fin]).values*1e3
            etotal_interg[t, depart:fin] = delta_z*etotal[t, depart:fin]
            etotal_interg[t, depart:fin] = etotal_interg[t, depart:fin] + etotal_interg[t, depart-1:fin-1]
            T2_total[t, depart:fin] = np.exp(-2 * etotal_interg[t, depart:fin])

    const_sr532_esti = (new_dd.MMB + PartBack355)/new_dd.MMB
    t2 = T2_total/T2_532
    SR532_esti = (const_sr532_esti*t2).values

    coords = {'time' : new_d['time'].values,'height' : new_d['height'].values}
    dims = ['time', 'height']

    new_d['Time'] = d['Time']
    new_dd['Time'] = d['Time']
    new_d['Height'] = d['Height']
    new_dd['Height'] = d['Height']

    PartExt355 = xr.DataArray(data = PartExt355, dims = dims, coords = coords)
    PartBack355 = xr.DataArray(data = PartBack355, dims = dims, coords = coords)

    new_d['MPB'] = PartBack355
    new_d['MPE'] = PartExt355

    SR532_esti = xr.DataArray(data = SR532_esti, dims = dims, coords = coords)
    alpha532_integ = xr.DataArray(data = alpha532_integ, dims = dims, coords = coords)
    T2_532 = xr.DataArray(data = T2_532, dims = dims, coords = coords)
    etotal = xr.DataArray(data = etotal, dims = dims, coords = coords)
    etotal_interg = xr.DataArray(data = etotal_interg, dims = dims, coords = coords)
    T2_total = xr.DataArray(data = T2_total, dims = dims, coords = coords)

    new_dd['SR532_estimated'] = SR532_esti
    new_dd['alpha532_integ'] = alpha532_integ
    new_dd['T2_532'] = T2_532
    new_dd['etotal'] = etotal
    new_dd['etotal_interg'] = etotal_interg
    new_dd['T2_total'] = T2_total
    new_dd['SR532'] = xr.DataArray(data = new_dd['ATB']/(new_dd['MMB']*new_dd['T2mol']), dims = dims, coords = coords)

    new_d['SR355'] = xr.DataArray(data = (new_d['PAB']+new_d['PerpAB'])/(new_d['MMB']*new_d['T2mol']), dims = dims, coords = coords)

    print(listfiles[i])
    print(new_d)
    # if Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset355.nc').is_file():
    #     Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset355.nc').unlink()
    # if Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset532.nc').is_file():
    #     Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset532.nc').unlink()
    new_d.to_netcdf(Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset355_v2.nc'))
    new_dd.to_netcdf(Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset532_v2.nc'))

    # print('MEAN ABSOLUTE ERROR')
    # pred = SR532_esti[np.logical_and(np.isfinite((new_dd.ATB/new_dd.MMB).values.ravel()), np.isfinite(SR532_esti.values.ravel()))]
    # real = (new_dd.ATB/new_dd.MMB).values[np.logical_and(np.isfinite((new_dd.ATB/new_dd.MMB).values.ravel()), np.isfinite(SR532_esti.values.ravel()))]
    # mae = mean_absolute_error(real, pred)
    # maes.append(mae)
    # r2 = r2_score(real, pred)
    # r2s.append(r2)

# print(listfiles, maes)
# with open('/homedata/nmpnguyen/LNG/test_Artem/maes.txt') as f:
#     f.write(maes)

# with open('/homedata/nmpnguyen/LNG/test_Artem/r2s.txt') as f:
#     f.write(r2s)
# fig, ((a0, a1), (a2, a3), (a4, a5)) = plt.subplots(ncols=2, nrows=3, figsize=(15, 15), sharex=True, sharey=True)
# plt.rcParams['pcolor.shading'] = 'auto'
# plt.rcParams['font.size']=11

# cmap = plt.cm.turbo
# cmap.set_under('lightgrey')
# cmap.set_over('dimgrey')

# p = a0.pcolormesh(d['Time'].values, d['Height'].values.T, new_d['MME'].values.T, norm=LogNorm(), cmap=cmap)
# plt.colorbar(p, ax=a0, label = 'model molecular extinction 355, m-1', extend='both')
# a0.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a0.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# a0.set_ylim(0, 15)
# # a0.set_xlim(54000, 56000)

# p = a1.pcolormesh(d['Time'].values, d['Height'].values.T, PartExt355.T, norm=LogNorm(vmin=1e-5, vmax=1e-2), cmap=cmap)
# plt.colorbar(p, ax=a1, label='retrieved particular extinction 355, m-1', extend='both')
# a1.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a1.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# a1.set_ylim(0, 15)
# # a1.set_xlim(54000, 56000)

# p = a2.pcolormesh(d['Time'].values, d['Height'].values.T, new_d['MMB'].values.T, norm=LogNorm(), cmap=cmap)
# plt.colorbar(p, ax=a2, label='model molecular backscatter 355, m-1.sr-1', extend='both')  
# a2.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a2.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')

# p = a3.pcolormesh(d['Time'].values, d['Height'].values.T, PartBack355.T, norm=LogNorm(vmin=1e-8), cmap=cmap)
# plt.colorbar(p, ax=a3, label='retrieved particular backscatter 355, m-1.sr-1', extend='both')
# a3.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a3.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')

# p = a4.pcolormesh(d['Time'].values, d['Height'].values.T, new_d['MPAB'].values.T, norm=LogNorm(vmin=1e-8, vmax=1e-5), cmap=cmap)
# plt.colorbar(p, ax=a4, label='attenuated molecular backscatter 355, m-1.sr-1', extend='both')
# a4.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a4.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')

# p = a5.pcolormesh(d['Time'].values, d['Height'].values.T, new_d['ATB'].values.T, norm=LogNorm(vmin=1e-8, vmax=1e-5), cmap=cmap)
# plt.colorbar(p, ax=a5, label='attenuated particular backscatter 355, m-1.sr-1', extend='both')
# a5.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a5.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')

# plt.savefig(Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig1.png'))
# plt.close()
# plt.clf()

# fig, ((a0, a1, a11), (a2, a3, a31), (a4, a5, a51)) = plt.subplots(ncols=3, nrows=3, figsize=(20, 15), sharex=True, sharey=True)
# plt.rcParams['pcolor.shading'] = 'auto'
# plt.rcParams['font.size']=11

# cmap = plt.cm.turbo
# cmap.set_under('lightgrey')
# cmap.set_over('dimgrey')

# p = a0.pcolormesh(d['Time'].values, d['Height'].values.T, alpha532.T, norm=LogNorm(vmin=6e-6, vmax=5e-5), cmap=cmap)
# a0.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a0.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a0, label = 'AlphaMol 532\n(m-1)', extend='both')
# a0.set_ylim(0, 15)
# # a0.set_xlim(54000, 56000)

# p = a1.pcolormesh(d['Time'].values, d['Height'].values.T, alpha532_integ.T, norm=LogNorm(vmin=1e-3, vmax=1e-1), cmap=cmap)
# a1.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a1.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a1, label='AlphaMol 532 integrated\n(m-1)', extend='both')
# a1.set_ylim(0, 15)
# # a1.set_xlim(54000, 56000)

# p = a11.pcolormesh(d['Time'].values, d['Height'].values.T, T2_532.T, norm=LogNorm(), cmap=cmap)
# a11.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a11.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a11, label='exp(-2*(AlphaMol 532 integrated))\n(m-1)', extend='both')

# p = a2.pcolormesh(d['Time'].values, d['Height'].values.T, etotal.T, norm=LogNorm(vmin=1e-6, vmax=5e-3), cmap=cmap)
# a2.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a2.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a2, label='AlphaMol 532 + 0.9*AlphaPart355\n(m-1)', extend='both')  

# p = a3.pcolormesh(d['Time'].values, d['Height'].values.T, etotal_interg.T, norm=LogNorm(vmin=1e-1, vmax=5e0), cmap=cmap)
# a3.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a3.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a3, label='AlphaMol 532 + 0.9*AlphaPart355 integrated\n(m-1)', extend='both')

# p = a31.pcolormesh(d['Time'].values, d['Height'].values.T, T2_total.T, norm=LogNorm(vmin=1e-1, vmax=1e1), cmap=cmap)
# a31.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a31.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a31, label='exp -2(AlphaMol 532 + 0.9*AlphaPart355 integrated)\n(m-1)', extend='both')

# p = a4.pcolormesh(d['Time'].values, d['Height'].values.T, const_sr532_esti.T, norm=LogNorm(vmin=1e0, vmax=1e2), cmap=cmap)
# a4.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a4.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a4, label='(BackMol532 + BackPart355)/BackMol532', extend='both')

# p = a5.pcolormesh(d['Time'].values, d['Height'].values.T, SR532_esti.T, norm=LogNorm(vmin=1e-1, vmax=1e2), cmap=cmap)
# a5.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a5.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a5, label='SR532 estimated', extend='both')

# p = a51.pcolormesh(d['Time'].values, d['Height'].values.T, (new_d.ATB/new_d.MPAB).values.T, norm=LogNorm(vmin=1e-1, vmax=1e2), cmap=cmap)
# a51.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k')
# a51.plot(d['Time'].values, d['Aircraft_Altitude'].values, color = 'k', linestyle = '--')
# plt.colorbar(p, ax=a51, label='SR532 mesured', extend='both')

# plt.savefig(Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig2.png'))
# plt.close()
# plt.clf()