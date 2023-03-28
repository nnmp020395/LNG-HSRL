import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import warnings
warnings.filterwarnings('ignore')

def plots(list_variables, list_coords, fig_params, fig_out):
    plt.rcParams['pcolor.shading'] = 'auto'
    plt.rcParams.update({'font.size':14})
    fig, axs = plt.subplots(nrows = fig_params['nrows'], ncols = fig_params['ncols'], figsize = fig_params['figsize'])
    
    cmap = plt.cm.turbo
    cmap.set_under('lightgrey')
    cmap.set_over('dimgrey')
    for k, ax in enumerate(axs.flat):
        print(list(list_variables.keys())[k])
        p = ax.pcolormesh(list_coords['time'], list_coords['height'].T, list(list_variables.values())[k].T, norm = fig_params['norm'][k], cmap = cmap) 
        ax.set(title = fig_params['titles'][k], xlabel = fig_params['xlabel'], ylabel = fig_params['ylabel'])
        ax.plot(list_coords['time'], fig_params['line_data'], color = 'k')
        ax.set_ylim(fig_params['ylim'])
        ax.set_xlim(fig_params['xlim'])
        cb = plt.colorbar(p, ax=ax, extend='both')
        cb.ax.tick_params(labelsize='large')
        ax.tick_params(axis = 'both', labelsize=13)
        # ax.set_xticks(fontsize=12)
        # ax.set_yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_out)
    plt.clf()
    plt.close()
    return 1

def hists(list_variables_x, list_variables_y, fig_params, fig_out):
    list_x = list(list_variables_x.values())
    list_y = list(list_variables_y.values())
    fig, axs = plt.subplots(nrows = fig_params['nrows'], ncols = fig_params['ncols'], figsize = fig_params['figsize'])
    plt.rcParams.update({'font.size':14})

    cmap = plt.cm.turbo
    cmap.set_under('lightgrey')
    cmap.set_over('dimgrey')

    for k, ax in enumerate(axs.flat):
        print(list(list_variables_x.keys())[k], list(list_variables_y.keys())[k])

        h = ax.hist2d(list_x[k], list_y[k], bins = fig_params['bins'])#, range = fig_params['range'][k], norm = fig_params['norm'][k])
        cb = plt.colorbar(p, ax=ax, extend='both')
        cb.ax.tick_params(labelsize='large')
        ax.set(title = fig_params['titles'][k], xlabel = fig_params['xlabel'][k], ylabel = fig_params['ylabel'][k])
    plt.tight_layout()
    plt.savefig(fig_out)
    plt.clf()
    plt.close()
    return 1   

def profils(list_x, list_coords, fig_params, fig_out, item_chosen):
    list_x = list(list_x.values())
    #
    # setting index of profil
    def find_nearest(array, value):
        idt = (np.abs(array - value)).argmin()
        value = array[idt]
        return idt, value

    index = find_nearest(list_coords['time'], item_chosen)[0]
    plt.rcParams.update({'font.size':14})

    fig, axs = plt.subplots(nrows = fig_params['nrows'], ncols = fig_params['ncols'], figsize = fig_params['figsize'])
    for k, ax in enumerate(axs.flat):
        print(list_x[k][index,:])
        ax.semilogx(list_x[k][index,:], list_coords['height'][index, :], label=fig_params['titles'][k], color='r')
        ax.set(title = f'{fig_params["titles"][k]}: {list_coords["time"][index]}', ylabel = 'Height, km')
        ax.axhline(fig_params['line_data'][index], color = 'k')
        ax.set_ylim(fig_params['ylim'])
        ax.tick_params(axis = 'both', labelsize=13)

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.clf()
    plt.close()
    return 1 



listfiles = sorted(Path('/homedata/nmpnguyen/LNG/NETCDF/').glob('*.nc'))
for i in range(1,len(listfiles))[1:2]: #len(listfiles)
    path355 = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset355.nc')
    path532 = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_dataset532.nc')
    if path355.is_file() & path532.is_file():
        print(f'{path355} and {path532} is found')
    
        new_d = xr.open_dataset(path355)
        new_dd = xr.open_dataset(path532)
        d = xr.open_dataset(listfiles[i], decode_cf=False)

        coords = {
            'time' : d['Time'].values,
            'height' : d['Height'].values,
        }

        # list_x1 = {
        #     'sr355' : new_d['ATB']/new_d['AMB'],
        #     'sr355' : new_d['ATB']/new_d['AMB'],
        #     'sr532' : new_dd['ATB']/new_dd['MMB'],            
        # }

        # list_y1 = {
        #     'sr532' : new_dd['ATB']/new_dd['MMB'],
        #     'sr532_esti' : new_dd['SR532_estimated'],
        #     'sr532_esti' : new_dd['SR532_estimated'],
        # }

        # params1 = {
        #     'xlabel' : ['SR355 measured', 'SR355 measured', 'SR532 measured'],
        #     'ylabel' : ['SR532 measured', 'SR532 estimated', 'SR532 estimated'],
        #     'titles' : ['SR355 measured vs SR532 measured', 'SR355 measured vs SR532 estimated ', 'SR532 measured vs SR532 estimated'],
        #     'nrows' : 1,
        #     'ncols' : 3,
        #     'figsize' : (16, 4),
        #     'norm' : [LogNorm(vmax=1e5), LogNorm(vmax=1e5), LogNorm(vmax=1e5)],
        #     'range' : [[[-1, 30], [-1, 80]], [[-1, 30], [-1, 80]], [[-1, 80], [-1, 80]]],
        #     'bins' : 100
        # }

        # hists(list_x1, list_y1, params1, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Scatterplot1.png'))


        # list_variables4 = {            
        #     'MPE355' : new_d['MPE'],
        #     'ratio_integ' : new_dd['T2_total']/new_dd['T2_532']
        # }

        # params4 = {
        #     'ylim' : (0, 15), 
        #     'xlim' : None,
        #     'xlabel' : 'Time',
        #     'ylabel' : 'Altitude, km',
        #     'titles' : ['Retrieved particular Extinction 355, m-1', 'T2_total/T2_532'],
        #     'nrows' : 2,
        #     'ncols' : 1,
        #     'figsize' : (8, 12),
        #     'norm' : [LogNorm(vmin=1e-4), LogNorm(vmin=1e-1, vmax=1e1)],
        #     'line_data' : d['Aircraft_Altitude'].values,
        # }
        # plots(list_variables4, coords, params4, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig7bis.png'))
        # profils(list_variables4, coords, params4, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig7profil.png'), item_chosen = 44300)

        # list_variables1 = {
        #     'ATB355' : new_d['ATB'].values,
        #     'MMB355' : new_d['MMB'].values,
        #     'MPAB355' : new_d['MPAB'].values,             
        # }
        
        # params1 = {
        #     'ylim' : (0, 15),
        #     'xlim' : (43000, coords['time'].max()),
        #     'xlabel' : 'Time',
        #     'ylabel' : 'Altitude, km',
        #     'titles' : ['ATB 355, m-1.sr-1', 'beta mol 355, m-1.sr-1' ,'AMB 355, m-1.sr-1'],
        #     'nrows' : 3,
        #     'ncols' : 1,
        #     'figsize' : (9, 15),
        #     'norm' : [LogNorm(vmin=1e-9, vmax=1e-5), LogNorm(vmin=1e-9, vmax=1e-5), LogNorm(vmin=1e-9, vmax=1e-5)],
        #     'line_data' : d['Aircraft_Altitude'].values,
        # }
        # # profils(list_variables1, coords, params1, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig5profil.png'), item_chosen = 44300)
        # plots(list_variables1, coords, params1, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig5.png'))


        # list_variables1 = {
        #     'ATB355' : new_d['ATB'].values,
        #     'MPAB355' : new_d['MPAB'].values, 
        #     'MME355' : new_d['MME'].values,
        # }
        
        # params1 = {
        #     'ylim' : (0, 15),
        #     'xlim' : (43000, coords['time'].max()),
        #     'xlabel' : 'Time',
        #     'ylabel' : 'Altitude, km',
        #     'titles' : ['ATB 355, m-1.sr-1','AMB 355, m-1.sr-1', 'Model Molecular Extinction 355, m-1'],
        #     'nrows' : 3,
        #     'ncols' : 1,
        #     'figsize' : (9, 15),
        #     'norm' : [LogNorm(vmin=1e-9, vmax=1e-5), LogNorm(vmin=1e-9, vmax=1e-5), LogNorm()],
        #     'line_data' : d['Aircraft_Altitude'].values,
        # }
        # # profils(list_variables1, coords, params1, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig5profil.png'), item_chosen = 44300)
        # plots(list_variables1, coords, params1, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig4.png'))

        # list_variables2 = {
        #     'MMB532' : new_dd['MMB'],
        #     'MPB355' : new_d['MPB'],
        #     'const' : (new_dd['MMB']+new_d['MPB'])/new_dd['MMB'],
        # }

        # params2 = {
        #     'ylim' : (0, 15), 
        #     'xlim' : (43000, coords['time'].max()),
        #     'xlabel' : 'Time',
        #     'ylabel' : 'Altitude, km',
        #     'titles' : ['Model Molecular Backscatter 532, m-1.sr-1','Retrieved Particular Backscatter 355, m-1.sr-1', '(beta_mol_532 + beta_part_355)/beta_mol_532'],
        #     'nrows' : 3,
        #     'ncols' : 1,
        #     'figsize' : (9, 15),
        #     'norm' : [LogNorm(vmin=1e-7, vmax=1e-5), LogNorm(vmin=1e-7, vmax=1e-5), LogNorm(vmin=1e0, vmax=1e2)],
        #     'line_data' : d['Aircraft_Altitude'].values,
        # }
        # profils(list_variables2, coords, params2, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig6profil.png'), item_chosen = 36000)
        # plots(list_variables2, coords, params2, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig6.png'))

        list_variables3 = {            
            'MMB532' : new_dd['MME'],
            'etotal' : new_dd['etotal'],
            # 'alpha532_integ' : new_dd['alpha532_integ'],            
            # 'etotal_interg' : new_dd['etotal_interg'],
            'T2_532' : new_dd['T2mol'],
            'T2_total' : new_dd['T2_total'],
        }

        params3 = {
            'ylim' : (0, 15), 
            'xlim' : None,
            'xlabel' : 'Time',
            'ylabel' : 'Altitude, km',
            'titles' : ['alpha_mol_532, m-1','alpha_sum = alpha_532+n*alpha_part_355, m-1', 'exp(-2 * alpha_mol_532_integrated)', 'exp(-2 * alpha_sum)'], #, 'alpha_mol_532_integrated', 'alpha_sum integrated', 
                        # 
            'nrows' : 2,
            'ncols' : 2,
            'figsize' : (16, 15),
            'norm' : [LogNorm(vmin=1e-6, vmax=1e-4), LogNorm(vmin=1e-6, vmax=1e-4), Normalize(vmin=0.5, vmax=1), Normalize(vmin=0.5, vmax=1)],
            'line_data' : d['Aircraft_Altitude'].values,
        }
        # profils(list_variables3, coords, params3, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig7profil.png'), item_chosen = 36000)
        plots(list_variables3, coords, params3, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig7.png'))

        # list_variables5 = {
        #     'SR532': new_dd['SR532'].values,
        #     'SR532_estimated' : new_dd['SR532_estimated'].values,
        #     'SR355' : new_d['SR355'].values,
        #     'diff' : (new_dd['SR532']-new_dd['SR532_estimated']).values
        # }

        # params5 = {
        #     'ylim' : (0, 15), 
        #     'xlim' : (43500, coords['time'].max()),
        #     'xlabel' : 'Time',
        #     'ylabel' : 'Altitude, km',
        #     'titles' : ['SR355 measured', 'SR532 measured', 'SR532 estimated', 'SR355 measured - SR532 estimated'],
        #     'nrows' : 4,
        #     'ncols' : 1,
        #     'norm' : [LogNorm(vmin=1e-1, vmax=1e2), LogNorm(vmin=1e-1, vmax=1e2), LogNorm(vmin=1e-1, vmax=1e2), LogNorm(vmin=1e-2, vmax=1e1)],
        #     'figsize' : (7, 15),
        #     'line_data' : d['Aircraft_Altitude'].values,
        # }
        # # profils(list_variables5, coords, params5, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig3profil.png'), item_chosen = 36000)
        # plots(list_variables5, coords, params5, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig3.png'))

        list_variables6 = {
            'ATB532_estimate' : (new_dd['MMB']+new_d['MPB'])/new_dd['T2_total'],
            'beta_mol_532' : (new_dd['MMB']*new_dd['T2mol'])
        }

        params6 = {
            'ylim' : (0, 15), 
            'xlim' : (coords['time'].min(), 37000),
            'xlabel' : 'Time',
            'ylabel' : 'Altitude, km',
            'titles' : ['ATB532 estimate, m-1.sr-1', 'beta mol attenuated 532, m-1.sr-1'],
            'nrows' : 2,
            'ncols' : 1,
            'norm' : [LogNorm(vmin=1e-7, vmax=1e-3), LogNorm(vmin=1e-7, vmax=1e-3)],
            'figsize' : (9, 10),
            'line_data' : d['Aircraft_Altitude'].values,
        }
        # profils(list_variables5, coords, params5, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig3profil.png'), item_chosen = 36000)
        # plots(list_variables6, coords, params6, fig_out = Path('/homedata/nmpnguyen/LNG/test_Artem', f'{listfiles[i].stem}_Fig8.png'))


    else:
        print('File cannot found')