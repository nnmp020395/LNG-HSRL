import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

def get_AMB_APB(data, channel):
    if channel == 355: 
        config_data_vars = {
            f'LNG_Molecular_Parallel_Attenuated_Backscatter_{channel}':'AMB',
            f'LNG_Particulate_Parallel_Attenuated_Backscatter_{channel}':'APB',
            f'Model_Molecular_Backscatter_{channel}':'MMB',
            f'Model_Molecular_Extinction_{channel}':'MME',
            'Height' : 'Height', 
            'Time' : 'Time'
        }
    else:
        config_data_vars = {
            f'LNG_Total_Attenuated_Backscatter_{channel}':'APB',
            f'Model_Molecular_Backscatter_{channel}':'MMB',
            f'Model_Molecular_Extinction_{channel}':'MME',
            'Height' : 'Height',
            'Time' : 'Time'
        }
#     config_data_vars = [f'LNG_Molecular_Parallel_Attenuated_Backscatter_{channel}', 
#                        f'LNG_Particulate_Parallel_Attenuated_Backscatter_{channel}',
#                        f'Model_Molecular_Backscatter_{channel}',
#                        'Height']
#     print(config_data_vars.keys())
    data = data.where((data['Validity_rate']==1) & (data['LNG_UpDown']==1), drop='True')
    tmp_data = data[list(config_data_vars.keys())]
    tmp_data = tmp_data.rename_vars(config_data_vars)
    return tmp_data



def compute_particular_backscatter(AMB, MMB, MME, APB, alt):
    def compute_transmittance_2way(AMB, MMB, MME, alt):
        # Calculer la transmittance 2-ways from AMB-Attenuated Molecular Backscatter & MMB-Model Molecular Backscatter
        # T2 = transmittance 2-ways
        T2 = -np.log(AMB/MMB)/2
        print(T2.shape)
        C = np.full(T2.shape, np.nan)#np.array([[],[]])
        C[:,0] = T2[:,0]

        for i in range(1, T2.shape[1]):
            C[:,i] = T2[:,i] - C[:, i-1]*(alt[:, i]-alt[:, i-1])
        # Calculer Extinction particular
        MPE = T2 - MME
        return T2, MPE
    T2, MPE = compute_transmittance_2way(AMB, MMB, MME, alt)    
    MPB = APB/T2
    return MPB, MPE


listfiles = sorted(Path('/homedata/nmpnguyen/LNG/NETCDF/').glob('*.nc'))
d = xr.open_dataset(listfiles[3],decode_cf=False)
d_new = get_AMB_APB(d, 355)

mpb_532, mpe_532 = compute_particular_backscatter(d_new['AMB'], d_new['MMB'], d_new['MME'], d_new['APB'], d_new['Height'])
