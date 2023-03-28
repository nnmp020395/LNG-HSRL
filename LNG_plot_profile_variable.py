import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from argparse import Namespace, ArgumentParser  

parser = ArgumentParser()
parser.add_argument("--filepath", "-f", type=str, help="input path of LNG file", required=True)
parser.add_argument("--variable", "-v", type=str, help="variale name choosen to plot", required=True)
parser.add_argument("--pointing", "-pt", type=str, help="choose among nadir/zenith/ADM", required=True)
opts = parser.parse_args()
print(opts)

filepath = Path(opts.filepath)
variable = opts.variable
pointing = opts.pointing

# def plot_profile_variable(filepath, variable, pointing):
import random
"""
- Plot script -
---------------
Input
    - filepath : path of LNG file
    - variable : variale name choosen to plot
    - pointing: choose among nadir/zenith/ADM
---------------
Return Plot
"""
data = xr.open_dataset(filepath)
validity_rate = (data['Validity_rate'] == 1)
time = data['Time'].isel(time=validity_rate)
height = data['Height'].isel(time=validity_rate)
variable_data = data[variable].isel(time=validity_rate)
pointing_data = data['LNG_UpDown'].isel(time=validity_rate)
if pointing == 'nadir':
    pointing_nb = 1
elif pointing == 'zenith':
    pointing_nb = 2
else:
    pointing_nb = 3

    
masked_variable_data = np.ma.masked_where(pointing_data.values != pointing_nb, variable_data)
# set random profil's index 
#--------------------------
random_profil_index = random.randint(0, len(time.values))

figure, axe = plt.subplots(figsize=(5,6))
axe.semilogx(masked_variable_data[random_profil_index,:], height[random_profil_index,:])
axe.set_ylim(-1, 18)
axe.set(title=f'{filepath.stem}\n{time[random_profil_index].values}', xlabel=variable, ylabel='Alt, km')
plt.show()
    
    # return 
