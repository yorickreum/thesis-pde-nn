from scipy.io import netcdf
import numpy as np
import pandas as pd

wsi_2_file = netcdf.NetCDFFile(f"./fuer_yorick/data/wsi_1-5_2_remap.nc", "r", mmap=True)
print(wsi_2_file.variables)
# OrderedDict([
# ('time', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810c9e8>),
# ('lon', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810c978>),
# ('lat', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810cac8>),
# ('lev', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810cb70>),
# ('hyai', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810cc88>),
# ('hybi', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810cd30>),
# ('hyam', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810cdd8>),
# ('hybm', <scipy.io.netcdf.netcdf_variable object at 0x7f0f7810cef0>),
# ('var1', <scipy.io.netcdf.netcdf_variable object at 0x7f0f78114048>),
# ('var2', <scipy.io.netcdf.netcdf_variable object at 0x7f0f781140f0>),
# ('var3', <scipy.io.netcdf.netcdf_variable object at 0x7f0f78114198>),
# ('var4', <scipy.io.netcdf.netcdf_variable object at 0x7f0f78114240>),
# ('var5', <scipy.io.netcdf.netcdf_variable object at 0x7f0f781142e8>)
# ])

time = wsi_2_file.variables['time']
print(time.units)
print(time.shape)
print(time[0])
print(time[-1])

lat = wsi_2_file.variables['lat']
print(lat.units)
print(lat.shape)
# lat[54][54] == 49.471848
# lon[71][71] == 10.454959

var1 = wsi_2_file.variables['var1']
var2 = wsi_2_file.variables['var2']
var3 = wsi_2_file.variables['var3']
var4 = wsi_2_file.variables['var4']
var5 = wsi_2_file.variables['var5']

var_i54j71 = np.array([[None, None, None, None, None, None]] * var1.shape[0])
i, j = 54, 71
for t in range(var1.shape[0]):
    print("t: " + str(t))
    var_i54j71[t][0] = time[t]
    var_i54j71[t][1] = var1[t][0][i][j]
    var_i54j71[t][2] = var2[t][0][i][j]
    var_i54j71[t][3] = var3[t][0][i][j]
    var_i54j71[t][4] = var4[t][0][i][j]
    var_i54j71[t][5] = var5[t][0][i][j]

# np.savetxt(f'./data/var_i54j71.txt', var_i54j71, delimiter=",")
pd.DataFrame(var_i54j71).to_csv(f'./data/var_i54j71_2.txt', index=False, header=False)

wsi_2_file.close()
