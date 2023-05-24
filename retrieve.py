import numpy as np
import xarray as xr
from numpy.lib.arraysetops import unique
from utils import calc_dry_air_column

data_vars = dict(
oco2 = [
    "time",
    "latitude",
    "longitude",
    "xco2",
    "vertex_latitude",
    "vertex_longitude",
    "xco2_quality_flag",
    "xco2_uncertainty",
    "Retrieval/psurf",
],
oco3 = [
    "time",
    "latitude",
    "longitude",
    "xco2",
    "vertex_latitude",
    "vertex_longitude",
    "xco2_quality_flag",
    "xco2_uncertainty",
    "Retrieval/psurf",
],
tomi_co = [
    "PRODUCT/latitude",
    "PRODUCT/longitude",
    "PRODUCT/carbonmonoxide_total_column_corrected",
    "PRODUCT/carbonmonoxide_total_column_precision",
    "PRODUCT/qa_value",
    "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_total_column",
    "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/pressure_levels",
    "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
    "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
    "PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude",
],
tomi_no2 = [
        'PRODUCT/latitude',
        'PRODUCT/longitude',
        'PRODUCT/air_mass_factor_total',
        'PRODUCT/nitrogendioxide_tropospheric_column',
        'PRODUCT/nitrogendioxide_tropospheric_column_precision',
        'PRODUCT/qa_value',
        'PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',
        'PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',
        'PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure',
        'PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude',
        'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_slant_column_density',
        'PRODUCT/tm5_constant_a',
        'PRODUCT/tm5_constant_b',
])


def ds_groups(keys):
    """
    Get the unique set of netCDF groups that should be loaded in.
    """
    grps = ["/".join(k.split("/")[:-1]) for k in keys]
    ugrps = unique(grps)
    vgrps = dict(zip(list(ugrps), [[] for _ in range(len(ugrps))]))
    for key in keys:
        k = "/".join(key.split("/")[:-1])
        var = key.split("/")[-1]
        vgrps[k].append(var)

    return vgrps


def read(floc, vgrps, dc_times=True):
    xds = []
    for grp, vsel in vgrps.items():
        with xr.open_dataset(floc, group=grp, decode_times=dc_times) as ods:
            ods = ods[vsel]
            xds.append(ods)
    dat = xr.merge(xds)
    return dat


def retrieve(floc, ds, dc_times=True, vlst=None):
    if vlst is None:
        vgrps=ds_groups(data_vars[ds])
    else:
        vgrps = ds_groups(vlst)

    dat = read(floc, vgrps, dc_times=dc_times)

    if ds == "oco2" or ds == "oco3":
        dat = dat.rename({"sounding_id": "sounding"})
        dat["sounding"] = dat.sounding.astype(np.int64)
        if "xco2_quality_flag" in dat:
            dat["xco2_quality_flag"] = dat.xco2_quality_flag.astype(np.int32)

    elif ds in ["tomi_co","tomi_no2","tomi_ch4"]:
        # Reshape to 1D arrays
        dat = dat.reset_coords(["latitude", "longitude"])
        dat["scanline"] = dat.scanline.astype(np.int32)
        dat["ground_pixel"] = dat.ground_pixel.astype(np.int32)
        dat = dat.squeeze("time").stack(sounding=["scanline", "ground_pixel"])

        if "corner" in dat.dims:
            if "layer" in dat.dims and "vertices" in dat.dims:
                dat = dat.rename({"vertices": "boundary"})
                dat = dat.transpose("sounding", "corner", "layer", "boundary")

            elif "layer" in dat.dims:
                dat = dat.transpose("sounding", "corner", "layer")
            else:
                dat = dat.transpose("sounding", "corner")
        else:
            if "layer" in dat.dims:
                dat = dat.transpose("sounding", "layer")
            else:
                pass
        
        if ds=='tomi_co':
            
            if 'pressure_levels' in dat.data_vars:
                dat["surface_pressure"]=dat.pressure_levels[:,-1]
                dry_air_column=calc_dry_air_column(dat.surface_pressure, dat.water_total_column, 
                                                dat.surface_altitude, dat.latitude)
                dat["xco"]=dat.carbonmonoxide_total_column_corrected/dry_air_column * 1e9
        
        elif ds=='tomi_no2':
            if "water_slant_column_density" in dat.keys():
                dat["water_total_column"] = (
                    dat.water_slant_column_density / dat.air_mass_factor_total
                )
                del dat["water_slant_column_density"]
                del dat["air_mass_factor_total"]

            if 'surface_pressure' in dat.data_vars:
                dry_air_column=calc_dry_air_column(dat.surface_pressure, dat.water_total_column, 
                                dat.surface_altitude, dat.latitude)
                dat["xno2"]=dat.nitrogendioxide_tropospheric_column/dry_air_column * 1e9
                
    return dat



if __name__ == "__main__":
    co_dir = "/export/data2/scratch/tropomi/collection_03/CO/"
    no2_dir = "/export/data2/scratch/tropomi/collection_03/NO2/"
    oco2_dir = "/export/data/scratch/oco-2/V11r/lite_l2_FP/"
    oco3_dir = "/export/data/scratch/oco-3_v10r/"

    co_floc = co_dir + "S5P_OFFL_L2__CO_____20221231T222149_20230101T000319_27035_03_020400_20230102T125826.nc"
    no2_floc = no2_dir + "S5P_RPRO_L2__NO2____20220316T230526_20220317T004657_22921_03_020400_20221106T130539.nc"
    oco2_floc = oco2_dir + "oco2_LtCO2_200101_B11014Ar_220902231034s.nc4"
    oco3_floc = oco3_dir + "oco3_LtCO2_210903_B10400Br_220318012613s.nc4"

    #a = retrieve(co_floc, "tomi_co")
    a = retrieve(no2_floc, "tomi_no2")
    #a = retrieve(oco2_floc, "oco2")
    #a = retrieve(oco3_floc, "oco3")
    print(a)