import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from shapely import STRtree
from functools import partial
from multiprocessing import Pool
from pandas import date_range, Timestamp
from shapely.geometry import Polygon, Point
from matplotlib.collections import PolyCollection

from plot import plot
from retrieve import retrieve
from utils import get_city_centroid, crop, quality_filter

basic_vars = [
    "PRODUCT/latitude",
    "PRODUCT/longitude",
    "PRODUCT/qa_value",
]

data_fields = {
    "tomi_co": [
        "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/pressure_levels",
        "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_total_column",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude",
        "/PRODUCT/latitude",
        "/PRODUCT/longitude",
        "/PRODUCT/qa_value",
        "/PRODUCT/time_utc",
        "/PRODUCT/carbonmonoxide_total_column_corrected",
        # "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/column_averaging_kernel",
    ],
    "tomi_no2": [
        "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_slant_column_density",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/northward_wind",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/eastward_wind",
        "/PRODUCT/latitude",
        "/PRODUCT/longitude",
        "/PRODUCT/qa_value",
        "/PRODUCT/time_utc",
        "/PRODUCT/nitrogendioxide_tropospheric_column",
        # "/PRODUCT/air_mass_factor_troposphere",
        "/PRODUCT/air_mass_factor_total",
    ],
}

def get_files(dmin, dmax, ds):
    if ds=='tomi_co':
        dir = f"/export/data2/scratch/tropomi/collection_03/CO/"
    elif ds=='tomi_no2':
        dir = f"/export/data2/scratch/tropomi/collection_03/NO2/"
    
    result_list = []

    for date in date_range(dmin, dmax):
        dates = glob.glob(dir+"*__" + date.strftime('%Y%m%d') + "*.nc")
        if len(dates) > 0:
            result_list.append(dates)
    return result_list


def get_other_file(floc, ds):
    dir = floc.rpartition("/")[0]
    just_file = floc.split("/")[-1]
    orbit_start_end = just_file[20:51]

    if ds == "tomi_no2":
        other_dir = dir.replace("/CO", "/NO2")
    elif ds == "tomi_co":
        other_dir = dir.replace("/NO2", "/CO")
    
    other_floc = other_dir + "/*" + orbit_start_end + "*.nc"

    file_check = glob.glob(other_floc)

    if len(file_check) == 1:
        return file_check[0]
    elif len(file_check) > 1:
        raise FileExistsError("Two files found")
    elif len(file_check) == 0:
        #raise FileNotFoundError("No corresponding file found to: " + floc)
        print("No corresponding file found to: " + floc)
        return None

def open_file(floc, ds, bbox):
    dat = retrieve(floc, ds, dc_times=True, vlst=data_fields[ds])
    dat = quality_filter(dat, ds)
    dat = crop(dat, bbox)

    dat = dat.drop_vars(["scanline", "ground_pixel", "sounding"])

    orbit = int(floc.split("/")[-1].split("_")[-4])
    orbit_da = xr.DataArray(np.full(len(dat.sounding), orbit), dims="sounding")
    dat=dat.assign(orbit=orbit_da)

    return dat


def colocate(op1, op2, ds1, ds2):

    if op1.sounding.size == 0 or op2.sounding.size == 0:
        return None

    pgons = [
        Polygon(zip(x, y))
        for x, y in zip(
            op1["longitude_bounds"].values, op1["latitude_bounds"].values
        )
    ]

    points = [
        Point(x, y)
        for x, y in zip(op2["longitude"].values, op2["latitude"].values)
    ]

    tree = STRtree(pgons)
    #res = tree.query(points)
    #res = [o for o in zip(res[0], res[1]) if pgons[o[1]].intersects(points[o[0]])]
    
    #input geometries, tree geometries
    ind2, ind1 = tree.query(points, predicate='within')

    if len(ind1) == 0:
        return None

    if ds1 == "tomi_co" and ds2 == "tomi_no2":
        ds_co, ds_no2 = op1, op2
        ind_co, ind_no2 = ind1, ind2
        
        ds_co = ds_co.rename(
            {
                "latitude_bounds": "latitude_bounds_co",
                "longitude_bounds": "longitude_bounds_co",
            }
        )

        ds_no2 = ds_no2.rename(
            {
                "latitude_bounds": "latitude_bounds_no2",
                "longitude_bounds": "longitude_bounds_no2",
            }
        )
        
        ds_combined = xr.merge(
            [
                ds_no2["xno2"][ind_no2],
                #ds_no2["xh2o"][ind_no2],
                ds_no2["nitrogendioxide_tropospheric_column"][ind_no2],
                ds_no2["orbit"][ind_no2],
                ds_no2["surface_pressure"][ind_no2],
                ds_no2["surface_altitude"][ind_no2],
                ds_no2["water_total_column"][ind_no2],
                ds_no2["time_utc"][ind_no2],
                ds_no2["northward_wind"][ind_no2],
                ds_no2["eastward_wind"][ind_no2],
                ds_no2["latitude_bounds_no2"][ind_no2],
                ds_no2["longitude_bounds_no2"][ind_no2],
                ds_no2["latitude"][ind_no2],
                ds_no2["longitude"][ind_no2],
                ds_co["xco"][ind_co],
                ds_co["carbonmonoxide_total_column_corrected"][ind_co],
                ds_co["latitude_bounds_co"][ind_co],
                ds_co["longitude_bounds_co"][ind_co],
            ],
            compat="override",  # skip comparing and copy attrs from the first dataset to the result
        )

        ds_combined["time_utc"] = ds_combined["time_utc"].astype(str)
        ds_combined = ds_combined.rename({"time": "day"}).expand_dims("day")
    else:
        raise NotImplementedError("Put other dataset combinations here")

    return ds_combined

def combine_days(day_list):
    longest_sounding_dim = max([day.dims["sounding"] for day in day_list])

    day_list_padded = []
    for day in day_list:
        ds_padded = day.pad(
            sounding=(0, longest_sounding_dim - day.dims["sounding"]),
            mode="constant",
            keep_attrs=True,
        )
        day_list_padded.append(ds_padded)

    ds_concat = xr.concat(day_list_padded, dim="day", combine_attrs="drop_conflicts")

    for var in ds_concat.keys():
        # ds[var].encoding['_FillValue'] = None
        if "coordinates" in ds_concat[var].attrs:
            del ds_concat[var].attrs["coordinates"]

    del ds_concat["nitrogendioxide_tropospheric_column"].attrs["ancillary_variables"]
    del ds_concat["carbonmonoxide_total_column_corrected"].attrs["ancillary_variables"]

    ds_concat["orbit"].attrs["long_name"] = "Orbit number. One or two per day"
    
    ds_concat["water_total_column"].attrs["long_name"] = "Water vapor total column density"
    ds_concat["water_total_column"].attrs["units"] = "mol m-2"

    ds_concat["latitude"].attrs["long_name"] = "pixel (no2) center latitude"
    ds_concat["longitude"].attrs["long_name"] = "pixel (no2) center longitude"

    ds_concat["xco"].attrs["standard_name"]="carbonmonoxide_dry_atmosphere_mole_fraction"
    ds_concat["xco"].attrs["long_name"]="Carbon monoxide column-averaged dry air mole fraction"
    ds_concat["xco"].attrs["units"] = "parts_per_billion_(1e-9)"

    ds_concat["xno2"].attrs["standard_name"]="nitrogendioxide_dry_atmosphere_mole_fraction"
    ds_concat["xno2"].attrs["long_name"]="Nitrogen dioxide column-averaged dry air mole fraction"
    ds_concat["xno2"].attrs["units"] = "parts_per_billion_(1e-9)"

    return ds_concat

def process_day(day_list, ds1, ds2, city, bound=1):
    city_centroid = get_city_centroid(city)

    bbox = [city_centroid[0] - bound, city_centroid[1] - bound, 
            city_centroid[0] + bound, city_centroid[1] + bound]

    dat_orbit_list = []
    for orbit_file in day_list:
        dat = retrieve(orbit_file, ds1, dc_times=False, vlst=basic_vars)
        dat = crop(dat, bbox)
        dat = quality_filter(dat, ds1)
        
        if dat.dims['sounding']>10:
            dat1 = open_file(orbit_file, ds1, bbox)
            print(np.nanmax(dat1['xco'].values))
            other_orbit_file = get_other_file(orbit_file, ds2)
            if other_orbit_file is None:
                continue

            dat2 = open_file(other_orbit_file, ds2, bbox)
            if other_orbit_file is not None:
                dat_orbit = colocate(dat1, dat2, ds1, ds2)

                if dat_orbit is not None:
                    dat_orbit_list.append(dat_orbit)

                    if False:
                        temp=dat_orbit.squeeze('day')
                        temp_co=temp.rename({'latitude_bounds_co':'latitude_bounds',
                                             'longitude_bounds_co':'longitude_bounds'})
                        temp_no2=temp.rename({'latitude_bounds_no2':'latitude_bounds',
                                              'longitude_bounds_no2':'longitude_bounds'})
                        plot(orbit_file, ds1, bbox, city)
                        plot(temp_co, ds1, bbox, city)
                        plot(other_orbit_file, ds2, bbox, city)
                        plot(temp_no2, ds2, bbox, city)
                    
    if len(dat_orbit_list) == 2:
        return xr.concat(
            dat_orbit_list, dim="sounding", combine_attrs="drop_conflicts"
        )

    elif len(dat_orbit_list) == 1:
        return dat_orbit_list[0]

    elif len(dat_orbit_list) == 0:
        return None

    else:
        raise ValueError("Something went wrong")


def main():
    ds1, ds2 = "tomi_co", "tomi_no2"
    city="Seattle"

    bound = 2.5
    dmin=Timestamp(2022, 9, 10)
    dmax=Timestamp(2022, 9, 10)



    day_list = get_files(dmin, dmax, ds1)

    print("Processing each day...")
    pool = Pool(20)
    results = pool.map(
        partial(process_day, ds1=ds1, ds2=ds2, 
                city=city, bound=bound), day_list
    )
    list_of_day_ds = [x for x in results if x is not None]

    print("Combining days...")
    ds_result = combine_days(list_of_day_ds)
    
    print("Saving to netcdf4...")
    ds_result.to_netcdf(f"{city}_colocated_{dmin.strftime('%Y%m%d')}_{dmax.strftime('%Y%m%d')}.nc")

if __name__ == "__main__":
    main()