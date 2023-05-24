import os
import csv
import numpy as np
from pandas import Timestamp, Timedelta, to_datetime

from utils import get_city_bbox, quality_filter
from retrieve import retrieve

dir_dict = {
    "tomi_co": "/export/data2/scratch/tropomi/collection_03/CO/",
    "oco2": "/export/data/scratch/oco-2/V11r/lite_l2_FP/",
    "oco3": "/export/data/scratch/oco-3_v10r/",
}

# Variables needed to search OCO-2/3
_ocox_vlst = [
    "latitude",
    "longitude",
    "time",
    "xco2_quality_flag",
]

# Variables needed to search TROPOMI
_tomi_vlst = [
    "PRODUCT/latitude",
    "PRODUCT/longitude",
    "PRODUCT/delta_time",
    "PRODUCT/qa_value",
]


def get_co2_overpass_time(clat, clon, floc, bbox, pixel_min=5, qual_flt=True):
    """
    Search a single day of OCO-2 data
    for overpasses of a city. Returns
    relevant information should there
    be an overpass.
    """
    fdir = os.path.dirname(floc)
    if fdir == os.path.dirname(dir_dict["oco2"]):
        ds = "oco2"
    elif fdir == os.path.dirname(dir_dict["oco3"]):
        ds = "oco3"

    # load in OCO-2/OCO-3 data for the day in question
    dat = retrieve(floc, ds, dc_times=True, vlst=_ocox_vlst)

    latmn, lonmn, latmx, lonmx = bbox

    lat_sel = (dat.latitude > latmn) * (dat.latitude < latmx)
    lon_sel = (dat.longitude > lonmn) * (dat.longitude < lonmx)
    sel = lat_sel * lon_sel
    dat = dat.where(sel, drop=True)

    if qual_flt:
        dat = quality_filter(dat, ds)

    time = dat.time.dt.floor("us").values

    file = floc.split("/")[-1]

    # soundings within bounding box
    if time.size >= pixel_min:

        def haversine(lat1, lon1, lat2, lon2):
            lat1_r = np.radians(lat1)
            lat2_r = np.radians(lat2)

            dlat_r = np.radians(lat2 - lat1)
            dlon_r = np.radians(lon2 - lon1)

            # calculate distance as per the reference
            a = np.power(np.sin(dlat_r / 2.0), 2.0) + np.cos(lat1_r) * np.cos(
                lat2_r
            ) * np.power(np.sin(dlon_r / 2.0), 2.0)
            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return 6371.0 * c

        dist = haversine(clat, clon, latmx, lonmx)

        # find the time of closest approach to the city
        closest_index = np.argmin(dist)
        closest_time = Timestamp(time[closest_index])

        overpass_time = Timestamp(closest_time)
        return overpass_time

    else:
        print(f"Less than {pixel_min} soundings in bounding box for: {file}")
        return None


def get_ocox_files(ds, date):
    """
    Get the file location for an
    OCO-2/3 lite file for a specific
    date.

    args:
        ds : str identifier for which
        satellite will be used. Either
        set to 'oco2' or 'oco3'

        date : pandas Timestamp of the day for which a file should be retrieved.
    """
    dstr = date.strftime("%y%m%d")
    flist = os.listdir(dir_dict[ds])

    # search for file with equal date string
    for fname in flist:
        if fname.split(".")[-1] == "nc4":
            if fname.split("_")[2] == dstr:
                fpath = os.path.join(dir_dict[ds], fname)
                return fpath
    else:
        return None


def get_tropomi_files(date):
    dfiles = []
    tropomi_files = os.listdir(dir_dict["tomi_co"])
    for f in tropomi_files:
        ext = f.split(".")[-1]
        if ext == "nc":
            fsplit = f.split("_")
            dfmt = "%Y%m%dT%H%M%S"
            dstart = to_datetime(fsplit[-6], format=dfmt).floor("d")
            dend = to_datetime(fsplit[-5], format=dfmt).floor("d")

            if (dstart == date) or (dend == date):
                dfiles.append(os.path.join(dir_dict["tomi_co"], f))

    return dfiles


def search(ds, dmin, dmax, bbox=None, city=None, pixel_min=10, qual_flt=True, save_floc=None):
    if bbox is None:
        bbox=get_city_bbox(city)
    latmn, lonmn, latmx, lonmx = bbox

    files = []

    date = dmin  # Search each day in the range
    while date <= dmax:
        print(date)
        if ds == "oco2" or ds == "oco3":
            f = get_ocox_files(ds, date)
            flocs = [] if f is None else [f]
            vlst = _ocox_vlst

        elif ds == "tomi_co":
            flocs = get_tropomi_files(date)
            vlst = _tomi_vlst

        for floc in flocs:
            dat = retrieve(floc, ds, dc_times=True, vlst=vlst)

            lat_sel = (dat.latitude > latmn) * (dat.latitude < latmx)
            lon_sel = (dat.longitude > lonmn) * (dat.longitude < lonmx)
            sel = lat_sel * lon_sel
            dat = dat.where(sel, drop=True)

            if qual_flt:
                dat = quality_filter(dat, ds)

            lat = dat.latitude.values

            if lat.size >= pixel_min:
                files.append(floc)

        date += Timedelta(days=1)

    if len(files) == 0:
        print("No files found for this date range")
        return []

    if save_floc:
        with open(save_floc, "w", newline="") as f:
            writer = csv.writer(f)
            for file in files:
                writer.writerow([file])

    return files


if __name__ == "__main__":
    dmin = Timestamp(2022, 9, 10)
    dmax = Timestamp(2022, 9, 10)
    #city="Toronto"
    #bbox=[43.5, -80, 44, -79]
    city='Seattle'
    #bbox=[33, -112.5, 34, -111.5]
    product='tomi_co'


    if product=='oco2':
        save_floc = f"{city}_oco2_{dmin.strftime('%Y%m%d')}_{dmax.strftime('%Y%m%d')}.csv"
    elif product=='oco3':
        save_floc = f"{city}_oco3_{dmin.strftime('%Y%m%d')}_{dmax.strftime('%Y%m%d')}.csv"
    elif product=='tomi_co':
        save_floc = f"{city}_co_{dmin.strftime('%Y%m%d')}_{dmax.strftime('%Y%m%d')}.csv"
    
    #search(product, dmin, dmax, bbox=bbox, save_floc=save_floc, pixel_min=25)
    print(search(product, dmin, dmax, city=city, pixel_min=1))

    #file = "/export/data/scratch/oco-2/V11r/lite_l2_FP/oco2_LtCO2_200101_B11014Ar_220902231034s.nc4"
    #time = get_co2_overpass_time(43.7, -79.4, file, bbox, pixel_min=10)
    #print(time)
