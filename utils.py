import csv
import shapefile
import numpy as np
from shapely.geometry.polygon import Polygon


def calculate_g_air(altitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Calculate pressure-weighted gravity for given surface altitudes and latitudes."""

    #Add one scale height
    altmat = altitude+8434.50969398
    latmat = latitude

    gm = 3.9862216e14  # Gravitational constant times Earth's Mass (m3/s2) in GFIT
    omega = 7.292116e-05  # Earth's angular rotational velocity (radians/s) from GFIT
    con = 0.006738  # (a/b)**2-1 where a & b are equatorial & polar radii from GFIT
    shc = 1.6235e-03  # 2nd harmonic coefficient of Earth's gravity field from GFIT
    eqrad = 6378178  # Equatorial Radius (m) from GFIT

    gclat = np.arctan(np.tan(latmat * np.pi / 180) / (1 + con))

    radius = altmat + eqrad / np.sqrt(1 + con * np.sin(gclat) ** 2)
    ff = (radius / eqrad) ** 2
    hh = radius * omega**2
    ge = gm / (eqrad**2)  # = gravity at Re

    g = (
        ge * (1 - shc * (3 * np.sin(gclat) ** 2 - 1) / ff) / ff
        - hh * np.cos(gclat) ** 2
    ) * (1 + 0.5 * (np.sin(gclat) * np.cos(gclat) * (hh / ge + 2 * shc / ff**2)) ** 2)

    return g


def calc_dry_air_column(psurf, h2o_column, altsurf, lat):
    _Mair = 0.0289644 # kg mol^-1
    _Mh2o = 0.01801528 # kg mol^-1
    
    g_air=calculate_g_air(altsurf, lat)
    #g_air=9.81
    return psurf/(g_air*_Mair) - h2o_column*(_Mh2o/_Mair)


def crop(ds, bbox):
    latmn, lonmn, latmx, lonmx = bbox

    lat_sel = (ds.latitude >= latmn) & (ds.latitude <= latmx)
    lon_sel = (ds.longitude >= lonmn) & (ds.longitude <= lonmx)

    sel = lat_sel & lon_sel
    ds = ds.where(sel, drop=True)

    return ds


def quality_filter(dat, ds):
    if ds == "oco2" or ds == "oco3":
        dat = dat.where(dat.xco2_quality_flag == 0, drop=True)
    elif ds == "tomi_co":
        dat = dat.where(dat.qa_value > 0.5, drop=True)
    elif ds == "tomi_no2":
        dat = dat.where(dat.qa_value > 0.75, drop=True)
    else:
        raise NotImplementedError("Quality filter not implemented for " + ds)

    dat = dat.where(dat['latitude'].notnull(), drop=True)
    
    return dat


def get_city_shape(city_name, out="coords"):
    csv_floc = "cities.csv"
    with open(csv_floc, "r") as csv_in:
        rows = csv.reader(csv_in)
        for k, row in enumerate(rows):
            if k == 0:
                attrs = [a.lower() for a in row]
                name_ind = attrs.index("name")
            else:
                if row[name_ind] == city_name:
                    uc_nm_mn = row[8]
                    break
        
    with open("city_shape_index_all.csv") as csv_in:
        rows = csv.reader(csv_in)

        for row in rows:
            name, ind = row
            if name == uc_nm_mn:
                sel = int(ind)
                break

    shpf = shapefile.Reader("GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0.shp")
    try:
        coords = np.array(shpf.shape(sel).points)
    except UnboundLocalError:
        if out == "coords":
            return np.array([])
        elif out == "poly":
            return None

    start = coords[0][np.newaxis, :]
    diff = coords[1:] - start
    closed = list(diff.sum(axis=1)).index(0.0)
    coords_closed = coords[: closed + 2, :]
    if out == "coords":
        return coords_closed
    elif out == "poly":
        return Polygon(coords_closed)


def get_city_centroid(city_name):
    csv_floc = "cities.csv"
    with open(csv_floc, "r") as csv_in:
        rows = csv.reader(csv_in)
        for k, row in enumerate(rows):
            if k == 0:
                attrs = [a.lower() for a in row]
                name_ind = attrs.index("name")
            else:
                if row[name_ind] == city_name:
                    centroid = [float(row[5]), float(row[6])]
                    return centroid

def get_city_bbox(city_name):
    csv_floc = "cities.csv"
    with open(csv_floc, "r") as csv_in:
        rows = csv.reader(csv_in)
        for k, row in enumerate(rows):
            if k == 0:
                attrs = [a.lower() for a in row]
                name_ind = attrs.index("name")
            else:
                if row[name_ind] == city_name:
                    bbox = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                    return bbox


if __name__=='__main__':
    #print(calculate_g_air(0,45))

    #print(get_city_centroid('Toronto'))
    print(get_city_shape('Toronto'))
