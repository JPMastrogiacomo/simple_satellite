import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection

from utils import get_city_shape, get_city_bbox, crop, quality_filter
from retrieve import retrieve

def add_features(ax):
    countries = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_countries",
        scale="10m",
        facecolor="none",
        zorder=11,
    )

    stprov = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
        zorder=11,
    )

    lakes = cfeature.NaturalEarthFeature(
        category="physical", name="lakes", scale="10m", facecolor="none", zorder=11
    )

    ax.coastlines(resolution="10m", linewidth=2.0, zorder=10)
    ax.add_feature(countries, edgecolor="k", linewidth=1.0)
    ax.add_feature(stprov, edgecolor="k", linewidth=0.5)
    ax.add_feature(lakes, edgecolor="k", linewidth=0.75)

def add_city(ax, city):
    pcoords = get_city_shape(city)

    cpoly = Polygon(
        pcoords,
        edgecolor="yellow",
        facecolor=np.array([0, 0, 0, 0]),
        zorder=100,
        linewidth=0.75,
    )
    ax.add_patch(cpoly)

def plot(file, ds, bbox=None, city=None, save_floc=None):
    if bbox is None:
        bbox=get_city_bbox(city)

    latmn, lonmn, latmx, lonmx = bbox 

    if isinstance(file,str):
        dat = retrieve(file, ds)
        dat = crop(dat, bbox)
        dat = quality_filter(dat, ds)
    else:
        dat=file

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    add_features(ax)

    if city is not None:
        add_city(ax, city)
    

    if ds == "oco2" or ds == "oco3":
        val = dat["xco2"].values
        vlat = dat["vertex_latitude"].values
        vlon = dat["vertex_longitude"].values
        cbar_label = "XCO2 [ppm]"
        cmap = 'plasma'

    elif ds == "tomi_co":
        val = dat["xco"].values
        vlat = dat["latitude_bounds"].values
        vlon = dat["longitude_bounds"].values
        cbar_label = "XCO [ppb]"
        cmap='viridis'
    
    elif ds == "tomi_no2":
        val = dat["xno2"].values
        vlat = dat["latitude_bounds"].values
        vlon = dat["longitude_bounds"].values
        cbar_label = "XNO2 [ppb]"
        cmap='plasma'

    val_min = max(0, val.mean() - 2 * val.std())
    val_max = val.mean() + 2 * val.std()

    verts = np.append(vlon[..., np.newaxis], vlat[..., np.newaxis], axis=-1)
    
    data_coll = PolyCollection(verts, array=val, edgecolor="none", cmap=cmap, zorder=8)
    data_coll.set_clim(vmin=val_min, vmax=val_max)

    ax.add_collection(data_coll)

    cbar = fig.colorbar(data_coll, ax=ax, orientation="vertical", aspect=20, pad=0.05)
    cbar.set_label(cbar_label)
    cbar.set_ticks(np.linspace(val_min, val_max, 5))
    cbar.ax.set_yticklabels(
        ["{:.1f}".format(xk) for xk in np.linspace(val_min, val_max, 5)]
    )

    plnmn = lonmn - 0.1
    plnmx = lonmx + 0.1
    pltmn = latmn - 0.1
    pltmx = latmx + 0.1
    
    xlim = (plnmn, plnmx)
    ylim = (pltmn, pltmx)

    xt = np.int32(np.arange(plnmn, plnmx , 1))
    yt = np.int32(np.arange(pltmn, pltmx , 1))

    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels(["{0}$^{1}$".format(xti, "\circ") for xti in xt])
    ax.set_yticklabels(["{0}$^{1}$".format(yti, "\circ") for yti in yt])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.setp(ax.spines["left"], linewidth=2.0)
    plt.setp(ax.spines["right"], zorder=1000)
    plt.setp(ax.spines["top"], zorder=1000)
    plt.setp(ax.spines["bottom"], zorder=1000)

    ax.grid(True, zorder=110)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if save_floc is not None:
        plt.savefig(save_floc)


if __name__ == "__main__":
    oco2_floc = "/export/data/scratch/oco-2/V11r/lite_l2_FP/oco2_LtCO2_190104_B11014Ar_221017170847s.nc4"
    #oco3_floc = 
    co_floc = "/export/data2/scratch/tropomi/collection_03/CO/S5P_RPRO_L2__CO_____20200701T192949_20200701T211118_14080_03_020400_20220828T222054.nc"
    no2_floc = "/export/data2/scratch/tropomi/collection_03/NO2/S5P_RPRO_L2__NO2____20200701T192949_20200701T211118_14080_03_020400_20221104T190024.nc"
    
    #bbox=[43, -80, 44, -79] #Toronto
    #bbox=[33, -112.5, 34, -111.5] #Phoenix
    bbox=[44, -125, 50, -120]

    #plot(oco2_floc, 'oco2', bbox)
    #plot(oco3_floc, 'oco3', bbox)
    #plot(co_floc, "tomi_co", bbox, city="Phoenix-Mesa")
    #plot(no2_floc, "tomi_no2", bbox, city="Phoenix-Mesa")
    plot(co_floc, "tomi_co", city="Seattle")
