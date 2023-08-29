import matplotlib.pyplot as plt
import table_helper as TH
import model_helper as MH
import fit_helper as FH
#import gammapy
import naima
import numpy as np
from astropy.table import vstack,Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from datasets import StandardOGIPDataset
from gammapy.estimators import FluxPoints
from sherpa.astro.xspec import XSwabs
from sherpa.models import PowLaw1D
from modeling import Fit
from modeling.models import SkyModel, NaimaSpectralModel
from modeling.models.models import SherpaSpectralModel
from datasets import Datasets, FluxPointsDataset, SpectrumDatasetOnOff
from maps import MapAxis
from catalog import SourceCatalogHGPS
from gammapy.utils.regions import (
    make_concentric_annulus_sky_regions
)
run_individual_fit=True # to try a manual fit we can adjust params above and skip full fit which might take an hour
run_joint_fit=True # fit broadband obs using a weighted scheme

# read Lucas test files as tutorial.ipynb at
# https://github.com/luca-giunti/gammapyXray/blob/main/tutorial.ipynb
#filename='/home/sheridan/Documents/gammapyXray-main/XMM_test_files/PN_PWN.grp'
# P0411780301M2S003SRSPEC0001.FTZ
# P0411780301PNS001SRSPEC0001.FTZ
filename='/home/sheridan/Documents/XMM_NEWTON/EPIC_SOURCE_SPECTRUM_GUEST13134731/0411780301/pps/P0411780301M1S002SRSPEC0001.FTZ'
#pn_dataset = SpectrumDatasetOnOff.read(
#        f"/home/sheridan/Documents/XMM_NEWTON/EPIC_SOURCE_SPECTRUM_GUEST13134731/0411780301/pps/P0411780301M1S002SRSPEC0001.FTZ"
#    )
#filename='/home/sheridan/Documents/XMM_NEWTON/EPIC_SOURCE_SPECTRUM_GUEST13134731/0411780301/pps/P0411780301M2S003SRSPEC0001.FTZ'
#filename='/home/sheridan/Documents/XMM_NEWTON/EPIC_SOURCE_SPECTRUM_GUEST13134731/0411780301/pps/P0411780301PNS001SRSPEC0001.FTZ'

pn_dataset=StandardOGIPDataset.read(filename)
pn_dataset.set_min_true_energy(0.5 * u.keV)

#filename='/home/sheridan/Documents/gammapyXray-main/XMM_test_files/MOS1_PWN.grp'
#mos1_dataset=StandardOGIPDataset.read(filename)
#filename='/home/sheridan/Documents/gammapyXray-main/XMM_test_files/MOS2_PWN.grp'
#mos2_dataset=StandardOGIPDataset.read(filename)
#
# ax = pn_dataset.plot_excess()
# ax = pn_dataset.grouped.plot_excess()
# plt.xlim(1,10)
# plt.show()

# HESSS
#HESS Data
# reduction as tutorial at https://docs.gammapy.org/1.0/tutorials/analysis-1d/spectral_analysis.html


from IPython.display import display
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    SkyModel,)
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion

# %matplotlib inline
import matplotlib.pyplot as plt

spectral_model = PowerLawSpectralModel(
    index=2,
    amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
)
model1 = SkyModel(spectral_model=spectral_model, name="PL")


#Load HESS data
from gammapy.visualization import plot_spectrum_datasets_off_regions

datastore = DataStore.from_dir("/home/sheridan/GIT_PROJECTS/TestData/HESS/hess_dl3_dr1/")
obs_ids = [47802,47803,47804,47827,47828,47829]
#datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
#obs_ids = [23523, 23526, 23559, 23592]
observations = datastore.get_observations(obs_ids,required_irf='point-like')
# from simbad 329.7169384374500 -30.2255884577200 https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=pks+2155-304&submit=SIMBAD+search
target_position = SkyCoord(ra=329.7169, dec=-30.2255, unit="deg", frame="icrs")
on_region_radius = Angle("0.1 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

# Exclude nearby TeV sources from http://gamma-sky.net
exclusion_region2 = CircleSkyRegion(
    center=SkyCoord(327.953, -30.429, unit="deg", frame="icrs"),
    radius=0.4 * u.deg,
)

exclusion_region1 = CircleSkyRegion(
    center=SkyCoord(329.817, -28.687, unit="deg", frame="icrs"),
    radius=0.4 * u.deg,
)


skydir = target_position
geom = WcsGeom.create(
    npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
)

exclusion_mask = ~geom.region_mask([exclusion_region1,exclusion_region2])
#exclusion_mask.plot()

# Run data reduction chain

energy_axis = MapAxis.from_energy_bounds(
    0.1, 40, nbin=10, per_decade=True, unit="TeV", name="energy"
)
energy_axis_true = MapAxis.from_energy_bounds(
    0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

plot_datasets=Datasets() # SED and residiuals -> plotting. PL Model (arbritary)
print("a")
print(plot_datasets)

fit_datasets=Datasets() # fit BHJet Model
HESSdatasets = Datasets()
HESS_fit_datasets=Datasets()

for obs_id, observation in zip(obs_ids, observations):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    HESSdatasets.append(dataset_on_off)
    #datasets.append(dataset_on_off)
hess_reduced_dataset=HESSdatasets.stack_reduce(name="hess_reduced") # demonstrates that stack reduced answer no different

fit_datasets.append(hess_reduced_dataset)
HESS_fit_datasets.append(hess_reduced_dataset)
plot_datasets.append(hess_reduced_dataset)
print("b")
print(plot_datasets)

print("c")
print(plot_datasets)

# fancy model?
# fig, ax = plt.subplots(figsize=(8, 6))
# energy_bounds = [0.5e-11, 100] * u.TeV #
# sed_type = "e2dnde"
# source_model="PL"
# energy_edges = MapAxis.from_bounds(1, 15, nbin=5, interp="log", unit="TeV").edges
# flux_points_hess = FluxPointsEstimator(
# energy_edges=energy_edges, source=source_model #, selection_optional=["ul"]
# ).run(hess_reduced_dataset)
# flux_points_hess.plot(ax=ax, sed_type=sed_type, label="HESS")
#
# ax.set_ylim(bottom=1e-50,top=1e-6)
# ax.set_xlim(left=1e-6,right=1e3)
# ax.legend()
#
# plt.show()
#
#



#pn_dataset.name='XMM'
plot_datasets.append(pn_dataset) # add an xmm newton dataset in here
print("d")
print(plot_datasets)

xmm_fit_datasets=Datasets()
fit_datasets.append(pn_dataset)
xmm_fit_datasets.append(pn_dataset)
# evaluate hess reduced dataset here - what points are there





# plt.figure()
# ax = exclusion_mask.plot()
# on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
# plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)
# plt.show()
# read in LAT data from Snakemake workflow
#fermi_datasets = Datasets.read("/home/sheridan/Documents/Fermi_reduction_snakemake/snakemake-workflow-fermi-lat/results/my-config/datasets/my-config-datasets-all.yaml")
# read from front back snake make analysis
fermi_datasets = Datasets.read("/home/sheridan/Documents/Fermi_reduction_snakemake/snakemake-workflow-fermi-lat-reduced/results/my-config/datasets/my-config-datasets-all.yaml")

# diagnostic plot which shows the point source counts hot spot at the centre of the field for varying PSF counts
# for x in range(1,4):
#     counts_image = fermi_datasets[x].counts.sum_over_axes()
#     counts_image.smooth("0.1 deg").plot(stretch="sqrt")
#    plt.show()
#    print(fermi_datasets[x].counts)

selection_region = CircleSkyRegion(
    center=SkyCoord(329.817, -28.687, unit="deg", frame="icrs"),
    radius=1.5 * u.deg,
)
#
# fermi_spectrum_datasets=Datasets()
# fermi_spectrum_datasets.append(fermi_datasets[0].to_spectrum_dataset(selection_region))
# fermi_spectrum_datasets.append(fermi_datasets[1].to_spectrum_dataset(selection_region))
# fermi_spectrum_dataset=fermi_spectrum_datasets.stack_reduce(name="fermi_reduce")

# fermi_spectrum_datasets.append(fermi_datasets[2].to_spectrum_dataset(selection_region))
# fermi_spectrum_datasets.append(fermi_datasets[3].to_spectrum_dataset(selection_region))

plot_datasets.append(fermi_datasets[0].to_spectrum_dataset(on_region=selection_region,name='Fermi'))
print("e")
print(plot_datasets)
fermi_fit_datasets=Datasets()
fit_datasets.append(fermi_datasets[0].to_spectrum_dataset(on_region=selection_region,name='Fermi'))
fermi_fit_datasets.append(fermi_datasets[0].to_spectrum_dataset(on_region=selection_region,name='Fermi'))
#datasets.append(fermi_datasets[1].to_spectrum_dataset(selection_region))



#datasets.append(fermi_datasets[2].to_spectrum_dataset(selection_region))
#datasets.append(fermi_datasets[3].to_spectrum_dataset(selection_region))



#fermi_dataset=fermi_datasets.stack_reduce(name="fermi_reduce")
#fermi_spectrum_dataset=fermi_spectrum_datasets.stack_reduce(name="fermi_reduce") # operands could not be broadcast together with shapes (31,30,1,1) (30,1,1,1)
#datasets.append(fermi_spectrum_dataset) # co-allesce 4 PSF classes

#info_table = datasets.info_table(cumulative=True) # not supported for mixed dataset (OGIP and spectrumdatasetonoff in this case)
# display(info_table)
#
# fig, (ax_excess, ax_sqrt_ts) = plt.subplots(figsize=(10, 4), ncols=2, nrows=1)
# ax_excess.plot(
#     info_table["livetime"].to("h"),
#     info_table["excess"],
#     marker="o",
#     ls="none",
# )
#
# ax_excess.set_title("Excess")
# ax_excess.set_xlabel("Livetime [h]")
# ax_excess.set_ylabel("Excess events")
#
# ax_sqrt_ts.plot(
#     info_table["livetime"].to("h"),
#     info_table["sqrt_ts"],
#     marker="o",
#     ls="none",
# )
#
# ax_sqrt_ts.set_title("Sqrt(TS)")
# ax_sqrt_ts.set_xlabel("Livetime [h]")
# ax_sqrt_ts.set_ylabel("Sqrt(TS)")
#
# plt.show()

# PKS 2155 MIcrowave mid flux value from digitizer of 1810.11341
#
e_ref=[3e-5,3.2e-5]
e_min=[2.9e-5,3.2e-5]
e_max=[3.2e-5,3.3e-5]
e2dnde=[2.1e-14,1.8e-14]
e2dnde_err=[0.1e-14,0.1e-14] # arbritary - there has to be an error
microwave_table = Table([e_ref, e_min, e_max,e2dnde, e2dnde_err], names=('e_ref', 'e_min','e_max','e2dnde', 'e2dnde_err'), units=('eV','eV','eV','erg cm-2 s-1','erg cm-2 s-1'))

microwave_table['e_ref']=microwave_table['e_ref'].to('keV')
microwave_table['e_min']=microwave_table['e_min'].to('keV')
microwave_table['e_max']=microwave_table['e_max'].to('keV')
microwave_table['e2dnde'] = microwave_table['e2dnde'].to(u.TeV/(u.cm*u.cm*u.s))
microwave_table['e2dnde_err'] = microwave_table['e2dnde_err'].to(u.TeV/(u.cm*u.cm*u.s))
microwave_table.meta["SED_TYPE"] = "e2dnde"

flux_points_microwave = FluxPoints.from_table(microwave_table)
microwave_dataset = FluxPointsDataset(model1, flux_points_microwave)
plot_datasets.append(microwave_dataset)
print("f")
print(plot_datasets)
fit_datasets.append(microwave_dataset)

microwave_fit_datasets=Datasets()
microwave_fit_datasets.append(microwave_dataset)


# PKS 2155-304 Optical from arxiv 0903.2924
e_ref=[1.9,2.3,2.8]
e_min=[1.89,2.29,2.79]
e_max=[1.91,2.31,2.81]
e2dnde=[1.28e-10,1.6e-10,1.61e-10]
e2dnde_err=[1e-11,1e-11,1e-11] # arbritary - there has to be an error
optical_table = Table([e_ref, e_min, e_max,e2dnde, e2dnde_err], names=('e_ref', 'e_min','e_max','e2dnde', 'e2dnde_err'), units=('eV','eV','eV','erg cm-2 s-1','erg cm-2 s-1'))

optical_table['e_ref']=optical_table['e_ref'].to('keV')
optical_table['e_min']=optical_table['e_min'].to('keV')
optical_table['e_max']=optical_table['e_max'].to('keV')
optical_table['e2dnde'] = optical_table['e2dnde'].to(u.TeV/(u.cm*u.cm*u.s))
optical_table['e2dnde_err'] = optical_table['e2dnde_err'].to(u.TeV/(u.cm*u.cm*u.s))

optical_table.meta["SED_TYPE"] = "e2dnde"
flux_points_optical = FluxPoints.from_table(optical_table)
optical_dataset = FluxPointsDataset(model1, flux_points_optical)
plot_datasets.append(optical_dataset)
print("f")
print(plot_datasets)
fit_datasets.append(optical_dataset)

optical_fit_datasets=Datasets()
optical_fit_datasets.append(optical_dataset)



#reduced_datasets=datasets.stack_reduce(name="reduced") # demonstrates that stack reduced answer no different
#reduced_datasets.models=[model]

# fit_joint = Fit()
# result_joint = fit_joint.run(datasets=datasets) # overflow float infinity to integer
# print(datasets)
#
# # we make a copy here to compare it later
# model_best_joint = model.copy()
#
# print(result_joint)
# display(result_joint.models.to_parameters_table())

# plt.figure()
# ax_spectrum, ax_residuals = datasets[0].plot_fit()
# ax_spectrum.set_ylim(1e-10, 40)
# ax_spectrum.set_xlim(1e-10, 40)
# datasets[0].plot_masks(ax=ax_spectrum)
# plt.show()

from gammapy.modeling.models import Models, BHJetSpectralModel, SkyModel


# test neg stat on Fermi fit
# modelBHJet=MH.fermi_neg_stat()
# BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
# models = Models([BHJetmodel])

if run_individual_fit:
    print ("Running individual fits to determine stat normalisation")
    print ("-------------------------------------------------------")
    #modelBHJet=MH.run_8()
    #modelBHJet=MH.run_9_low_hess_stats()
    #modelBHJet=MH.run_9_low_hess_stats()
    #modelBHJet=MH.run_8()

    # Fermi first - to see if 1e36 issue surfaces
    modelBHJet=MH.Matteo_1810_11341()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])
    fermi_results=FH.get_fit(fermi_fit_datasets,models)



    # individual fits to obtain a normalisation so that one
    # dataset with a very high fit statistic does not dominate?
    # left out optical as GSL croaks when considering optical alone
    # try to fit microwave and see if same deal - microwave is fine ! - reinstate optical
    modelBHJet=MH.Matteo_1810_11341()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])
    optical_results=FH.get_fit(optical_fit_datasets,models)

    modelBHJet=MH.Matteo_1810_11341()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])
    microwave_results=FH.get_fit(microwave_fit_datasets,models)


    modelBHJet=MH.Matteo_1810_11341()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])
    HESS_results=FH.get_fit(HESS_fit_datasets,models)
    #
    modelBHJet=MH.Matteo_1810_11341()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])
    xmm_results=FH.get_fit(xmm_fit_datasets,models)

    all_stats={}

    all_stats[HESS_fit_datasets[0].name]=HESS_results.total_stat
    all_stats[fermi_fit_datasets[0].name]=fermi_results.total_stat
    all_stats[xmm_fit_datasets[0].name]=xmm_results.total_stat
    all_stats[optical_fit_datasets[0].name]=optical_results.total_stat
    all_stats[microwave_fit_datasets[0].name]=microwave_results.total_stat

    sorted_all_stats=dict(sorted(all_stats.items(), key=lambda item: item[1]))
    sorted_all_stats_list=list(sorted_all_stats.values())
    min_stat=sorted_all_stats_list[0]
    print("----------")
    print("min_stat "+ str(min_stat))
    #scaling_factors={}
    for k,v in sorted_all_stats.items():
        fit_datasets[k].joint_fit_weight=v/min_stat # joint fit weight is 1 by default on initialised dataset
        print ( str(fit_datasets[k].name) + " " + str(fit_datasets[k].tag) + "Joint fit weight " + str(fit_datasets[k].joint_fit_weight))
    print("----------")

if run_joint_fit:

    print ("Running joint fit")
    print ("-----------------")

    if not run_individual_fit: #normalising weights
        fit_datasets["hess_reduced"].joint_fit_weight=1 # enhance HESS weighting
        fit_datasets["Fermi"].joint_fit_weight = 6 # val from individuak fit
        for k in fit_datasets:
            if k.tag=="StandardOGIPDataset":
                k.joint_fit_weight=21384 # val from individual fat

    modelBHJet=MH.Matteo_1810_11341()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])

    print("JOINT FIT DATASET")
    print(fit_datasets)

    results_joint=FH.get_fit(fit_datasets,models)

    # old fit code before helper
    # fit_datasets.models = models
    # fit_joint = Fit()
    # results_joint = fit_joint.run(datasets=fit_datasets)
    # print(results_joint)


plot_datasets.models = [model1] #pl model


fig, ax = plt.subplots(figsize=(8, 6))
energy_bounds = [0.5e-11, 10] * u.TeV #
sed_type = "e2dnde"
source_model="PL"
print("g_SJL_PLOT")
print(plot_datasets)
print("--------")


# reset models - simple append doesn't pick up model


energy_edges = MapAxis.from_bounds(0.5, 10, nbin=5, interp="log", unit="keV").edges

x=1
flux_points_xmm_newton = FluxPointsEstimator(
    energy_edges=energy_edges, source=source_model
    ).run([plot_datasets[x].grouped])
#
flux_points_xmm_newton.plot(ax=ax, sed_type=sed_type, label="XMM NEWTON")
table_xmm_newton=flux_points_xmm_newton.to_table(sed_type="e2dnde",format="gadf-sed",formatted=True)
print ("XMM e2dnde_err unit " + str(table_xmm_newton['e2dnde_err'].unit))
print ("XMM Table")
print(table_xmm_newton)

energy_edges = MapAxis.from_bounds(1, 15, nbin=5, interp="log", unit="TeV").edges
hess_tables=[]

x=0
flux_points_hess = FluxPointsEstimator(
energy_edges=energy_edges, source=source_model #, selection_optional=["ul"]
).run([plot_datasets[x]])
flux_points_hess.plot(ax=ax, sed_type=sed_type, label="HESS")
table_hess = flux_points_hess.to_table(sed_type="e2dnde", format="gadf-sed", formatted=True)
print ("HESS Table")
print(table_hess)
hess_tables.append(table_hess)
table_hess_all=vstack(hess_tables)
print("HESS Table ALLLLLL")
print(table_hess_all)

energy_edges = MapAxis.from_bounds(100, 100000, nbin=12, interp="log", unit="MeV").edges
#
fermi_tables=[]
x=2

flux_points_fermi = FluxPointsEstimator(
energy_edges=energy_edges, source=source_model #, selection_optional=["ul"]
).run([plot_datasets[x]])
table_fermi = flux_points_fermi.to_table(sed_type="e2dnde", format="gadf-sed", formatted=True)
flux_points_fermi.plot(ax=ax, sed_type=sed_type, label="Fermi")
fermi_tables.append(table_fermi)

flux_points_optical.plot(ax=ax, sed_type=sed_type, label="Optical") # direct plot without estimator
flux_points_microwave.plot(ax=ax, sed_type=sed_type, label="Microwave") # direct plot without estimator

print("Fermi consituent Table")
print(table_fermi['e_ref','e2dnde'])
table_fermi_all=vstack(fermi_tables)


# does vstack convert units- no - we must convert manually - rationalise this to a function when working
table_xmm_newton['e_ref']=table_xmm_newton['e_ref'].to('keV')
table_hess_all['e_ref']=table_hess_all['e_ref'].to('keV')
table_fermi_all['e_ref']=table_fermi_all['e_ref'].to('keV')

table_xmm_newton['e_min']=table_xmm_newton['e_min'].to('keV')
table_hess_all['e_min']=table_hess_all['e_min'].to('keV')
table_fermi_all['e_min']=table_fermi_all['e_min'].to('keV')

table_xmm_newton['e_max']=table_xmm_newton['e_max'].to('keV')
table_hess_all['e_max']=table_hess_all['e_max'].to('keV')
table_fermi_all['e_max']=table_fermi_all['e_max'].to('keV')


# astropy can convert fine
table_xmm_newton['e2dnde'] = table_xmm_newton['e2dnde'].to(u.TeV/(u.cm*u.cm*u.s))
table_xmm_newton['e2dnde_err'] = table_xmm_newton['e2dnde_err'].to(u.TeV/(u.cm*u.cm*u.s))
table_fermi_all['e2dnde']=table_fermi_all['e2dnde'].to(u.TeV/(u.cm*u.cm*u.s))
table_fermi_all['e2dnde_err']=table_fermi_all['e2dnde_err'].to(u.TeV/(u.cm*u.cm*u.s))

# removing ULs still doesn't fix spurious 1 MeV flux in residuals plot so this may be
# be unecessary
table_xmm_newton=TH.del_ULs(table_xmm_newton)
table_fermi_all=TH.del_ULs(table_fermi_all)
table_hess_all=TH.del_ULs(table_hess_all)
table_hess_all=TH.del_fluxes_below_limit(table_hess_all,1e-13) # see if this expands range of residual plot - omits low points but still plots model at those eneergies (as exxpected)
# stack in sensible instrument order
table_all=vstack([microwave_table,optical_table,table_xmm_newton,table_fermi_all,table_hess_all], join_type='outer')

del table_all['ts']
del table_all['sqrt_ts']
del table_all['npred']
del table_all['npred_excess']
del table_all['stat']
del table_all['is_ul']
del table_all['counts']
del table_all['success']
ax.set_ylim(bottom=1e-14,top=1e-6)
ax.set_xlim(left=1e-9,right=1e10)
ax.legend()

plt.show()

all_flux_points=FluxPoints.from_table(table_all,sed_type='e2dnde')

# compare flux points read and table all
print(table_all)

for row in table_all:
    # assume - flux points must be sorted in same order as table as read from there
    # correct these from the table
    all_flux_points.energy_ref[row.index]=row['e_ref']*table_all['e_ref'].unit
    all_flux_points.energy_min[row.index] = row['e_min'] * table_all['e_min'].unit
    all_flux_points.energy_max[row.index] = row['e_max'] * table_all['e_max'].unit

if not run_joint_fit:
    modelBHJet=MH.joint_fit_model_norm_weight()
    BHJetmodel = SkyModel(spectral_model=modelBHJet, name="BHJet")
    models = Models([BHJetmodel])

dataset = FluxPointsDataset(BHJetmodel, all_flux_points)
#configuring optional parameters
kwargs_spectrum = {"kwargs_model": {"color":"red", "ls":"--"}, "kwargs_fp":{"color":"green", "marker":"o"}}  # noqa: E501
kwargs_residuals = {"color": "blue", "markersize":4, "marker":'s', }
dataset.plot_fit(kwargs_residuals=kwargs_residuals, kwargs_spectrum=kwargs_spectrum)
plt.show()