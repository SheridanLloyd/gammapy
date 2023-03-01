
from pathlib import Path
from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.datasets import Datasets, FluxPointsDataset, SpectrumDatasetOnOff, SpectrumDataset, OGIPDatasetReader
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import Models, create_crab_spectral_model
from gammapy.utils.scripts import make_path


dataset_all=Datasets()


Fermi_dataset_path = make_path("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml")
Fermi_model_path = make_path("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml")
Naima_Fermi_model_path=make_path("$GAMMAPY_DATA/fermi-3fhl-crab/fermi_naima.yaml")

filename="/home/sheridan/Documents/gammapy-master/datasets/gammapy-datasets/joint-crab/spectra/fermi/pha_obs0.fits"
fermi_reader=OGIPDatasetReader(filename)
dataset_fermi=fermi_reader.read()
print("SJL-Loaded OGIP dataset_fermi")
#dataset_fermi.name="Fermi-LAT"
print(dataset_fermi)


#dataset_fermi = Datasets.read(filename=Fermi_dataset_path)
Fermi_model = Models.read(Fermi_model_path)
#Naima_Fermi_model=Models.read(Naima_Fermi_model_path)

models=Models.read(Fermi_model_path)

#datasets=Datasets.read(filename=Fermi_dataset_path)
datasets=Datasets()
datasets.append(dataset_fermi)

datasets_hess = Datasets()
#for obs_id in [23523, 23526]:
#    dataset = SpectrumDatasetOnOff.read(
#        f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
#    )
#    datasets_hess.append(dataset)

for obs_id in [23523, 23526, 23559,23592]:
    dataset = SpectrumDatasetOnOff.read(
        f"/home/sheridan/Documents/gammapy-master/datasets/gammapy-datasets/hess-dl3-dr1/sjl_1D_datasets/spectrum_analysis/obs_{obs_id}.fits.gz"
    )
    datasets_hess.append(dataset)

dataset_hess = datasets_hess.stack_reduce(name="HESS")

datasets.append(dataset_hess)
dataset_all.append(dataset_hess)

print(dataset_all)

filename = "$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits"
flux_points_hawc = FluxPoints.read(
    filename, reference_model=create_crab_spectral_model("meyer")
)

dataset_hawc = FluxPointsDataset(data=flux_points_hawc, name="HAWC")

datasets.append(dataset_hawc)
dataset_all.append(dataset_hawc)
#datasets.append(dataset_fermi)
print ("SJL-Dataset fermi")
print(dataset_fermi)

path = Path("crab-3datasets")
path.mkdir(exist_ok=True)

filename = path / "crab_10GeV_100TeV_datasets.yaml"

datasets.write(filename, overwrite=True)

datasets = Datasets.read(filename)
datasets.models = models

print("SJL-Models and Datasets for LP Fit")
print(models)
print(datasets)
# Fit of LP Model
fit_joint = Fit()
results_joint = fit_joint.run(datasets=datasets)
print ("LP models and model fit")
print(models)
print(results_joint)

crab_spec = datasets[0].models["Crab Nebula"].spectral_model

#fermi_bkg_model=datasets[0].models["Fermi-LAT-bkg"]

print ("SJL- Crab spec Model")
print ("--------------------")
print(crab_spec)

# compute Fermi-LAT and HESS flux points
energy_edges = MapAxis.from_energy_bounds("10 GeV", "2 TeV", nbin=5).edges

flux_points_fermi = FluxPointsEstimator(
    energy_edges=energy_edges,
    source="Crab Nebula",
).run([datasets[0]])

energy_edges = MapAxis.from_bounds(1, 15, nbin=6, interp="log", unit="TeV").edges

flux_points_hess = FluxPointsEstimator(
    energy_edges=energy_edges, source="Crab Nebula", selection_optional=["ul"]
).run([datasets["HESS"]])

# display spectrum and flux points
fig, ax = plt.subplots(figsize=(8, 6))

energy_bounds = [0.01, 300] * u.TeV
sed_type = "dnde"
#sed_type = "e2dnde"

crab_spec.plot(ax=ax, energy_bounds=energy_bounds, sed_type=sed_type, label="LP Model")
crab_spec.plot_error(ax=ax, energy_bounds=energy_bounds, sed_type=sed_type)

flux_points_fermi.plot(ax=ax, sed_type=sed_type, label="Fermi-LAT")
flux_points_hess.plot(ax=ax, sed_type=sed_type, label="HESS")
flux_points_hawc.plot(ax=ax, sed_type=sed_type, label="HAWC")

ax.set_xlim(energy_bounds)
ax.legend()

# Naima Fit

from astropy import units as u
import matplotlib.pyplot as plt
import naima
from gammapy.modeling.models import Models, NaimaSpectralModel, SkyModel

particle_distribution = naima.models.ExponentialCutoffPowerLaw(
    1e30 / u.eV, 10 * u.TeV, 2.0, 30 * u.TeV
)
radiative_model = naima.radiative.InverseCompton(
    particle_distribution,
    seed_photon_fields=["CMB", ["FIR", 26.5 * u.K, 0.415 * u.eV / u.cm**3]],
    Eemin=100 * u.GeV,
)


# Plot the separate contributions from each seed photon field
for seed, ls in zip(["CMB", "FIR"], ["-", "--"]):
    modelNaima = NaimaSpectralModel(radiative_model, seed=seed, distance=1.5 * u.kpc)


model = SkyModel(spectral_model=modelNaima, name="naima-model")
models = Models([model])

print("Naima model")
print(models.to_yaml())

#fermi_models=Models([model])
#fermi_models.append(fermi_bkg_model)
#dataset_fermi.models=Models(Fermi_model)
#dataset_fermi.models.
#print("SJL-Loaded dataset_fermi")
#print(dataset_fermi)

# Attempt to find fermi data with Naima
# dataset_all.append(dataset_fermi)


#Load crab using OGIP format data rather than horrendous serialised format
#dataset_fermi = Datasets.read(filename="/home/sheridan/Documents/gammapy-master/datasets/gammapy-datasets/joint-crab/spectra/fermi/pha_obs0.fits")
#dataset_fermi=SpectrumDataset.read(filename="/home/sheridan/Documents/gammapy-master/datasets/gammapy-datasets/joint-crab/spectra/fermi/pha_obs0.fits",format="OGIP")

#energy_edges = MapAxis.from_energy_bounds("10 GeV", "2 TeV", nbin=5).edges

#flux_points_fermi = FluxPointsEstimator(
#    energy_edges=energy_edges,source="naima-model"
#).run([dataset_fermi])

#flux_points_fermi.plot(ax=ax, sed_type=sed_type, label="Fermi-LAT - fron OGIP file - fitted")


print("SJL - dataset_all with Fermi-LAT include")
print("Dataset all")
# Unfitted model for comparison
particle_distribution1 = naima.models.ExponentialCutoffPowerLaw(
    1e30 / u.eV, 10 * u.TeV, 2.0, 30 * u.TeV
)
radiative_model1 = naima.radiative.InverseCompton(
    particle_distribution1,
    seed_photon_fields=["CMB", ["FIR", 25.5 * u.K, 0.415 * u.eV / u.cm**3]],
    Eemin=100 * u.GeV,
)

for seed, ls in zip(["CMB", "FIR"], ["-", "--"]):
    modelNaima1 = NaimaSpectralModel(radiative_model1, seed=seed, distance=1.5 * u.kpc)


#dataset_fermi_as_spectrumdataset=dataset_fermi.to_spectrum_datasets()
#dataset_all.append(dataset_fermi_as_spectrumdataset)

print ("SJL- Naima Model- before fit")
print ("--------------------")

print(models.to_yaml())
print("Dataset fermi before append")
#dataset_fermi.models=Fermi_model
#print(dataset_fermi)
dataset_all.append(dataset_fermi)
dataset_all.models = models
print ("Dataset all for NAIMA fit")
print(dataset_all)
fit_joint = Fit()
results_joint = fit_joint.run(datasets=dataset_all)

print(results_joint)
print ("SJL- Naima Model- after fit")
print ("--------------------")

print(models.to_yaml())

opts = {
    "energy_bounds": [10 * u.GeV, 80 * u.TeV],
    "sed_type": "dnde",
}

# Plot the total inverse Compton emission
modelNaima.plot(label="Naima IC (total)", **opts)
#modelNaima1.plot(label="Naima - Before Fit", **opts)


ax.set_ylim([3e-18, 5e-10])
plt.legend();
plt.title("Naima and LP model fit to Crab - LAT data now fitted")

plt.show()


