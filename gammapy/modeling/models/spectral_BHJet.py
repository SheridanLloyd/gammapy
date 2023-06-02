import numpy as np
from astropy import units as u
from gammapy.modeling import Parameter,Parameters
from scipy import interpolate
from .spectral import  SpectralModel
import copy
import ctypes
import pathlib

__all__ = [
    "BHJetSpectralModel",
]

class BHJetSpectralModel(SpectralModel):
    r"""A wrapper for BHJet models.

    For more information see :ref:`naima-spectral-model`.

    Parameters
    ----------
    """

    tag = ["BHJetSpectralModel", "BHJet"]
    name="BHJet"

    def __init__(
        self,
        **kwargs,
    ):

        # Load the BHJet shared library into ctypes - TODO - abstract hard coded path
        npar = int(27)
        libname = "/home/sheridan/Documents/PYTHON_CPP_BINDING/BHJet-master/BHJet-master/BHJet/pyjetmain"
        self.BHJet_lib = ctypes.CDLL(libname)
        self.param = (ctypes.c_double * npar)()

        parameters=[]

        parameters.append(Parameter("Mbh",kwargs.get("Mbh",1e9),frozen=True))
        parameters.append(Parameter("theta",kwargs.get("theta",2.5),frozen=True))
        parameters.append(Parameter("dist",kwargs.get("dist",543e3),frozen=True))
        parameters.append(Parameter("redsh",kwargs.get("redsh",0),frozen=True))
        parameters.append(Parameter("jetrat",kwargs.get("jetrat",.9e-2),frozen=True))
        parameters.append(Parameter("r_0",kwargs.get("r_0",18),frozen=True))
        parameters.append(Parameter("z_diss",kwargs.get("z_diss",600),frozen=True))
        parameters.append(Parameter("z_acc",kwargs.get("z_acc",600),frozen=True))
        parameters.append(Parameter("z_max",kwargs.get("z_max",6.6e5),frozen=True))
        parameters.append(Parameter("t_e", kwargs.get("t_e",300),frozen=False))# free
        parameters.append(Parameter("f_nth",kwargs.get("f_nth",.1),frozen=False))#free
        parameters.append(Parameter("f_pl",kwargs.get("f_pl",0),frozen=True))
        parameters.append(Parameter("pspec",kwargs.get("pspec",1.95),frozen=True))
        parameters.append(Parameter("f_heat",kwargs.get("f_heat",12.),frozen=False))#free
        parameters.append(Parameter("f_beta",kwargs.get("f_beta",10.),frozen=True))
        parameters.append(Parameter("f_sc",kwargs.get("f_sc",3e-6),frozen=False))#free
        parameters.append(Parameter("p_beta",kwargs.get("p_beta",2),frozen=True))
        parameters.append(Parameter("sig_acc",kwargs.get("sig_acc",0.025),frozen=True))
        parameters.append(Parameter("l_disk",kwargs.get("l_disk",0.01),frozen=True))
        parameters.append(Parameter("r_in",kwargs.get("r_in",50),frozen=True))
        parameters.append(Parameter("r_out",kwargs.get("r_out",1e3),frozen=True))
        parameters.append(Parameter("compar1",kwargs.get("compar1",3300),frozen=True))
        parameters.append(Parameter("compar2",kwargs.get("compar2",3e-10),frozen=True))
        parameters.append(Parameter("compar3",kwargs.get("compar3",3.e10),frozen=True))
        parameters.append(Parameter("compsw",kwargs.get("compsw",0),frozen=True))
        parameters.append(Parameter("velsw",kwargs.get("velsw",15),frozen=True))
        parameters.append(Parameter("infosw",kwargs.get("infosw",1),frozen=True))

        # Sky model initialisation requires a norm parameter or throws error - however BHJet returns
        # physically distance adjusted fluxes where no nomrlaisation is required - we freeze this to 1
        dummy_norm = Parameter("dummy_norm",1,is_norm=True,frozen=True)
        parameters.append(dummy_norm)

        self.default_parameters=Parameters(parameters)

        super().__init__()

    # Python deep copy is called on this model
    # when dealing with SkyModel
    # However deep copy cannot deal with ctypes - throws exception -
    # so we override, here to initially exclude ctypes
    # and then add back in manually later
    # they do not require a deeep copy
    def __deepcopy__(self, memo):
        cls = self.__class__  # Extract the class of the object
        result = cls.__new__(cls)  # Create a new instance of the object based on extracted class
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # skip ctypes and exclude or set manually <TBD>
            if k in ['BHJet_lib','ebins','param','logHz','logmJy']:
                continue
            setattr(result, k, copy.deepcopy(v,memo))  # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.


        # Load the BHJet shared library into ctypes - TODO - abstract hard coded path
        libname = "/home/sheridan/Documents/PYTHON_CPP_BINDING/BHJet-master/BHJet-master/BHJet/pyjetmain"

        result.BHJet_lib = ctypes.CDLL(libname)
        #todo Set emin and emax from energy ranges passed in
        npar = int(27)    # number of parameters for BHJet
        ne = int(201)     # number of energy bins
        emin = float(-10) # spans whole spectrum
        emax = float(12)
        einc = (emax - emin) / ne

        result.ebins = (ctypes.c_double * ne)()
        result.param = (ctypes.c_double * npar)()
        result.logHz = (ctypes.c_double * (ne - 1))()
        result.logmJy = (ctypes.c_double * (ne - 1))()
        for i in range(ne):
            result.ebins[i] = pow(10, (emin + i * einc))
        result.ne=ne

        return result

    # For diagnostic only, remove in final version
    def ScatterPlot(self,BHJetFreqHz,BHJetmJy,interpolateHz,interpolatemJy):
        import matplotlib.pyplot as plt


        # ax1.set_ylim(top=200)
        plt.title("Interpolation diagnostic")
        plt.ylabel('flux mJy')
        plt.xlabel('freq Hz' )
        plt.loglog(BHJetFreqHz, BHJetmJy, '-', label="BHJetModel return values", color='b')
        plt.loglog(interpolateHz, interpolatemJy, 'o', label="Numpy Interpolation function", color='r')
        plt.legend(loc='upper right')
        plt.show()  # if you want to see a plot uncomment this
        return

    def evaluate(self, energy, **kwargs):

        """Evaluate the model.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy to evaluate the model at.
            This can be a list of energies
        Returns
        -------
        dnde : `~astropy.units.Quantity`
            Differential flux at given energy.
        """
        self.param[0] = kwargs.get("Mbh")
        self.param[1] = kwargs.get("theta")
        self.param[2] = kwargs.get("dist")
        self.param[3] = kwargs.get("redsh")
        self.param[4] = kwargs.get("jetrat")
        self.param[5] = kwargs.get("r_0")
        self.param[6] = kwargs.get("z_diss")
        self.param[7] = kwargs.get("z_acc")
        self.param[8] = kwargs.get("z_max")
        self.param[9] = kwargs.get("t_e")
        self.param[10] = kwargs.get("f_nth")
        self.param[11] = kwargs.get("f_pl")
        self.param[12] = kwargs.get("pspec")
        self.param[13] = kwargs.get("f_heat")
        self.param[14] = kwargs.get("f_beta")
        self.param[15] = kwargs.get("f_sc")
        self.param[16] = kwargs.get("p_beta")
        self.param[17] = kwargs.get("sig_acc")
        self.param[18] = kwargs.get("l_disk")
        self.param[19] = kwargs.get("r_in")
        self.param[20] = kwargs.get("r_out")
        self.param[21] = kwargs.get("compar1")
        self.param[22] = kwargs.get("compar2")
        self.param[23] = kwargs.get("compar3")
        self.param[24] = kwargs.get("compsw")
        self.param[25] = kwargs.get("velsw")
        self.param[26] = kwargs.get("infosw")

        print("kwargs in evaluate")
        print(kwargs)

        ne = int(201)     # number of energy bins
        emin = float(-10) # span
        emax = float(10)
        einc = (emax - emin) / ne

        self.ebins = (ctypes.c_double * ne)()
        self.logHz = (ctypes.c_double * (ne - 1))()
        self.logmJy = (ctypes.c_double * (ne - 1))()

        for i in range(ne):
            self.ebins[i] = pow(10, (emin + i * einc))
        self.ne=ne

        self.BHJet_lib.pyjetmain(self.ebins, self.ne - 1, self.param, self.logHz, self.logmJy)

        # Extract range of frequencies (Hz) vs energies (mJy) and interpolate for passed in Tev energy to this evaluate
        # BHJet returns distance and redshift adjusted energy values from radio to the TeV
        # which is the desired range of observations to fit for MM/MWL
        freqHz=[]
        energymJy=[]

        for i in self.logHz:
            freqHz.append(10**i)
        for j in self.logmJy:
            energymJy.append(10**j)

        dnde=[]
        interpolateHz=[]
        interpolatemJy=[]

        # Convert passed in energy(ies) to Hz for interpolation against Hz vs mJy returned by BHJet model
        # for an arbritary range of energies. May enhance BHJet in future to accept specific energy
        # lookup
        energyHz=energy.to(u.Hz, equivalencies=u.spectral())

        # Initialisation call can pass an energy tuple with a single 1 TeV value
        # not sure if this is by design and might happen in other contexts
        # so will process for now
        if str(energyHz.shape) =='(1, 1, 1)':
            for energy_to_interpolate in energyHz:
                interpolateFlux = (np.interp(energy_to_interpolate[0][0].value,freqHz, energymJy) * u.mJy).value
                interpolateHz.append(energy_to_interpolate[0][0].value)
                interpolatemJy.append(interpolateFlux)
                #self.ScatterPlot(freqHz, energymJy, interpolateHz, interpolatemJy)
        else:
            # a sensible array of normal quantities - passed in from fit of SkyModel
            interpolatemJy = (np.interp(energyHz.value,freqHz, energymJy) * u.mJy).value
            #self.ScatterPlot(freqHz, energymJy, energyHz, interpolatemJy)

        # dnde = dnde.reshape(energy.shape)
        # unit = 1 / (energy.unit * u.cm**2 * u.s)

        # convert interpolated mJy from BHJet to expected return units
        # 
        y = (interpolatemJy*u.mJy).to(u.photon / u.cm ** 2 / u.s / u.Hz,
                                   equivalencies=u.spectral_density(energy.to(u.Hz, equivalencies=u.spectral())))
        # manual conversion as astropy units won't handle it
        dnde = 1 / (u.TeV * u.cm ** 2 * u.s)
        dnde = dnde * y.value * 2.417 * 1e26  # (1TeV to Hz)

        return dnde

    def to_dict(self, full_output=True):
        # for full_output to True otherwise broken
        return super().to_dict(full_output=True)

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError(
        "Currently the BHJetSpectralModel cannot be read from YAML"
    )

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        raise NotImplementedError(
        "Currently the BHJetSpectralModel cannot be built from a list of parameters."
    )



