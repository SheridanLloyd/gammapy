# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from astropy.table import Table
from ..utils.scripts import make_path
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.energy import EnergyBounds
from .effective_area import EffectiveAreaTable2D, EffectiveAreaTable
from .background import Background3D
from .energy_dispersion import EnergyDispersion2D
from .psf_gauss import EnergyDependentMultiGaussPSF

__all__ = ["load_CTA_1DC_IRF"]

def load_CTA_1DC_IRF(filename):
    """load CTA instrument response function and return a dictionary container.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.

    The various IRFs are accessible with the following keys
    'aeff' is a `~gammapy.irf.EffectiveAreaTable2D`
    'edisp'  is a `~gammapy.irf.EnergyDispersion2D`
    'psf' is a `~gammapy.irf.EnergyDependentMultiGaussPSF`
    'bkg' is a  `~gammapy.irf.Background3D`

    Parameters
    ----------
    filename : str
        the input filename

    Returns
    -------
    cta_irf : dict
        the IRF dictionary
    """
    filename = str(make_path(filename))

    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")
    bkg = Background3D.read(filename, hdu="BACKGROUND")
    edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    return dict(aeff=aeff, bkg=bkg, edisp=edisp, psf=psf)


