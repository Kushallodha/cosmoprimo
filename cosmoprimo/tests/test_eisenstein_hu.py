import pytest
import numpy as np

from cosmoprimo import Cosmology, fiducial
from cosmoprimo.eisenstein_hu import EisensteinHuEngine, Background, Primordial, Transfer, Fourier


def test_engine():
    with pytest.warns(UserWarning, match='massive neutrinos'):
        Cosmology(engine='eisenstein_hu', m_ncdm=0.1)

    with pytest.warns(UserWarning, match='non-zero curvature'):
        Cosmology(engine='eisenstein_hu', Omega_k=0.1)

    with pytest.warns(UserWarning, match='non-constant dark energy'):
        Cosmology(engine='eisenstein_hu', w0_fld=-0.9)

cosmo = Cosmology(engine='eisenstein_hu')

def test_background():
    assert cosmo.growth_factor(0) == 1.
    fsigma8 = cosmo.growth_rate(0) * cosmo.sigma8_m
    assert (fsigma8 <=0.5) and (fsigma8 >=0.4)

def test_primordial():
    pm = cosmo.get_primordial()
    little_h = cosmo.get_params()['h']
    assert pm._h == little_h
    assert pm.ln_1e10_A_s == np.log(1e10 * cosmo.A_s)
    A_s_cal = pm.pk_interpolator()(pm._k_pivot)
    assert A_s_cal == cosmo.A_s * little_h**3

def test_thermodynamics():
    th = cosmo.get_thermodynamics()
    assert th._z_drag == cosmo.z_drag

def test_fourier():
    fo = cosmo.get_fourier()
    sigma_vel = fo.sigma8_z(0., of='theta_m')
    assert (sigma_vel <=0.5) and (sigma_vel >=0.4)

    

if __name__ == "__main__":
    test_engine()
    test_background()
    test_primordial()
    test_thermodynamics()
    test_fourier()