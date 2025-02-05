import os
import tempfile

import pytest
import numpy as np

from cosmoprimo import (Cosmology, Background, Thermodynamics, Primordial,
                        Harmonic, Fourier, CosmologyError, CosmologyInputError, CosmologyComputationError,
                        constants)


def test_params():
    cosmo = Cosmology()
    with pytest.raises(CosmologyError):
        cosmo = Cosmology(sigma8=1., A_s=1e-9)
    params = {'Omega_cdm': 0.3, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = Cosmology(**params)
    assert cosmo['omega_cdm'] == 0.3 * 0.8**2
    assert len(cosmo['z_pk']) == 30
    assert cosmo['sigma8'] == 0.8
    for neutrino_hierarchy in ['normal', 'inverted', 'degenerate']:
        cosmo = Cosmology(m_ncdm=0.1, neutrino_hierarchy=neutrino_hierarchy)
        assert len(cosmo['m_ncdm']) == 3
        assert np.allclose(sum(cosmo['m_ncdm']), 0.1)
        assert np.allclose(cosmo['m_ncdm_tot'], 0.1)
    m_ncdm = [0.01, 0.02, 0.05]
    cosmo = Cosmology(m_ncdm=m_ncdm)
    Background(cosmo, engine='class')
    Fourier(cosmo)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'cosmo.npy')
        cosmo.save(fn)
        cosmo = Cosmology.load(fn)

    assert np.allclose(cosmo['m_ncdm'], m_ncdm)
    assert cosmo.engine.__class__.__name__ == 'ClassEngine'
    Fourier(cosmo)

    with pytest.raises(CosmologyInputError):
        cosmo = Cosmology(tau=0.05, tau_reio=0.06)
    cosmo = Cosmology(ombh2=0.05, omch2=0.1)
    assert np.allclose(cosmo['omega_b'], 0.05) and np.allclose(cosmo['omega_cdm'], 0.1)


def test_engine():
    cosmo = Cosmology(engine='class')
    cosmo.set_engine(engine='camb')
    cosmo.set_engine(engine=cosmo.engine)
    ba = cosmo.get_background()
    ba = Background(cosmo)
    assert ba._engine is cosmo.engine
    ba = cosmo.get_background(engine='camb', set_engine=False)
    ba = Background(cosmo, engine='camb', set_engine=False)
    assert cosmo.engine is not ba._engine
    assert type(ba) is not type(Background(cosmo, engine='class'))
    assert type(cosmo.get_background()) is not type(cosmo.get_background(engine='camb'))
    assert type(cosmo.get_background()) is type(cosmo.get_background(engine='camb'))


list_params = [{}, {'sigma8': 1., 'non_linear': 'mead'}, {'logA': 3., 'non_linear': 'mead'},
               {'A_s': 2e-9, 'alpha_s': -0.2}, {'lensing': True},
               {'m_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}, {'Omega_k': 0.1},
               {'w0_fld': -0.9, 'wa_fld': 0.1, 'cs2_fld': 0.9}, {'w0_fld': -1.1, 'wa_fld': 0.2}][-2:]


@pytest.mark.parametrize('params', list_params)
def test_background(params, seed=42):

    rng = np.random.RandomState(seed=seed)
    cosmo = Cosmology(**params)

    for engine in ['class', 'camb', 'astropy', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        ba = cosmo.get_background(engine=engine)
        for name in ['T0_cmb', 'T0_ncdm', 'Omega0_cdm', 'Omega0_b', 'Omega0_k', 'Omega0_g', 'Omega0_ur', 'Omega0_r',
                     'Omega0_pncdm', 'Omega0_pncdm_tot', 'Omega0_ncdm', 'Omega0_ncdm_tot',
                     'Omega0_m', 'Omega0_Lambda', 'Omega0_fld', 'Omega0_de']:
            assert np.allclose(getattr(ba, name), cosmo[name.replace('0', '')], atol=0, rtol=1e-3)
            assert np.allclose(getattr(ba, name), getattr(ba, name.replace('0', ''))(0.), atol=0, rtol=1e-3)

        for name in ['H0', 'h', 'N_ur', 'N_ncdm', 'm_ncdm', 'm_ncdm_tot', 'N_eff', 'w0_fld', 'wa_fld', 'cs2_fld', 'K']:
            assert np.allclose(getattr(ba, name), cosmo[name], atol=1e-9, rtol=1e-8 if name not in ['N_eff'] else 1e-4)

    ba_class = Background(cosmo, engine='class')
    ba = cosmo.get_background(engine='camb')

    def assert_allclose(ba, name, atol=0, rtol=1e-4):
        test, ref = getattr(ba, name), getattr(ba_class, name)
        has_species = name.endswith('ncdm')
        shape = (cosmo['N_ncdm'], ) if has_species else ()
        z = rng.uniform(0., 1., 10)
        assert np.allclose(test(z=z), ref(z), atol=atol, rtol=rtol)
        assert test(0.).shape == shape
        assert test([]).shape == shape + (0, )
        z = np.array(0.)
        assert test(z).dtype.itemsize == z.dtype.itemsize
        z = np.array([0., 1.])
        assert test(z).shape == shape + z.shape
        z = np.array([[0., 1.]] * 4, dtype='f4')
        assert test(z).shape == shape + z.shape
        assert test(z).dtype.itemsize == z.dtype.itemsize
        if has_species and cosmo['N_ncdm']:
            assert test(0., species=0).shape == ()
            assert test([], species=0).shape == (0, )
            assert test([0., 1.], species=0).shape == (2, )
            assert test([0., 1.], species=[0]).shape == (1, 2, )

    for engine in ['class', 'camb', 'astropy', 'eisenstein_hu']:
        ba = cosmo.get_background(engine=engine)
        for name in ['T_cmb', 'T_ncdm']:
            assert_allclose(ba, name, atol=0, rtol=1e-4)
        for name in ['rho_crit', 'p_ncdm', 'p_ncdm_tot', 'Omega_pncdm', 'Omega_pncdm_tot']:
            assert_allclose(ba, name, atol=0, rtol=1e-4)
        for density in ['rho', 'Omega']:
            for species in ['cdm', 'b', 'k', 'g', 'ur', 'r', 'ncdm', 'ncdm_tot', 'm', 'Lambda', 'fld', 'de']:
                name = '{}_{}'.format(density, species)
                assert_allclose(ba, name, atol=0, rtol=1e-4)

        names = ['efunc', 'hubble_function']
        for name in names:
            assert_allclose(ba, name, atol=0, rtol=2e-4)
        names = []
        rtol = 2e-4
        if engine in ['class', 'camb', 'astropy', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
            names += ['time', 'comoving_radial_distance', 'luminosity_distance', 'angular_diameter_distance', 'comoving_angular_distance']
        if engine in ['class']:
            names += ['growth_factor', 'growth_rate']
        if engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks'] and not cosmo['N_ncdm'] and not cosmo._has_fld:
            rtol = 2e-2
            names += ['growth_factor', 'growth_rate']
        for name in names:
            assert_allclose(ba, name, atol=0, rtol=rtol)
        if engine in ['class', 'camb', 'astropy']:
            z1, z2 = rng.uniform(0., 1., 10), rng.uniform(0., 1., 10)
            assert np.allclose(ba.angular_diameter_distance_2(z1, z2), ba_class.angular_diameter_distance_2(z1, z2), atol=0, rtol=5e-3 if engine == 'astropy' else 2e-4)
            for name in ['age', 'K']:
                assert np.allclose(getattr(ba, name), getattr(ba_class, name), atol=0, rtol=1e-3)


@pytest.mark.parametrize('params', list_params)
def test_thermodynamics(params):
    cosmo = Cosmology(**params)
    th_class = Thermodynamics(cosmo, engine='class')

    for engine in ['camb']:
        th = Thermodynamics(cosmo, engine=engine)
        for name in ['z_drag', 'rs_drag', 'z_star', 'rs_star']:  # weirdly enough, class's z_rec seems to match camb's z_star much better
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=5e-3 if 'star' in name else 2e-4)
        for name in ['theta_star', 'theta_cosmomc']:
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=5e-3 if 'star' in name else 5e-5)
        for name in ['YHe']:
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=1e-2)
        assert np.allclose(th_class.theta_cosmomc, cosmo['theta_cosmomc'], atol=0., rtol=3e-6)
        assert np.allclose(th.theta_cosmomc, cosmo['theta_cosmomc'], atol=0., rtol=3e-6)
    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants']:
        for name in ['z_drag', 'rs_drag']:
            assert np.allclose(getattr(th, name), getattr(th_class, name), atol=0, rtol=1e-2)


@pytest.mark.parametrize('params', list_params)
def test_primordial(params, seed=42):
    rng = np.random.RandomState(seed=seed)
    cosmo = Cosmology(**params)
    pm_class = Primordial(cosmo, engine='class')
    for engine in ['camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        pm = Primordial(cosmo, engine=engine)
        for name in (['A_s'] if 'sigma8' not in cosmo._params else []) + ['n_s', 'alpha_s', 'beta_s', 'k_pivot']:
            assert np.allclose(getattr(pm_class, name), cosmo['k_pivot'] / cosmo['h'] if name == 'k_pivot' else cosmo[name])
            assert np.allclose(getattr(pm, name), getattr(pm_class, name), atol=0, rtol=1e-5)

    for engine in ['camb']:
        pm = Primordial(cosmo, engine=engine)
        for name in ['A_s', 'ln_1e10_A_s']:
            assert np.allclose(getattr(pm, name), getattr(pm_class, name), atol=0, rtol=2e-3)
        k = rng.uniform(1e-3, 10., 100)
        for mode in ['scalar', 'tensor']:
            assert np.allclose(pm.pk_k(k, mode=mode), pm_class.pk_k(k, mode=mode), atol=0, rtol=2e-3)
            assert np.allclose(pm.pk_interpolator(mode=mode)(k), pm_class.pk_interpolator(mode=mode)(k), atol=0, rtol=2e-3)

    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        pm = Primordial(cosmo, engine=engine)
        for name in ['A_s', 'ln_1e10_A_s']:
            assert np.allclose(getattr(pm, name), getattr(pm_class, name), atol=0, rtol=1e-1)

    k = np.logspace(-3, 1, 100)
    for engine in ['camb', 'class', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        pm = Primordial(cosmo, engine=engine)
        assert np.allclose(pm.pk_interpolator(mode='scalar')(k), (cosmo.h**3 * pm.A_s * (k / pm.k_pivot) ** (pm.n_s - 1. + 1. / 2. * pm.alpha_s * np.log(k / pm.k_pivot))))


@pytest.mark.parametrize('params', list_params)
def test_harmonic(params):
    cosmo = Cosmology(**params)
    hr_class = Harmonic(cosmo, engine='class')
    test = hr_class.unlensed_cl()
    ref = hr_class.unlensed_table(of=['tt', 'ee', 'bb', 'te'])
    assert all(np.allclose(test[key], ref[key]) for key in ref.dtype.names)

    for engine in ['camb']:
        hr = Harmonic(cosmo, engine=engine)
        for name in ['lensed_cl', 'lens_potential_cl']:
            for ellmax in [100, -1]:
                if not cosmo['lensing']:
                    if engine == 'class':
                        from pyclass import ClassComputationError
                        with pytest.raises(ClassComputationError):
                            getattr(hr, name)(ellmax=ellmax)
                    if engine == 'camb':
                        from camb import CAMBError
                        with pytest.raises(CAMBError):
                            getattr(hr, name)(ellmax=ellmax)
                else:
                    tmp_class = getattr(hr_class, name)(ellmax=ellmax)
                    tmp = getattr(hr, name)(ellmax=ellmax)
                    assert tmp_class.dtype == tmp.dtype
                    for field in tmp_class.dtype.names[1:]:
                        if name == 'lensed_cl':
                            atol = tmp_class[field].std() * 1e-2  # to deal with 0 oscillating behavior
                            rtol = 1e-2
                            # print(name, field, tmp_class[field], tmp_camb[field])
                            # print(hr_class.unlensed_cl(ellmax=ellmax)[field], hr_camb.unlensed_cl(ellmax=ellmax)[field])
                        else:
                            atol = tmp_class[field].std() * 1e-1  # to deal with 0 oscillating behavior
                            rtol = 1e-1
                        assert np.allclose(tmp[field], tmp_class[field], atol=atol, rtol=rtol)
        for name in ['unlensed_cl']:
            for ellmax in [100, -1]:
                tmp_class = getattr(hr_class, name)(ellmax=ellmax)
                tmp = getattr(hr, name)(ellmax=ellmax)
                assert tmp_class.dtype == tmp.dtype
                for field in tmp_class.dtype.names[1:]:
                    atol = tmp_class[field].std() * 1e-2
                    assert np.allclose(tmp[field], tmp_class[field], atol=atol, rtol=2e-2)


@pytest.mark.parametrize('params', list_params)
def test_fourier(params, seed=42):
    rng = np.random.RandomState(seed=seed)
    cosmo = Cosmology(**params)

    if 'sigma8' in cosmo._params:
        assert cosmo['sigma8'] == params.get('sigma8', 0.8)  # sigma8 is set as default
        with pytest.raises(CosmologyError):
            cosmo['A_s']
    else:
        for name in ['A_s', 'logA']:
            if name in params:
                assert cosmo[name] == params[name]
        for name in ['ln10^{10}A_s', 'ln10^10A_s']:
            assert cosmo[name] == np.log(10**10 * cosmo['A_s'])
        with pytest.raises(CosmologyError):
            cosmo['sigma8']

    fo_class = Fourier(cosmo, engine='class', gauge='newtonian')

    for engine in ['class', 'camb'][1:]:
        fo = Fourier(cosmo, engine=engine)
        zmax = 2.5 if params.get('non_linear', False) else 6.  # because classy fails at higher z when non_linear
        z = rng.uniform(0., zmax, 20)
        r = rng.uniform(1., zmax, 10)
        if 'sigma8' in cosmo._params:
            assert np.allclose(fo.sigma8_z(0, of='delta_m'), cosmo._params['sigma8'], atol=0., rtol=1e-3)
            assert np.allclose(fo.pk_interpolator(non_linear=False, of='delta_m').sigma8_z(z=0.), cosmo._params['sigma8'], atol=0., rtol=1e-3)
        for of in ['delta_m', 'delta_cb', ('delta_cb', 'theta_cb'), 'theta_cb']:
            assert np.allclose(fo.sigma_rz(r, z, of=of), fo_class.sigma_rz(r, z, of=of), atol=0., rtol=1e-3)
            assert np.allclose(fo.sigma8_z(z, of=of), fo_class.sigma8_z(z, of=of), atol=0., rtol=1e-3)

        z = rng.uniform(0., zmax, 10)
        k = rng.uniform(1e-3, 1., 20)

        for of in ['delta_m', 'delta_cb']:
            #assert np.allclose(fo.pk_interpolator(non_linear=False, of=of)(k, z=z), fo_class.pk_interpolator(non_linear=False, of=of)(k, z=z), rtol=2.5e-3)
            assert np.allclose(fo.pk_interpolator(non_linear=False, of=of).sigma8_z(z=z), fo.sigma8_z(z, of=of), atol=0., rtol=1e-4)

        z = np.linspace(0., zmax, 5)
        for of in ['theta_cb', ('delta_m',), ('delta_cb', 'theta_cb'), 'phi_plus_psi']:
            #print(of, fo.pk_interpolator(non_linear=False, of=of)(k, z=z) / fo_class.pk_interpolator(non_linear=False, of=of)(k, z=z))
            assert np.allclose(fo.pk_interpolator(non_linear=False, of=of)(k, z=z), fo_class.pk_interpolator(non_linear=False, of=of)(k, z=z), atol=0., rtol=2.5e-3)

        if params.get('non_linear', False):
            assert np.allclose(fo_class._rsigma8, 1., atol=0., rtol=1e-5) and ('sigma8' not in cosmo._params or fo_class._rsigma8 != 1.)  # small numerical inaccuracies expected
            for of in ['delta_m']:
                #print(np.abs(fo.pk_interpolator(non_linear=True, of=of)(k, z=z) / fo_class.pk_interpolator(non_linear=True, of=of)(k, z=z) - 1).max())
                assert np.allclose(fo.pk_interpolator(non_linear=True, of=of)(k, z=z), fo_class.pk_interpolator(non_linear=True, of=of)(k, z=z), atol=0., rtol=5e-3)

        # if not cosmo['N_ncdm']:
        z = rng.uniform(0., zmax, 20)
        r = rng.uniform(1., zmax, 10)
        pk = fo.pk_interpolator(of='delta_cb')

        for z in np.linspace(0.2, zmax, 5):
            for r in np.linspace(2., 20., 5):
                for dz in [1e-3, 1e-2]:
                    rtol = 1e-3
                    # assert np.allclose(ba_class.growth_rate(z), pk_class.growth_rate_rz(r=r, z=z, dz=dz), atol=0, rtol=rtol)
                    f = fo.sigma_rz(r, z, of='theta_cb') / fo.sigma_rz(r, z, of='delta_cb')
                    assert np.allclose(f, pk.growth_rate_rz(r=r, z=z, dz=dz), atol=0., rtol=rtol)

    for engine in ['eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks']:
        fo = Fourier(cosmo, engine=engine)
        pk_class = fo_class.pk_interpolator(non_linear=False, of='delta_m')
        pk = fo.pk_interpolator()
        rtol = 0.3 if engine == 'bbks' else 0.15
        assert np.allclose(pk(k, z=z), pk_class(k, z=z), atol=0., rtol=rtol)
        r = rng.uniform(1., 10., 10)
        assert np.allclose(pk.growth_rate_rz(r=r, z=z), pk_class.growth_rate_rz(r=r, z=z), atol=0., rtol=0.15)


def test_pk_norm():
    cosmo = Cosmology(engine='class')
    power_prim = cosmo.get_primordial().pk_interpolator()
    z = 1.
    k = np.logspace(-3., 1., 1000)
    assert np.allclose(cosmo.sigma8_z(0, of='delta_m'), cosmo['sigma8'], rtol=1e-3)
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    pk = power(k)
    pk_prim = power_prim(k)
    k0 = power.k[0]
    tk = (pk / power_prim(k) / k / (power(k0) / power_prim(k0) / k0))**0.5

    potential_to_density = (3. * cosmo.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * k**2)) ** (-2)
    curvature_to_potential = 9. / 25. * 2. * np.pi**2 / k**3 / cosmo.h**3
    znorm = 10.
    normalized_growth_factor = cosmo.growth_factor(z) / cosmo.growth_factor(znorm) / (1 + znorm)
    pk_test = normalized_growth_factor**2 * tk**2 * potential_to_density * curvature_to_potential * pk_prim
    assert np.allclose(pk_test, pk, atol=0., rtol=1e-3)


def plot_primordial_power_spectrum():
    from matplotlib import pyplot as plt
    cosmo = Cosmology(Omega_k=0.1)
    pm_class = Primordial(cosmo, engine='class')
    pm_camb = Primordial(cosmo, engine='camb')
    pm_eh = Primordial(cosmo, engine='eisenstein_hu')
    k = np.logspace(-6, 2, 500)
    plt.loglog(k, pm_class.pk_interpolator()(k), label='class')
    plt.loglog(k, pm_camb.pk_interpolator()(k), label='camb')
    plt.loglog(k, pm_eh.pk_interpolator()(k), label='eisenstein_hu')
    plt.legend()
    plt.show()


def plot_harmonic():
    from matplotlib import pyplot as plt
    cosmo = Cosmology(lensing=True)
    hr_class = Harmonic(cosmo, engine='class')
    cls = hr_class.lensed_cl()
    ells_factor = (cls['ell'] + 1) * cls['ell'] / (2 * np.pi)
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='class')
    hr_camb = Harmonic(cosmo, engine='camb')
    cls = hr_camb.lensed_cl()
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='camb')
    cosmo = Cosmology(lensing=True, m_ncdm=0.1)
    hr_class = Harmonic(cosmo, engine='class')
    cls = hr_class.lensed_cl()
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='class + neutrinos')
    hr_camb = Harmonic(cosmo, engine='camb')
    cls = hr_camb.lensed_cl()
    plt.plot(cls['ell'], ells_factor * cls['tt'], label='camb + neutrinos')
    plt.legend()
    plt.show()


def plot_non_linear():
    from matplotlib import pyplot as plt
    cosmo = Cosmology(non_linear='mead')
    k = np.logspace(-3, 1, 1000)
    z = 1.
    for of in ['delta_m', 'delta_cb']:
        for engine, color in zip(['class', 'camb'], ['C0', 'C1']):
            for non_linear, linestyle in zip([False, True], ['-', '--']):
                pk = cosmo.get_fourier(engine=engine).pk_interpolator(non_linear=non_linear, of=of)(k, z=z)
                plt.loglog(k, pk, color=color, linestyle=linestyle, label=engine + (' non-linear' if non_linear else ''))
        plt.legend()
        plt.show()

    for engine, color in zip(['class', 'camb'], ['C0', 'C1']):
        for non_linear, linestyle in zip([False, True], ['-', '--']):
            cosmo = Cosmology(lensing=True, non_linear='mead' if non_linear else '')
            cls = cosmo.get_harmonic(engine=engine).lens_potential_cl()
            ells_factor = (cls['ell'] + 1)**2 * cls['ell']**2 / (2 * np.pi)
            plt.plot(cls['ell'], ells_factor * cls['pp'], color=color, linestyle=linestyle, label=engine + (' non-linear' if non_linear else ''))
            plt.xscale('log')
    plt.legend()
    plt.show()


def plot_matter_power_spectrum():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    fo_class = Fourier(cosmo, engine='class')
    # k, z, pk = fo_class.table()
    # plt.loglog(k, pk)
    z = 1.
    k = np.logspace(-6, 2, 500)
    pk = fo_class.pk_interpolator(non_linear=False, of='delta_m', extrap_kmin=1e-7)
    # pk = fo_class.pk_kz
    plt.loglog(k, pk(k, z=z), label='class')
    pk = Fourier(cosmo, engine='camb').pk_interpolator(non_linear=False, of='delta_m', extrap_kmin=1e-7)
    plt.loglog(k, pk(k, z=z), label='camb')
    pk = Fourier(cosmo, engine='eisenstein_hu').pk_interpolator()
    plt.loglog(k, pk(k, z=z), label='eisenstein_hu')
    pk = Fourier(cosmo, engine='eisenstein_hu_nowiggle').pk_interpolator()
    plt.loglog(k, pk(k, z=z), label='eisenstein_hu_nowiggle')
    pk = Fourier(cosmo, engine='eisenstein_hu_nowiggle_variants').pk_interpolator()
    plt.loglog(k, pk(k, z=z), label='eisenstein_hu_nowiggle_variants')
    plt.legend()
    plt.show()


def plot_non_linear_matter_power_spectrum():
    from matplotlib import pyplot as plt
    non_linear = 'mead'
    cosmo = Cosmology(lensing=True, non_linear=non_linear, engine='class')
    #cosmo.get_harmonic()  # previous bug with pyclass
    fo_class = Fourier(cosmo)
    # k, z, pk = fo_class.table()
    # plt.loglog(k, pk)
    z = np.linspace(1., 2., 3)
    k = np.logspace(-6, 2, 500)
    pklin = fo_class.pk_interpolator(non_linear=False, of='delta_m')
    pknonlin = fo_class.pk_interpolator(non_linear=True, of='delta_m')
    for iz, zz in enumerate(z):
        plt.loglog(k, pklin(k, z=zz), color='C0', label='linear' if iz == 0 else None)
        plt.loglog(k, pknonlin(k, z=zz), color='C1', label=non_linear if iz == 0 else None)
    plt.legend()
    plt.show()

    fo_camb = Fourier(cosmo, engine='camb')
    pklin_camb = fo_camb.pk_interpolator(non_linear=False, of='delta_m')
    pknonlin_camb = fo_camb.pk_interpolator(non_linear=True, of='delta_m')
    for iz, zz in enumerate(z):
        plt.plot(k, pknonlin(k, z=zz) / pklin(k, z=zz), color='C0', label='class boost' if iz == 0 else None)
        plt.plot(k, pknonlin_camb(k, z=zz) / pklin_camb(k, z=zz), color='C1', label='camb boost' if iz == 0 else None)
    plt.xscale('log')
    plt.legend()
    plt.show()


def plot_matter_power_spectra():
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    for ii, logA in enumerate(np.linspace(2.9, 3.1, 5)):
        cosmo = fiducial.clone(logA=logA)
        fo_class = Fourier(cosmo, engine='class')
        # k, z, pk = fo_class.table()
        # plt.loglog(k, pk)
        z = 1.
        k = np.logspace(-6, 2, 500)
        for of in [('delta_m', 'delta_m'), ('delta_cb', 'delta_cb'), ('delta_cb', 'theta_cb'), ('theta_cb', 'theta_cb')]:
            pk = fo_class.pk_interpolator(non_linear=False, of=of)
            # pk = fo_class.pk_kz
            plt.loglog(k, pk(k, z=z), label=str(of) if ii == 0 else None)

    plt.legend()
    plt.show()


def plot_eisenstein_hu_nowiggle_variants():
    from matplotlib import pyplot as plt
    cosmo = Cosmology()
    z = 1.
    k = np.logspace(-6, 2, 500)
    for m_ncdm in [0., 1.1]:
        cosmo = cosmo.clone(m_ncdm=m_ncdm, T_ncdm_over_cmb=None)
        pk = Fourier(cosmo, engine='eisenstein_hu_nowiggle_variants').pk_interpolator()
        plt.loglog(k, pk(k, z=z), label=r'$m_{{ncdm}} = {:.2f} \mathrm{{eV}}$'.format(m_ncdm))
    plt.legend()
    plt.show()


def test_external_camb():
    import camb
    from camb import CAMBdata

    params = camb.CAMBparams(H0=70, omch2=0.15, ombh2=0.02)
    params.DoLensing = True
    params.Want_CMB_lensing = True
    tr = CAMBdata()
    tr.calc_power_spectra(params)
    print(tr.get_lens_potential_cls(lmax=100, CMB_unit=None, raw_cl=True))

    params = camb.CAMBparams(H0=70, omch2=0.15, ombh2=0.02)
    As = params.InitPower.As
    ns = params.InitPower.ns
    params.DoLensing = False
    # params.Want_CMB_lensing = True
    # params.Want_CMB_lensing = True
    tr = camb.get_transfer_functions(params)
    tr.Params.InitPower.set_params(As=As, ns=ns)
    tr.calc_power_spectra()
    tr.Params.DoLensing = True
    tr.Params.Want_CMB_lensing = True
    print(tr.get_lens_potential_cls(lmax=100, CMB_unit=None, raw_cl=True))

    params = camb.CAMBparams(H0=70, omch2=0.15, ombh2=0.02)
    #params.WantCls = False
    params.Want_CMB = False
    # params.WantTransfer = True
    tr = camb.get_transfer_functions(params)
    params.Want_CMB = True
    tr.calc_power_spectra(params)
    print(tr.get_unlensed_scalar_cls(lmax=100, CMB_unit=None, raw_cl=True))
    # print(tr.get_total_cls(lmax=100, CMB_unit=None, raw_cl=True))


def test_external_pyccl():
    try: import pyccl
    except ImportError: return
    print('With pyccl')
    params = {'sigma8': 0.8, 'Omega_c': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_nu': 0.1, 'm_nu_type': 'normal'}
    cosmo = pyccl.Cosmology(**params)
    print(pyccl.background.growth_rate(cosmo, 1))
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}
    cosmo = Cosmology(**params)
    print(Background(cosmo, engine='class').growth_rate(0))

    params = {'sigma8': 0.8, 'Omega_c': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = pyccl.Cosmology(**params)
    print(pyccl.background.growth_rate(cosmo, 1))
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = Cosmology(**params)
    print(Background(cosmo, engine='class').growth_rate(0))


def benchmark():
    import timeit
    import pyccl
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}
    pyccl_params = {'sigma8': 0.8, 'Omega_c': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_nu': 0.1, 'm_nu_type': 'normal', 'transfer_function': 'boltzmann_class'}
    z = np.linspace(0., 10., 10000)
    z_pk = 1.  # ccl does not support vectorization over scale factor
    k = np.logspace(-4, 2, 500)
    a = 1. / (1 + z)
    a_pk = 1. / (1 + z_pk)
    d = {}
    d['cosmoprimo initialisation'] = {'stmt': "Cosmology(**params)", 'number': 1000}
    d['pyccl initialisation'] = {'stmt': "pyccl.Cosmology(**pyccl_params)", 'number': 1000}

    d['cosmoprimo initialisation + background'] = {'stmt': "cosmo = Cosmology(**params); cosmo.get_background('class').comoving_radial_distance(z)",
                                                   'number': 10}
    d['pyccl initialisation + background'] = {'stmt': "cosmo = pyccl.Cosmology(**pyccl_params); pyccl.background.comoving_radial_distance(cosmo, a)",
                                              'number': 10}

    d['cosmoprimo initialisation + background single z'] = {'stmt': "cosmo = Cosmology(**params); cosmo.get_background('class').comoving_radial_distance(z_pk)",
                                                            'number': 10}
    d['pyccl initialisation + background single z'] = {'stmt': "cosmo = pyccl.Cosmology(**pyccl_params); pyccl.background.comoving_radial_distance(cosmo, a_pk)",
                                                       'number': 10}

    d['cosmoprimo initialisation + pk'] = {'stmt': "cosmo = Cosmology(**params); cosmo.get_fourier('class').pk_interpolator()(k, z_pk)",
                                           'number': 2}
    d['pyccl initialisation + pk'] = {'stmt': "cosmo = pyccl.Cosmology(**pyccl_params); pyccl.linear_matter_power(cosmo, k*cosmo['h'], a_pk)",
                                      'number': 2}

    cosmo = Cosmology(**params)
    pyccl_cosmo = pyccl.Cosmology(**pyccl_params)
    ba_class = cosmo.get_background('class')
    fo_class = cosmo.get_fourier('class')
    d['cosmoprimo background'] = {'stmt': "ba_class.comoving_radial_distance(z)", 'number': 100}
    d['pyccl background'] = {'stmt': "pyccl.background.comoving_radial_distance(pyccl_cosmo, a)", 'number': 100}
    d['cosmoprimo pk'] = {'stmt': "fo_class.pk_interpolator()(k, z_pk)", 'number': 2}
    d['pyccl pk'] = {'stmt': "pyccl.linear_matter_power(pyccl_cosmo, k*pyccl_cosmo['h'], a_pk)", 'number': 2}

    for key, value in d.items():
        dt = timeit.timeit(**value, globals={**globals(), **locals()}) / value['number'] * 1e3
        print('{} takes {:.3f} milliseconds'.format(key, dt))


def test_repeats():
    import timeit
    params = {'sigma8': 0.8, 'Omega_cdm': 0.28, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96, 'm_ncdm': 0.1, 'neutrino_hierarchy': 'normal'}
    cosmo = Cosmology(**params)
    fo_class = cosmo.get_fourier('class')
    d = {}
    for section in ['background', 'fourier']:
        d['init {}'.format(section)] = {'stmt': "c = Cosmology(**params); c.get_{}('class')".format(section), 'number': 2}
        d['get {}'.format(section)] = {'stmt': "cosmo.get_{}()".format(section), 'number': 100}

    for key, value in d.items():
        dt = timeit.timeit(**value, globals={**globals(), **locals()}) / value['number'] * 1e3
        print('{} takes {: .3f} milliseconds'.format(key, dt))


def test_neutrinos():
    from cosmoprimo import constants
    from cosmoprimo.cosmology import _compute_ncdm_momenta

    T_eff = constants.TCMB * constants.TNCDM_OVER_CMB
    pncdm = _compute_ncdm_momenta(T_eff, 1e-14, z=0, epsrel=1e-7, out='p')
    rhoncdm = _compute_ncdm_momenta(T_eff, 1e-14, z=0, epsrel=1e-7, out='rho')
    assert np.allclose(3. * pncdm, rhoncdm, rtol=1e-6)

    for m_ncdm in [0.06, 0.1, 0.2, 0.4]:
        # print(_compute_ncdm_momenta(T_eff, m_ncdm, z=0, out='rho'), _compute_ncdm_momenta(T_eff, m_ncdm, z=0, out='p'))
        omega_ncdm = _compute_ncdm_momenta(T_eff, m_ncdm, z=0, out='rho') / constants.rho_crit_over_Msunph_per_Mpcph3
        assert np.allclose(omega_ncdm, m_ncdm / 93.14, rtol=1e-3)
        domega_over_dm = _compute_ncdm_momenta(T_eff, m_ncdm, out='drhodm', z=0) / constants.rho_crit_over_Msunph_per_Mpcph3
        assert np.allclose(domega_over_dm, 1. / 93.14, rtol=1e-3)

    for m_ncdm in [0.06, 0.1, 0.2, 0.4]:
        cosmo = Cosmology(m_ncdm=m_ncdm)
        # print(m_ncdm, cosmo['Omega_ncdm'], sum(cosmo['m_ncdm'])/(93.14*cosmo['h']**2))
        assert np.allclose(cosmo['Omega_ncdm'], sum(cosmo['m_ncdm']) / (93.14 * cosmo['h']**2), rtol=1e-3)
        cosmo = Cosmology(Omega_ncdm=cosmo['Omega_ncdm'])
        assert np.allclose(cosmo['m_ncdm'], m_ncdm)

    m = 0.06
    z = 0.01
    niterations = 100
    z = np.linspace(0., 1., niterations)
    import time
    t0 = time.time()
    #for i in range(niterations): _compute_ncdm_momenta(T_eff, m, z)
    toret = _compute_ncdm_momenta(T_eff, m, z, method='quad')
    toret2 = _compute_ncdm_momenta(T_eff, m, z, method='laguerre')
    print((toret2 - toret) / toret)

    import time
    t0 = time.time()
    toret2 = _compute_ncdm_momenta(T_eff, m, z)
    print(time.time() - t0, toret)


def test_clone():

    cosmo = Cosmology(omega_cdm=0.2, engine='class')
    engine = cosmo.engine

    for factor in [1., 1.1]:
        cosmo_clone = cosmo.clone(omega_cdm=cosmo['omega_cdm'] * factor)
        assert type(cosmo_clone.engine) == type(engine)
        assert cosmo_clone.engine is not engine
        z = np.linspace(0.5, 2., 100)
        test = np.allclose(cosmo_clone.get_background().comoving_radial_distance(z), cosmo.get_background().comoving_radial_distance(z))
        if factor == 1:
            assert test
        else:
            assert not test
        cosmo_clone = cosmo.clone(base='internal', sigma8=cosmo.sigma8_m * factor)
        assert np.allclose(cosmo_clone.get_fourier().sigma_rz(8, 0, of='delta_m'), cosmo.sigma8_m * factor, rtol=1e-4)  # interpolation error
        cosmo_clone = cosmo.clone(base='internal', h=cosmo.h * factor)
        assert np.allclose(cosmo_clone.Omega0_m, cosmo.Omega0_m)
        cosmo_clone = cosmo.clone(base='input', h=cosmo.h * factor)
        assert np.allclose(cosmo_clone.Omega0_cdm, cosmo.Omega0_cdm / factor**2)


def test_shortcut():
    cosmo = Cosmology()
    z = [0.1, 0.3]
    with pytest.raises(AttributeError):
        d = cosmo.comoving_radial_distance(z)
    assert 'tau_reio' not in dir(cosmo)
    cosmo.set_engine('class')
    assert 'tau_reio' in dir(cosmo)
    assert 'table' not in dir(cosmo)
    assert 'table' in dir(Fourier(cosmo))
    d = cosmo.comoving_radial_distance(z)
    assert np.all(d == cosmo.get_background().comoving_radial_distance(z))
    assert cosmo.gauge == 'synchronous'  # default
    cosmo.set_engine('class', gauge='newtonian')
    assert cosmo.gauge == 'newtonian'


def test_theta_cosmomc():

    cosmo = Cosmology(engine='camb')
    from cosmoprimo.cosmology import _compute_rs_cosmomc

    rs, zstar = _compute_rs_cosmomc(cosmo.Omega0_b * cosmo.h**2, cosmo.Omega0_m * cosmo.h**2, cosmo.hubble_function)
    theta_cosmomc = rs * cosmo.h / cosmo.comoving_angular_distance(zstar)
    assert np.allclose(theta_cosmomc, cosmo.theta_cosmomc, atol=0., rtol=2e-6)


def test_bisect():

    from scipy import optimize

    def clone(fiducial, params):

        theta_MC_100 = params.pop('theta_MC_100', None)
        fiducial = fiducial.clone(base='input', **params)

        if theta_MC_100 is not None:
            if 'h' in params:
                raise ValueError('Cannot provide both theta_MC_100 and h')

            # With cosmo.get_thermodynamics().theta_cosmomc
            # Typically takes 18 iterations and ~0.8 s
            # The computation of the thermodynamics is the most time consuming
            # The 'theta_cosmomc' call takes ~0.1 s and is accurate within 3e-6 (rel.), ~1% of Planck errors
            def f(h):
                cosmo = fiducial.clone(base='input', h=h)
                return theta_MC_100 - 100. * cosmo['theta_cosmomc']
                #return theta_MC_100 - 100. * cosmo.get_thermodynamics().theta_cosmomc

            limits = [0.1, 2.]  # h-limits
            xtol = 1e-6  # 1 / 5000 of Planck errors
            rtol = xtol
            try:
                h = optimize.bisect(f, *limits, xtol=xtol, rtol=rtol, disp=True)
            except ValueError as exc:
                raise ValueError('Could not find proper h value in the interval that matches theta_MC_100 = {:.4f} with [f({:.3f}), f({:.3f})] = [{:.4f}, {:.4f}]'.format(theta_MC_100, *limits, *list(map(f, limits)))) from exc
            cosmo = fiducial.clone(base='input', h=h)

        return cosmo

    cosmo = clone(Cosmology(engine='class'), dict(theta_MC_100=1.04092))
    #from cosmoprimo.fiducial import DESI
    #cosmo = clone(DESI(engine='class'), dict(theta_MC_100=1.04092))
    print(cosmo.get_thermodynamics().theta_cosmomc)
    cosmo2 = Cosmology(engine='class').solve('h', 'theta_MC_100', 1.04092)
    assert np.allclose(cosmo2['h'], cosmo['h'])


    def clone(fiducial, params):

        theta = params.pop('theta_star', None)
        fiducial = fiducial.clone(base='input', **params)

        if theta is not None:
            if 'h' in params:
                raise ValueError('Cannot provide both theta_star and h')

            def f(h):
                cosmo = fiducial.clone(base='input', h=h)
                return 100. * (theta - cosmo.get_thermodynamics().theta_star)

            limits = [0.6, 0.9]  # h-limits
            xtol = 1e-6  # 1 / 5000 of Planck errors
            rtol = xtol
            try:
                h = optimize.bisect(f, *limits, xtol=xtol, rtol=rtol, disp=True)
            except ValueError as exc:
                raise ValueError('Could not find proper h value in the interval that matches theta_star = {:.4f} with [f({:.3f}), f({:.3f})] = [{:.4f}, {:.4f}]'.format(theta, *limits, *list(map(f, limits)))) from exc
            cosmo = fiducial.clone(base='input', h=h)

        return cosmo


    from cosmoprimo.fiducial import DESI
    cosmo = clone(DESI(engine='class'), dict(theta_star=0.0104))
    print(cosmo.get_thermodynamics().theta_star)
    cosmo2 = DESI(engine='class').solve('h', lambda cosmo: 100. * cosmo.get_thermodynamics().theta_star, target=100. * 0.0104, limits=[0.6, 0.9], xtol=1e-6, rtol=1e-6)
    assert np.allclose(cosmo2['h'], cosmo['h'])


def test_isitgr(plot=False):
    cosmo_camb = Cosmology(engine='camb')
    try:
        cosmo = Cosmology(engine='isitgr')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=5e-3)

    cosmo = Cosmology(engine='isitgr', MG_parameterization='mueta', E11=-0.5, E22=-0.5, extra_params=dict(AccuracyBoost=1.1))
    assert not np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=5e-3)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='isitgr')
    cosmo['Q0']
    assert 'Q0' in cosmo.get_default_params()
    assert 'Q0' in cosmo.get_default_parameters()

    if plot:
        z = 1.
        k = np.linspace(0.001, 0.2, 100)
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for kwargs in [{}, {'mu0': -0.5, 'Sigma0': 0.}, {'mu0': -0.5, 'Sigma0': 1.}]:
            pk = Cosmology(engine='isitgr', MG_parameterization='muSigma', **kwargs).get_fourier().pk_interpolator(of='delta_cb').to_1d(z=z)
            #ax.plot(k,  k * pk(k), label=str(kwargs))
            k = pk.k; ax.loglog(k,  pk(k), label=str(kwargs))
        ax.legend()
        plt.show()


def test_axiclass():

    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='axiclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'scf_potential': 'axion', 'n_axion': 2.6, 'log10_axion_ac': -3.531, 'fraction_axion_ac': 0.1, 'scf_parameters': [2.72, 0.0], 'scf_evolve_as_fluid': False,
              'scf_evolve_like_axionCAMB': False, 'attractor_ic_scf': False, 'compute_phase_shift': False, 'include_scf_in_delta_m': True, 'include_scf_in_delta_cb': True}
    cosmo = Cosmology(engine='axiclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='axiclass', **params)
    cosmo['log10_axion_ac']


def test_mochiclass():
    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='mochiclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1, 'gravity_model': 'brans dicke', 'parameters_smg': [0.7, 50, 1., 1.e-1],
              'skip_stability_tests_smg': 'no', 'a_min_stability_test_smg': 1e-6}
    cosmo = Cosmology(engine='mochiclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), cosmo.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='mochiclass', **params)
    cosmo['parameters_smg']
    cosmo.get_fourier().pk_interpolator(of='theta_cb')


def test_negnuclass():
    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='negnuclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'m_ncdm': -0.4}
    cosmo = Cosmology(engine='negnuclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), cosmo.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    from matplotlib import pyplot as plt
    ax = plt.gca()
    for m_ncdm in [-0.04, 0.06]:
        params.update(m_ncdm=m_ncdm)
        cosmo = DESI(engine='negnuclass', **params)
        pk = cosmo.get_fourier().pk_interpolator(of='theta_cb')
        ax.loglog(pk.k, pk(pk.k, z=1.))
    plt.show()


def test_neff():
    for m_ncdm in [[], [0.] * 3]:
        cosmo = Cosmology(engine='class', m_ncdm=m_ncdm)
        print(cosmo.Omega0_r)
        cosmo = Cosmology(engine='camb', m_ncdm=m_ncdm)
        print(cosmo.Omega0_r)


def test_error():

    with pytest.raises(CosmologyInputError):
        cosmo = Cosmology(Omega_m=-0.1)


def test_precompute_ncdm():
    from cosmoprimo.cosmology import _precompute_ncdm_momenta, _compute_ncdm_momenta
    import time
    t0 = time.time()
    cache = _precompute_ncdm_momenta()
    print(time.time() - t0)
    for m_ncdm in [0.06, 0.2, 0.5, 1.]:
        T_eff = constants.TCMB * constants.TNCDM_OVER_CMB * 0.9
        z = np.linspace(0., 10., 100)
        for out in ['p', 'rho', 'drhodm']:
            assert np.allclose(cache[out](m_ncdm, z, T_eff=T_eff), _compute_ncdm_momenta(T_eff, m_ncdm, z, out=out), rtol=1e-5, atol=0.)


def plot_z_sampling():
    from cosmoprimo.fiducial import DESI

    cosmo = DESI()

    from matplotlib import pyplot as plt
    z = cosmo.get_background().table()['z'][::-1]
    z2 = np.insert(np.logspace(-8, 2, 2048), 0, 0.)

    plt.plot(z[z < 100.])
    plt.plot(z2)
    plt.show()

    z = cosmo.get_fourier().table()[1][::-1]
    z2 = np.linspace(0., 10**0.5, 30)**2
    plt.plot(z)
    plt.plot(z2)
    plt.show()


def test_jax():
    import time
    from jax import numpy as jnp
    from jax import jit, jacfwd
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.cosmology import DefaultBackground, _cache, _precompute_ncdm_momenta

    from cosmoprimo.bbks import Background
    cosmo = Cosmology(neutrino_hierarchy='normal', m_ncdm=0.1, engine='bbks')
    cosmo.clone(m_ncdm=0.1).get_background().comoving_radial_distance(1.)
    t0 = time.time()
    n = 10
    for m_ncdm in np.linspace(0.1, 0.2, n):
        cosmo.clone(m_ncdm=float(m_ncdm)).get_background().comoving_radial_distance(1.)
        #cosmo.get_background().comoving_radial_distance(1.)
    print((time.time() - t0) / n)

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', **params)
        return DefaultBackground(cosmo).growth_rate(1.)

    test_jit = jit(test)
    print(test_jit(dict(m_ncdm=0.1)))
    t0 = time.time()
    n = 10
    for m_ncdm in np.linspace(0.01, 0.1, n):
        test_jit(dict(m_ncdm=float(m_ncdm)))
    print((time.time() - t0) / n)

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', **params)
        return cosmo

    test_jit = jit(test)
    print(test_jit(dict(m_ncdm=np.array(0.1))))
    assert np.allclose(test_jit(dict(m_ncdm=np.array(0.2)))['m_ncdm_tot'], 0.2)

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', engine='bbks', **params)
        return cosmo._engine

    test_jit = jit(test)
    print(test_jit(dict(m_ncdm=np.array(0.1))))

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', **params)
        return cosmo.get_background(engine='bbks')

    test_jit = jit(test)
    print(test_jit(dict(m_ncdm=np.array(0.1))).comoving_radial_distance(1.))

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', engine='bbks', **params)
        pk = cosmo.get_fourier().pk_interpolator()
        return pk

    test_jit = jit(jacfwd(lambda params: test(params)(20., z=1.)))
    test_jit(dict(m_ncdm=np.array(0.1)))

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', engine='bbks', **params)
        pk = cosmo.get_fourier().pk_interpolator()
        return pk, pk.to_1d(z=1.), pk.to_xi(), pk.to_xi().to_1d(z=0.)

    test_jit = jit(test)
    tmp = test_jit(dict(m_ncdm=np.array(0.1)))
    print(tmp[0](0.1, z=0.), tmp[1](0.1), tmp[2](20., z=1.), tmp[3](20.))

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', **params)
        background = DefaultBackground(cosmo)
        return background.comoving_radial_distance(1.)

    #print(test(1.), test(1.1))
    test_jit = jit(test)
    test_jit(dict(m_ncdm=np.array(0.1)))
    n = 20
    t0 = time.time()
    for value in np.linspace(0.01, 0.2, n):
        test_jit(dict(m_ncdm=value))
    print((time.time() - t0) / n)

    def test(theta_MC_100):
        cosmo = DESI(engine='eisenstein_hu').solve('h', 'theta_MC_100', target=theta_MC_100)
        return cosmo.comoving_radial_distance(1.)

    test_jit = jit(test)
    test_jit(1.)
    n = 10
    t0 = time.time()
    for value in np.linspace(1., 1.1, n):
        test_jit(value)
    print((time.time() - t0) / n)

    test_jacfwd = jacfwd(test)
    assert not np.allclose(test_jacfwd(1.), 0.)

    def test(params):
        cosmo = Cosmology(neutrino_hierarchy='normal', **params)
        background = DefaultBackground(cosmo)
        return background.comoving_radial_distance(1.)

    #print(test(1.), test(1.1))
    test_jit = jit(test)
    test_jit(dict(m_ncdm=np.array(0.1)))
    n = 20
    t0 = time.time()
    for value in np.linspace(0.01, 0.2, n):
        test_jit(dict(m_ncdm=value))
    print((time.time() - t0) / n)

    test_jacfwd = jacfwd(test)
    assert not np.allclose(test_jacfwd(dict(m_ncdm=jnp.array(0.1)))['m_ncdm'], 0.)

    def test(params):
        cosmo = DESI(engine='eisenstein_hu', **params)
        z = jnp.linspace(0., 1., 10)
        return cosmo.comoving_radial_distance(z)

    test_jacfwd = jit(jacfwd(test))
    test_jacfwd(dict(Omega_m=0.3, w0_fld=-1., wa_fld=0.))
    n = 20
    list_params = [dict(Omega_m=0.3, w0_fld=-1., wa_fld=0.3)] * n
    t0 = time.time()
    for params in list_params:
        test_jacfwd(params)
    print((time.time() - t0) / n)

    def test(params):
        cosmo = DESI(engine='eisenstein_hu', **params)
        z = jnp.linspace(0., 1., 10)
        return cosmo.sigma8_z(z)

    test_jit = jit(test)
    test_jit(dict(Omega_m=0.3, logA=3., w0_fld=-1., wa_fld=0.))
    test_jacfwd = jacfwd(test)
    assert not np.allclose(test_jacfwd(dict(Omega_m=0.3, logA=3.))['logA'], 0.)


def test_interp():
    from matplotlib import pyplot as plt
    from jax import numpy as jnp

    from cosmoprimo.fiducial import DESI
    from cosmoprimo.cosmology import BaseBackground, DefaultBackground
    from cosmoprimo.jax import odeint, Interpolator1D

    z = jnp.geomspace(0.0001, 1000., 10000)


    if 0:
        def ncdm_ref(self, name, z):
            func = getattr(BaseBackground, name)
            zc = 1. / np.logspace(-8, 0., 2000)[::-1] - 1.
            tmp = Interpolator1D(zc, func(self, zc).T)  # interpolation along axis = 0
            return tmp(z)

        def ncdm_1(self, name, z):
            func = getattr(BaseBackground, name)
            zc = 1. / np.logspace(-8, 0., 120)[::-1] - 1.
            tmp = Interpolator1D(zc, func(self, zc).T)  # interpolation along axis = 0
            return tmp(z)

        cosmo = DESI(h=jnp.array(0.7), engine=None)
        ba = DefaultBackground(cosmo)

        ax = plt.gca()
        ax.plot(z, jnp.abs(ncdm_1(ba, 'rho_ncdm', z) / ncdm_ref(ba, 'rho_ncdm', z) - 1.), label='rho')
        ax.plot(z, jnp.abs(ncdm_1(ba, 'p_ncdm', z) / ncdm_ref(ba, 'p_ncdm', z) - 1.), label='p')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    if 1:
        def time_ref(self, z):
            def integrand(y, z):
                return constants.c / 1e3 / (1. + z) / (100. * self.efunc(z))

            zc = 1. / np.logspace(-8, 0., 1000)[::-1] - 1.
            tmp = odeint(integrand, 0., zc)
            tmp = Interpolator1D(zc, (tmp[-1] - tmp) / self.h / constants.gigayear_over_megaparsec)
            return tmp(z)

        def time_1(self, z):
            def integrand(y, z):
                return constants.c / 1e3 / (1. + z) / (100. * self.efunc(z))

            zc = 1. / np.logspace(-8, 0., 300)[::-1] - 1.
            tmp = odeint(integrand, 0., zc)
            tmp = Interpolator1D(zc, (tmp[-1] - tmp) / self.h / constants.gigayear_over_megaparsec)
            return tmp(z)

        def age_1(self, z):
            def integrand(y, z):
                return constants.c / 1e3 / (1. + z) / (100. * self.efunc(z))

            zc = 1. / np.logspace(-8, 0., 300)[::-1] - 1.
            tmp = odeint(integrand, 0., zc)
            tmp = (tmp[-1] - tmp[0]) / self.h / constants.gigayear_over_megaparsec
            return tmp

        cosmo = DESI(m_ncdm=0.2, h=jnp.array(0.7), engine=None)
        ba = DefaultBackground(cosmo)

        ax = plt.gca()
        print(np.abs(age_1(ba) / time_ref(ba, 0) - 1.))
        ax.plot(z, jnp.abs(time_1(ba, z) / time_ref(ba, z) - 1.), label='cubic')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    if 0:
        def comoving_radial_distance_ref(self, z):
            def integrand(y, z):
                return constants.c / 1e3 / (100. * self.efunc(z))

            zc = 1. / np.logspace(-1, 0., 1000)[::-1] - 1.
            tmp = odeint(integrand, 0., zc)
            tmp = Interpolator1D(zc, tmp)
            return tmp(z)

        def comoving_radial_distance_0(self, z):
            def integrand(y, loga):
                a = jnp.exp(loga)
                z = 1 / a - 1.
                return constants.c / 1e3 / (100. * self.efunc(z)) / a**2 * a

            ac = np.logspace(-3, 0., 256)
            tmp = odeint(integrand, 0., jnp.log(ac))
            tmp = tmp[-1] - tmp
            tmp = Interpolator1D(ac, tmp, k=1)
            return tmp(1. / (1. + z))

        def comoving_radial_distance_1(self, z):
            def integrand(y, z):
                return constants.c / 1e3 / (100. * self.efunc(z))

            zm = 0.3
            zc = np.concatenate([np.linspace(0., zm, 20)[:-1], 1. / np.geomspace(1e-4, 1. / (1 + zm), 100)[::-1] - 1.])
            tmp = odeint(integrand, 0., zc)
            tmp = Interpolator1D(zc, tmp)
            return tmp(z)

        def comoving_radial_distance_2(self, z):
            def integrand(y, z):
                return constants.c / 1e3 / (100. * self.efunc(z))

            def integral_desitter(z):
                return 2. - 2. / (1. + z)**0.5

            #zc = 1. / np.geomspace(1e-2, 1., 1000)[::-1] - 1.
            zm = 0.3
            zc = jnp.concatenate([jnp.linspace(0., zm, 100)[:-1], 1. / np.geomspace(1e-4, 1. / (1 + zm), 128)[::-1] - 1.])
            tmp = odeint(integrand, 0., zc)
            #tmp = jnp.where(zc > 0., tmp / integral_desitter(zc), 0.)
            tmp = Interpolator1D(zc, tmp, k=1)
            return tmp(z)# * integral_desitter(z)

        cosmo = DESI(m_ncdm=0.2, h=0.7)
        ref = cosmo.comoving_radial_distance(np.array(z))

        cosmo = DESI(m_ncdm=0.2, h=jnp.array(0.7), engine=None)
        ba = DefaultBackground(cosmo)

        ax = plt.gca()
        #ax.plot(z, comoving_radial_distance_0(ba, z) / comoving_radial_distance_ref(ba, z), label='loga')
        ax.plot(z, jnp.abs(comoving_radial_distance_ref(ba, z) / ref - 1.), label='class')
        ax.plot(z, jnp.abs(comoving_radial_distance_1(ba, z) / comoving_radial_distance_ref(ba, z) - 1.), label='cubic')
        #ax.plot(z, jnp.abs(comoving_radial_distance_2(ba, z) / comoving_radial_distance_ref(ba, z) - 1.), label='linear')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    if 0:
        def integral_desitter(z):
            #return constants.c / 1e3 / 100. * (2. - 2. / (1. + z)**0.5)
            return constants.c / 1e3 / 100. * z
        z = jnp.linspace(0.1, 2., 100)
        ref = comoving_radial_distance_0(ba, z)
        desitter = integral_desitter(z)
        ax = plt.gca()
        ax.plot(z, ref)
        ax.plot(z, desitter)
        plt.show()
        exit()


def test_default_background():
    import time
    import jax
    from cosmoprimo.cosmology import BaseBackground, DefaultBackground
    from cosmoprimo.fiducial import DESI

    z = np.linspace(0., 10., 100)

    params = {'m_ncdm': 0.2}
    ba_ref = DESI(**params).get_background()
    cosmo = DESI(**params, engine=None)
    ba = DefaultBackground(cosmo)

    def test(z, OmM0):

        def interp(k, x, y):
            from scipy.interpolate import CubicSpline
            inter = CubicSpline(x, y)
            return inter(k)

          #Getting f0
        def OmM(eta):
            return 1./(1. + ((1-OmM0)/OmM0)*np.exp(3*eta) )

        def f1(eta):
            return 2. - 3./2. * OmM(eta)

        def f2(eta):
            return 3./2. * OmM(eta)

        etaini = -6  #initial eta, early enough to evolve as EdS (D + \propto a)
        zfin = -0.99

        def etaofz(z):
            return np.log(1/(1 + z))

        etafin = etaofz(zfin)

        from scipy.integrate import odeint

        # differential eq.
        def Deqs(Df, eta):
            Df, Dprime = Df
            return [Dprime, f2(eta)*Df - f1(eta)*Dprime]

        # eta range and initial conditions
        eta = np.linspace(etaini, etafin, 1001)
        Df0 = np.exp(etaini)
        Df_p0 = np.exp(etaini)

        # solution
        Dplus, Dplusp = odeint(Deqs, [Df0, Df_p0], eta).T
        #print(Dplus, Dplusp)

        Dplusp_ = interp(etaofz(z), eta, Dplusp)
        Dplus_ = interp(etaofz(z), eta, Dplus)

        return Dplus_, Dplusp_ / Dplus_

    """
    test = test(z, OmM0=cosmo['Omega_m'])[0]
    print(ba.growth_factor(z, mass='m') / (test / test[0]))
    assert np.allclose(ba.growth_factor(z, mass='m'), (test / test[0]), rtol=1e-3, atol=1e-4)
    """
    assert np.allclose(ba_ref.growth_factor(z), ba.growth_factor(z, mass='cb'), rtol=1e-6, atol=1e-5)

    for name in ['time', 'comoving_radial_distance', 'Omega_ncdm', 'theta_cosmomc']:

        if name == 'theta_cosmomc':
            def ref(**params):
                cosmo = DESI(**params)
                return cosmo[name]

            def test(**params):
                cosmo = DESI(**params, engine='bbks')
                return cosmo[name]

        else:
            def ref(**params):
                cosmo = DESI(**params)
                return getattr(cosmo.get_background(), name)(z)

            def test(**params):
                cosmo = DESI(**params, engine=None)
                background = DefaultBackground(cosmo)
                #background = BaseBackground(cosmo)
                return getattr(background, name)(z)

        test_jit = jax.jit(test)
        list_params = [{'m_ncdm': 0.4}, {'m_ncdm': 0.4, 'w0_fld': -0.6, 'wa_fld': -1.}, {'m_ncdm': 5., 'w0_fld': -0.8, 'wa_fld': -0.5}]
        for params in list_params:
            ref(**params)
            test(**params)
            test_jit(**params)
            assert np.allclose(test(**params), ref(**params), rtol=1e-5, atol=1e-4), (name, test(**params) / ref(**params))
            assert np.allclose(test_jit(**params), ref(**params), rtol=1e-5, atol=1e-4)
        n = 10
        t0 = time.time()
        for params in list_params * n: test_jit(**params)
        dt_test = time.time() - t0
        t0 = time.time()
        for params in list_params * n: ref(**params)
        dt_ref = time.time() - t0
        print(dt_test / (len(list_params) * n), dt_ref / (len(list_params) * n))


def test_fk():

    def interp(k, x, y):
        from scipy.interpolate import CubicSpline
        inter = CubicSpline(x, y)
        return inter(k)

    def f_over_f0_EH(zev, k, OmM0, h, fnu, Nnu=3):
        """
        Routine to get f(k)/f0 and f0.
        f(k)/f0 is obtained following H&E (1998), arXiv:astro-ph/9710216
        f0 is obtained by solving directly the differential equation for the linear growth at large scales.

        Args:
            zev: redshift
            k: wave-number
            OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
            h = H0/100
            fnu: Omega_nu/OmM0
            Nnu: number of neutrinos
        Returns:
            f(k)/f0 (when 'EdSkernels = True' f(k)/f0 = 1)
            f0
        """
        eta = np.log(1 / (1 + zev))   #log of scale factor
        Neff = 3.046                   # effective number of neutrinos
        omrv = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff)) #rad: including neutrinos
        aeq = omrv/OmM0           #matter-radiation equality

        pcb = 5./4 - np.sqrt(1 + 24*(1 - fnu))/4     #neutrino supression
        c = 0.7
        theta272 = (1.00)**2                        # T_{CMB} = 2.7*(theta272)
        pf = (k * theta272)/(OmM0 * h**2)
        DEdS = np.exp(eta)/aeq                      #growth function: EdS cosmology

        fnunonzero = np.where(fnu != 0., fnu, 1.)
        yFS = 17.2*fnu*(1 + 0.488*fnunonzero**(-7/6))*(pf*Nnu/fnunonzero)**2  #yFreeStreaming
        # pcb = 0. and yFS = 0. when fnu = 0.
        rf = DEdS/(1 + yFS)
        fFit = 1 - pcb/(1 + (rf)**c)                #f(k)/f0

        #Getting f0
        def OmM(eta):
            return 1./(1. + ((1-OmM0)/OmM0)*np.exp(3*eta) )

        def f1(eta):
            return 2. - 3./2. * OmM(eta)

        def f2(eta):
            return 3./2. * OmM(eta)

        etaini = -6  #initial eta, early enough to evolve as EdS (D + \propto a)
        zfin = -0.99

        def etaofz(z):
            return np.log(1/(1 + z))

        etafin = etaofz(zfin)

        from scipy.integrate import odeint

        # differential eq.
        def Deqs(Df, eta):
            Df, Dprime = Df
            return [Dprime, f2(eta)*Df - f1(eta)*Dprime]

        # eta range and initial conditions
        eta = np.linspace(etaini, etafin, 1001)
        Df0 = np.exp(etaini)
        Df_p0 = np.exp(etaini)

        # solution
        Dplus, Dplusp = odeint(Deqs, [Df0, Df_p0], eta).T
        #print(Dplus, Dplusp)

        Dplusp_ = interp(etaofz(zev), eta, Dplusp)
        Dplus_ = interp(etaofz(zev), eta, Dplus)
        f0 = Dplusp_/Dplus_

        return (k, fFit, f0)

    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI
    k = np.geomspace(0.0001, 2., 1000)
    z = 0.5
    ax = plt.gca()
    m_ncdm = 0.4
    cosmo = DESI(m_ncdm=m_ncdm) #, neutrino_hierarchy='degenerate')
    Omega0_m, h = cosmo.Omega0_m, cosmo.h
    fnu = cosmo.Omega0_ncdm_tot / Omega0_m
    _, fk, f0 = f_over_f0_EH(z, k, Omega0_m, h, fnu, Nnu=1)
    pk_dd_noncdm =  DESI(m_ncdm=0.).get_fourier().pk_interpolator(of='delta_cb').to_1d(z=z)(k)
    for i, engine in enumerate(['class', 'camb']):
        cosmo = DESI(engine=engine, m_ncdm=m_ncdm)
        fo = cosmo.get_fourier()
        pk_tt = fo.pk_interpolator(of=('theta_cb', 'delta_cb'))(k, z=z)
        pk_dd = fo.pk_interpolator(of='delta_cb')(k, z=z)
        ratio = pk_tt / pk_dd
        print(ratio[0] / f0)
        ratio /= f0
        color = 'C{:d}'.format(i)
        ax.plot(k, ratio, color=color, label=engine)
        ratio = pk_dd / pk_dd_noncdm
        ratio /= ratio[0]
        #ax.plot(k, ratio, color=color, linestyle='--')
    ax.plot(k, fk, label='EH')
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$P_{\theta\theta} / P_{\delta\delta}$')
    ax.set_xscale('log')
    ax.legend()
    ax.grid()
    plt.savefig('test_fk_m_ncdm_{:.1f}.png'.format(m_ncdm))
    plt.close(plt.gcf())

    ax = plt.gca()
    _, fk, f0 = f_over_f0_EH(z, k, Omega0_m, h, fnu, Nnu=3)
    for i, engine in enumerate(['class', 'camb']):
        color = 'C{:d}'.format(i)
        cosmo = DESI(engine=engine, m_ncdm=m_ncdm, neutrino_hierarchy='degenerate')
        fo = cosmo.get_fourier()
        pk_tt = fo.pk_interpolator(of='theta_cb')(k, z=z)
        pk_dd = fo.pk_interpolator(of='delta_cb')(k, z=z)
        ratio = np.sqrt(pk_tt / pk_dd)
        ratio /= f0
        ax.plot(k, ratio, color=color, label=engine)
        dz = 1e-2
        hdz = dz / 2.
        pk_interp = fo.pk_interpolator(of='delta_cb')
        ratio = - np.log(pk_interp(k, z + hdz) / pk_interp(k, z - hdz)) / dz * (1. + z) / 2.
        print(ratio[0] / f0)
        ratio /= f0
        #ax.plot(k, ratio, color=color, linestyle='--')
    ax.plot(k, fk, color='C2', label='EH')
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$P_{\delta\theta} / P_{\delta\delta}$')
    ax.set_xscale('log')
    ax.legend()
    ax.grid()
    plt.savefig('test_fk_dpdd_m_ncdm_{:.1f}.png'.format(m_ncdm))
    plt.close(plt.gcf())

    z = np.linspace(0., 1., 4)
    ax = plt.gca()
    cosmo = DESI(engine='class', m_ncdm=m_ncdm)
    fo = cosmo.get_fourier()
    pk_tt = fo.pk_interpolator(of='theta_cb')(k, z=z)
    pk_dd = fo.pk_interpolator(of='delta_cb')(k, z=z)
    for iz, zz in enumerate(z):
        color = 'C{:d}'.format(iz)
        ratio = pk_dd[..., iz] / pk_dd[..., 0]
        ax.plot(k, ratio / ratio[0], color=color, label='$z = {:.2f}$'.format(zz))
        ratio = pk_tt[..., iz] / pk_tt[..., 0]
        ax.plot(k, ratio / ratio[0], color=color, linestyle='--')
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$P_{XX}(k, z) / P_{XX}(k, z=0)$')
    ax.legend()
    plt.savefig('test_pkz_m_ncdm_{:.1f}.png'.format(m_ncdm))
    plt.close(plt.gcf())


def test_emu():
    from cosmoprimo import Cosmology
    cosmo = Cosmology(logA=3., engine='capse')
    cosmo.lensed_cl()
    print(cosmo.rs_drag)

    cosmo = Cosmology(logA=3., engine='cosmopower_bolliet2023')
    cosmo.lensed_cl()
    print(cosmo.rs_drag)


def test_rs():

    from cosmoprimo.fiducial import DESI

    for params in [{}, {'w0_fld': -0.5, 'wa_fld': 0.5}]:
        cosmo_class = DESI(**params, engine='class')
        cosmo_camb = DESI(**params, engine='camb')
        cosmo = DESI(**params, engine='bbks')
        rs = cosmo.get_background().rs(cosmo_camb.z_drag)
        print(params, cosmo_camb.rs_drag, cosmo_class.rs_drag / cosmo_camb.rs_drag, rs / cosmo_camb.rs_drag)


def test_bbn():
    from cosmoprimo.fiducial import DESI
    fiducial = DESI(extra_params={'sBBN file': 'bbn/sBBN.dat'})
    cosmo = DESI(extra_params={'sBBN file': 'bbn/sBBN_2017.dat'})
    test = DESI()
    assert test.rs_drag == cosmo.rs_drag
    print(fiducial.rs_drag / cosmo.rs_drag - 1.)


if __name__ == '__main__':

    #test_precompute_ncdm()
    #test_interp()
    test_jax()
    test_params()
    test_engine()
    for params in list_params:
        test_background(params)
        test_thermodynamics(params)
        test_primordial(params)
        test_harmonic(params)
        test_fourier(params)

    test_repeats()
    test_neutrinos()
    test_clone()
    test_shortcut()
    test_error()
    test_pk_norm()
    # plot_non_linear()
    # plot_primordial_power_spectrum()
    # plot_harmonic()
    # plot_matter_power_spectrum()
    # plot_non_linear_matter_power_spectrum()
    # plot_eisenstein_hu_nowiggle_variants()
    # test_external_camb()
    test_external_pyccl()
    test_bisect()
    test_isitgr()
    test_axiclass()
    test_mochiclass()
    test_negnuclass()
    test_default_background()
    #test_fk()