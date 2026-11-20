"""Einstein Telescope detector configurations for bilby: triangle and 2L.

Geometry follows Branchesi et al. 2023, "Science with the Einstein Telescope: a
comparison of different designs" (arXiv:2303.15923, the ET "CoBA" study):

  sites      Sardinia (Sos Enattos)  lat 40 deg 31',        lon  9 deg 25'
             Meuse-Rhine             lat 50 deg 43' 23",    lon  5 deg 55' 14"
  triangle   one 10 km triangle (60 deg opening angle), placed in Sardinia
  2L         two L-shaped detectors with 15 km arms, one per site;
             relative orientation alpha, measured with respect to the local
             North at each site: alpha = 0 deg "aligned", 45 deg "misaligned"

bilby conventions used here: arm azimuths in degrees NORTH OF EAST
(counter-clockwise seen from above); y-arm = x-arm + 90 deg for an L,
+60 deg for the triangle. We point the Sardinia L's x-arm to local North and
rotate the Meuse-Rhine L by alpha. Only the RELATIVE angle is physical for the
2L comparison; the sign of the rotation is a convention.

Sensitivity (ASD): for a FAIR comparison use the CoBA curves for both designs
("HFLF cryogenic", 10 km for the triangle, 15 km for the 2L), available at
https://apps.et-gw.eu/tds/ql/?c=16492 (Annex files). Each curve is the ASD of
ONE interferometer of that design. Without a file, bilby's ET-D curve is used
for both (fine for testing the code, NOT for the comparison).

Usage::

    from pe.et_detectors import et_triangle, et_2l
    ifos = et_triangle(asd_file="asd/ET_10km_HFLF_cryo.txt")         # 3 interferometers
    ifos = et_2l(asd_file="asd/ET_15km_HFLF_cryo.txt", alpha_deg=45)   # 2 interferometers
    # then exactly as with InterferometerList(["ET"]):
    ifos.set_strain_data_from_zero_noise(sampling_frequency=FS, duration=T, start_time=t0)
"""
from __future__ import annotations

import warnings

import bilby
from bilby.gw.detector import (Interferometer, InterferometerList, PowerSpectralDensity,
                               TriangularInterferometer)

SITES = {
    "sardinia": dict(latitude=40 + 31 / 60, longitude=9 + 25 / 60),
    "meuse_rhine": dict(latitude=50 + 43 / 60 + 23 / 3600, longitude=5 + 55 / 60 + 14 / 3600),
}


def _psd(asd_file: str | None) -> PowerSpectralDensity:
    if asd_file is None:
        warnings.warn("no ASD file given: using bilby's ET-D curve (OK for tests, not for the "
                      "triangle-vs-2L comparison; use the CoBA curves)")
        return PowerSpectralDensity(psd_file="ET_D_psd.txt")
    return PowerSpectralDensity(asd_file=asd_file)


def et_triangle(asd_file: str | None = None, length_km: float = 10.0, site: str = "sardinia",
                xarm_azimuth: float = 0.0, minimum_frequency: float = 5.0,
                maximum_frequency: float = 2048.0) -> InterferometerList:
    """ET triangle: three co-located interferometers with 60 deg opening angle."""
    return TriangularInterferometer(
        name="ET", power_spectral_density=_psd(asd_file),
        minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency,
        length=length_km, elevation=0.0, xarm_azimuth=xarm_azimuth,
        yarm_azimuth=xarm_azimuth + 60.0, **SITES[site])


def et_2l(asd_file: str | None = None, length_km: float = 15.0, alpha_deg: float = 45.0,
          minimum_frequency: float = 5.0, maximum_frequency: float = 2048.0) -> InterferometerList:
    """ET 2L: one L in Sardinia, one in Meuse-Rhine, relative orientation alpha_deg
    (0 = aligned, 45 = misaligned; Branchesi et al. 2023)."""
    psd = _psd(asd_file)
    north = 90.0                                            # local North, in degrees north of East
    ifos = []
    for name, site, x in (("ETS", "sardinia", north), ("ETMR", "meuse_rhine", north - alpha_deg)):
        ifos.append(Interferometer(
            name=name, power_spectral_density=psd,
            minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency,
            length=length_km, elevation=0.0, xarm_azimuth=x % 360, yarm_azimuth=(x + 90.0) % 360,
            **SITES[site]))
    return InterferometerList(ifos)


def sky_averaged_response(ifos: InterferometerList, n: int = 20000, seed: int = 0) -> float:
    """Average over sky position, polarisation angle and time of sum_i (F+^2 + Fx^2):
    a geometry-only figure of merit (0.4 for one L; 3 x 0.75 x 0.4 = 0.9 for a triangle)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 2 * np.pi, n); dec = np.arcsin(rng.uniform(-1, 1, n))
    psi = rng.uniform(0, np.pi, n); t = 1_000_000_000 + rng.uniform(0, 86164, n)
    tot = 0.0
    for ifo in ifos:
        fp = np.array([ifo.antenna_response(r, d, tt, p, "plus") for r, d, tt, p in zip(ra, dec, t, psi)])
        fc = np.array([ifo.antenna_response(r, d, tt, p, "cross") for r, d, tt, p in zip(ra, dec, t, psi)])
        tot = tot + fp ** 2 + fc ** 2
    return float(np.mean(tot))


if __name__ == "__main__":
    # quick self-check:  python -m pe.et_detectors
    for label, ifos in (("triangle 10 km, Sardinia", et_triangle()),
                        ("2L 15 km, alpha = 45", et_2l(alpha_deg=45)),
                        ("2L 15 km, alpha = 0", et_2l(alpha_deg=0))):
        print(f"{label:26s}: {len(ifos)} interferometers, sky-averaged sum(F+^2+Fx^2) = "
              f"{sky_averaged_response(ifos, n=4000):.3f}")
