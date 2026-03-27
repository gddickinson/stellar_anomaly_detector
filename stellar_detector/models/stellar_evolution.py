"""Stellar evolution models: main sequence, mass-luminosity, lifetime, isochrones.

Provides analytical approximations for stellar properties as functions of
mass, temperature, luminosity, and metallicity.
"""

from __future__ import annotations

import numpy as np


class StellarEvolutionModels:
    """Collection of standard stellar astrophysics models."""

    # Solar reference values
    SOLAR_ABS_MAG_G = 4.67
    SOLAR_TEFF = 5778.0
    SOLAR_MASS = 1.0

    def main_sequence_mag_from_color(self, bp_rp: float) -> float:
        """Expected absolute G magnitude for a main-sequence star given BP-RP color.

        Piecewise polynomial fit to the solar-metallicity main sequence
        from Gaia DR3 empirical CMD.
        """
        if bp_rp < -0.3:
            return -3.0 + 2.0 * bp_rp
        elif bp_rp < 0.5:
            return -1.0 + 5.0 * bp_rp
        elif bp_rp < 1.5:
            return 1.5 + 3.5 * (bp_rp - 0.5)
        elif bp_rp < 3.0:
            return 5.0 + 3.0 * (bp_rp - 1.5)
        else:
            return 9.5 + 2.0 * (bp_rp - 3.0)

    def main_sequence_mag_low_z(self, bp_rp: float) -> float:
        """Main sequence for low metallicity ([M/H] < -0.5). Brighter by ~0.5-1 mag."""
        return self.main_sequence_mag_from_color(bp_rp) - 0.7

    def main_sequence_mag_high_z(self, bp_rp: float) -> float:
        """Main sequence for high metallicity ([M/H] > 0.3). Fainter by ~0.3 mag."""
        return self.main_sequence_mag_from_color(bp_rp) + 0.3

    def mass_from_abs_mag(self, abs_mag: float) -> float:
        """Estimate stellar mass (solar units) from absolute G magnitude.

        Uses the inverse of the mass-luminosity relation: L ~ M^3.5
        Combined with M_bol - M_bol_sun = -2.5 * log10(L/L_sun).
        """
        luminosity_solar = 10 ** ((self.SOLAR_ABS_MAG_G - abs_mag) / 2.5)
        if luminosity_solar <= 0:
            return 0.1
        mass = luminosity_solar ** (1 / 3.5)
        return float(np.clip(mass, 0.08, 150.0))

    def mass_from_luminosity(self, luminosity_solar: float) -> float:
        """Mass-luminosity relation: M ~ L^(1/3.5) for main-sequence stars."""
        if luminosity_solar <= 0:
            return 0.1
        return float(np.clip(luminosity_solar ** (1 / 3.5), 0.08, 150.0))

    def luminosity_from_mass(self, mass_solar: float) -> float:
        """L ~ M^3.5 for main-sequence stars."""
        return float(mass_solar ** 3.5)

    def main_sequence_lifetime(
        self, mass_solar: np.ndarray | float, metallicity: np.ndarray | float = 0.0
    ) -> np.ndarray | float:
        """Expected main-sequence lifetime in Gyr.

        tau_MS = 10 * (M / M_sun)^{-2.5} Gyr, adjusted for metallicity:
        tau_adjusted = tau_base * (1.0 + 0.1 * [M/H])
        """
        mass = np.asarray(mass_solar, dtype=float)
        mass = np.clip(mass, 0.08, 150.0)
        z = np.asarray(metallicity, dtype=float)

        tau_base = 10.0 * mass ** (-2.5)
        tau_adjusted = tau_base * (1.0 + 0.1 * z)
        return np.clip(tau_adjusted, 0.001, 1000.0)

    def total_lifetime(self, mass_solar: float) -> float:
        """Total stellar lifetime including post-MS phases (approximate)."""
        ms = self.main_sequence_lifetime(mass_solar)
        if mass_solar < 0.5:
            return float(ms * 1.1)  # Low-mass stars: MS dominates
        elif mass_solar < 8.0:
            return float(ms * 1.15)  # Add ~15% for RGB + AGB
        else:
            return float(ms * 1.05)  # Massive stars: brief post-MS

    def zams_mag_from_teff(self, teff: float) -> float:
        """Zero-Age Main Sequence absolute magnitude from effective temperature.

        ZAMS is the faintest (highest mag) position a star occupies on the MS.
        """
        if teff <= 0:
            return 15.0
        log_t = np.log10(teff)
        # Empirical fit
        return float(-8.0 * log_t + 34.5)

    def tams_mag_from_teff(self, teff: float) -> float:
        """Terminal-Age Main Sequence absolute magnitude.

        TAMS is brighter (lower mag) than ZAMS by ~0.75-1.5 mag depending on mass.
        """
        zams = self.zams_mag_from_teff(teff)
        # Width of MS band increases for hotter (more massive) stars
        if teff > 10000:
            return zams - 1.5
        elif teff > 6000:
            return zams - 1.0
        else:
            return zams - 0.75

    def temperature_from_color(self, bp_rp: float) -> float:
        """Approximate effective temperature from Gaia BP-RP color.

        Calibration based on Gaia DR3 GSP-Phot results.
        """
        # Polynomial fit: log10(Teff) = a0 + a1*(BP-RP) + a2*(BP-RP)^2 + ...
        x = np.clip(bp_rp, -0.5, 5.0)
        log_t = 3.999 - 0.234 * x + 0.030 * x**2 - 0.005 * x**3
        return float(10 ** log_t)

    def color_from_temperature(self, teff: float) -> float:
        """Approximate BP-RP color from effective temperature (inverse of above)."""
        if teff <= 0:
            return 4.0
        log_t = np.log10(teff)
        # Rough inverse
        return float((3.999 - log_t) / 0.234)
