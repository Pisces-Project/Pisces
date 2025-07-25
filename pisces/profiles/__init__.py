"""Mathematical profiles for use in model construction.

This module provides a growing library of analytical and empirical profiles
commonly used in astrophysical modeling. Available profiles include density
profiles for dark matter halos, stellar systems, and gas distributions, with
support for both standard and generalized forms.

Subclasses are organized by profile type, with current support primarily
focused on spherical density profiles. Additional profile families will be
introduced in future versions.
"""

__all__ = [
    "profile_from_dict",
    "build_dynamic_profile_class",
    "BaseSphericalDensityProfile",
    "NFWDensityProfile",
    "HernquistDensityProfile",
    "EinastoDensityProfile",
    "SingularIsothermalDensityProfile",
    "CoredIsothermalDensityProfile",
    "PlummerDensityProfile",
    "DehnenDensityProfile",
    "JaffeDensityProfile",
    "BurkertDensityProfile",
    "MooreDensityProfile",
    "CoredNFWDensityProfile",
    "KingDensityProfile",
    "VikhlininDensityProfile",
    "AM06DensityProfile",
    "SNFWDensityProfile",
    "TNFWDensityProfile",
    "PseudoIsothermalDensityProfile",
    "DoublePowerLawDensityProfile",
    "CoredPowerLawDensityProfile",
    "GeneralizedNFWDensityProfile",
    "BetaModelDensityProfile",
    "BaseDiskDensityProfile",
    "ExponentialDiskDensityProfile",
    "SechSquaredDiskDensityProfile",
    "SechDiskDensityProfile",
    "FlaredDiskDensityProfile",
    "SersicProfile",
    "GaussianRingProfile",
    "CoreSersicProfile",
    "BaseSphericalTemperatureProfile",
    "IsothermalTemperatureProfile",
    "BetaModelTemperatureProfile",
    "DoubleBetaTemperatureProfile",
    "CoolingFlowTemperatureProfile",
    "AM06TemperatureProfile",
    "VikhlininTemperatureProfile",
    "BaseSphericalEntropyProfile",
]

from .base import build_dynamic_profile_class, profile_from_dict

# --- Density Profiles --- #
from .density import (
    AM06DensityProfile,
    BaseDiskDensityProfile,
    BaseSphericalDensityProfile,
    BetaModelDensityProfile,
    BurkertDensityProfile,
    CoredIsothermalDensityProfile,
    CoredNFWDensityProfile,
    CoredPowerLawDensityProfile,
    CoreSersicProfile,
    DehnenDensityProfile,
    DoublePowerLawDensityProfile,
    EinastoDensityProfile,
    ExponentialDiskDensityProfile,
    FlaredDiskDensityProfile,
    GaussianRingProfile,
    GeneralizedNFWDensityProfile,
    HernquistDensityProfile,
    JaffeDensityProfile,
    KingDensityProfile,
    MooreDensityProfile,
    NFWDensityProfile,
    PlummerDensityProfile,
    PseudoIsothermalDensityProfile,
    SechDiskDensityProfile,
    SechSquaredDiskDensityProfile,
    SersicProfile,
    SingularIsothermalDensityProfile,
    SNFWDensityProfile,
    TNFWDensityProfile,
    VikhlininDensityProfile,
)
from .entropy import BaseSphericalEntropyProfile
from .temperature import (
    AM06TemperatureProfile,
    BaseSphericalTemperatureProfile,
    BetaModelTemperatureProfile,
    CoolingFlowTemperatureProfile,
    DoubleBetaTemperatureProfile,
    IsothermalTemperatureProfile,
    VikhlininTemperatureProfile,
)
