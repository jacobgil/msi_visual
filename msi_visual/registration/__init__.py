"""MSI ↔ H&E registration and LAB fusion overlays."""

from msi_visual.registration.he_msi import (
    EccStageConfig,
    RegistrationResult,
    apply_registration_to_he,
    default_registration_stages,
    letterbox_matrix,
    load_registration,
    register_he_to_msi,
    resolve_registration_path,
    save_registration,
    warp_he_to_msi,
)
from msi_visual.registration.lab_overlay import lab_overlay_he_l_mics_ab
from msi_visual.registration.paths import msi_sample_id, resolve_he_thumbnail, slide_key_from_npy

__all__ = [
    "EccStageConfig",
    "RegistrationResult",
    "apply_registration_to_he",
    "default_registration_stages",
    "lab_overlay_he_l_mics_ab",
    "letterbox_matrix",
    "load_registration",
    "msi_sample_id",
    "register_he_to_msi",
    "resolve_he_thumbnail",
    "resolve_registration_path",
    "save_registration",
    "slide_key_from_npy",
    "warp_he_to_msi",
]
