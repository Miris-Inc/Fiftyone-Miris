"""Auto-segmentation: Grounding DINO + SAM2 + per-pixel depth."""
from .pipeline import run_dino_sam2_pipeline

__all__ = ["run_dino_sam2_pipeline"]
