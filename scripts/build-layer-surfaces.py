#!/usr/bin/env python
"""Build compact browser-ready layer surfaces from stratigraphy NPZ output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def require_array(archive: np.lib.npyio.NpzFile, key: str, input_path: Path) -> np.ndarray:
    """Return a required array or raise a clear format error."""
    if key not in archive.files:
        raise ValueError(f"{input_path} is missing required array: {key}")
    return archive[key]


def spatial_coordinates(spatial_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Create index-based millimetre coordinates for the model grid."""
    if len(spatial_shape) == 1:
        return {"x_mm": np.arange(spatial_shape[0], dtype=np.float32)}
    if len(spatial_shape) == 2:
        y_size, x_size = spatial_shape
        return {
            "x_mm": np.arange(x_size, dtype=np.float32),
            "y_mm": np.arange(y_size, dtype=np.float32),
        }
    raise ValueError(
        "Only 2-D cross-section arrays (time, x) and 3-D surface arrays "
        "(time, y, x) are supported."
    )


def layer_ids_from(layer_id: np.ndarray) -> np.ndarray:
    """Return sorted non-negative layer IDs present in the stratigraphy archive."""
    finite_layer_id = layer_id[np.isfinite(layer_id)] if np.issubdtype(layer_id.dtype, np.floating) else layer_id
    unique_layer_ids = np.unique(finite_layer_id)
    return unique_layer_ids[unique_layer_ids >= 0].astype(np.int16)


def layer_step_bounds(layer_id: np.ndarray, layer_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the first and last timestep where each layer ID appears."""
    start_steps = np.full(layer_ids.shape, -1, dtype=np.int32)
    end_steps = np.full(layer_ids.shape, -1, dtype=np.int32)

    for index, current_layer_id in enumerate(layer_ids):
        timestep_matches = np.any(layer_id == current_layer_id, axis=tuple(range(1, layer_id.ndim)))
        matching_steps = np.flatnonzero(timestep_matches)
        if matching_steps.size:
            start_steps[index] = int(matching_steps[0])
            end_steps[index] = int(matching_steps[-1])

    return start_steps, end_steps


def build_layer_surfaces(input_path: Path) -> dict[str, np.ndarray]:
    """Convert timestep stratigraphy into compact per-layer browser surfaces."""
    with np.load(input_path) as archive:
        deposition = require_array(archive, "deposition_increment_mm", input_path)
        layer_id = require_array(archive, "layer_id", input_path)

        if deposition.shape != layer_id.shape:
            raise ValueError(
                "deposition_increment_mm and layer_id must have the same shape; "
                f"got {deposition.shape} and {layer_id.shape}."
            )
        if deposition.ndim not in (2, 3):
            raise ValueError(
                "Expected deposition_increment_mm shape (time, x) or (time, y, x); "
                f"got {deposition.shape}."
            )

        spatial_shape = deposition.shape[1:]
        layer_ids = layer_ids_from(layer_id)
        if layer_ids.size == 0:
            raise ValueError(f"{input_path} does not contain any non-negative layer IDs.")

        if "active_mask" in archive.files:
            active_mask = archive["active_mask"].astype(bool)
            if active_mask.shape != spatial_shape:
                raise ValueError(
                    f"active_mask shape {active_mask.shape} does not match spatial shape {spatial_shape}."
                )
        else:
            active_mask = np.ones(spatial_shape, dtype=bool)

        if "initial_dome_height_mm" in archive.files:
            base_height = archive["initial_dome_height_mm"].astype(np.float32)
            if base_height.shape != spatial_shape:
                raise ValueError(
                    "initial_dome_height_mm shape "
                    f"{base_height.shape} does not match spatial shape {spatial_shape}."
                )
        else:
            base_height = np.zeros(spatial_shape, dtype=np.float32)

        layer_thicknesses = []
        cumulative_thicknesses = []
        cumulative_thickness = np.zeros(spatial_shape, dtype=np.float32)

        for current_layer_id in layer_ids:
            layer_deposition = np.where(layer_id == current_layer_id, deposition, 0.0)
            layer_thickness = np.sum(layer_deposition, axis=0, dtype=np.float32)
            cumulative_thickness = cumulative_thickness + layer_thickness

            layer_thickness = layer_thickness.astype(np.float32)
            layer_thickness[~active_mask] = np.nan
            cumulative_snapshot = cumulative_thickness.astype(np.float32).copy()
            cumulative_snapshot[~active_mask] = np.nan

            layer_thicknesses.append(layer_thickness)
            cumulative_thicknesses.append(cumulative_snapshot)

        layer_thickness_mm = np.stack(layer_thicknesses).astype(np.float32)
        growth_thickness_mm = np.stack(cumulative_thicknesses).astype(np.float32)
        height_mm = growth_thickness_mm + base_height.astype(np.float32)
        height_mm[:, ~active_mask] = np.nan

        start_steps, end_steps = layer_step_bounds(layer_id, layer_ids)

        output = {
            **spatial_coordinates(spatial_shape),
            "layer_ids": layer_ids,
            "layer_start_step": start_steps,
            "layer_end_step": end_steps,
            "active_mask": active_mask,
            "base_height_mm": base_height,
            "height_mm": height_mm,
            "growth_thickness_mm": growth_thickness_mm,
            "layer_thickness_mm": layer_thickness_mm,
        }

        if "initial_dome_height_mm" in archive.files:
            output["initial_dome_height_mm"] = base_height

        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert stratigraphy NPZ output into compact browser-ready layer surfaces."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Existing stratigraphy NPZ file to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Browser-ready layer surface NPZ file to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_surfaces = build_layer_surfaces(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **layer_surfaces)

    layer_count = len(layer_surfaces["layer_ids"])
    spatial_shape = layer_surfaces["active_mask"].shape
    print(f"Saved {layer_count} layer surface(s) with grid shape {spatial_shape} to {args.output}")


if __name__ == "__main__":
    main()
