#!/usr/bin/env python
"""Build an interactive Plotly render of the domed stromatolite surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "output" / "3d-circular-domed-layer-surfaces.npz"


def require_array(archive: np.lib.npyio.NpzFile, key: str, input_path: Path) -> np.ndarray:
    """Return a required NPZ array or raise a clear format error."""
    if key not in archive.files:
        raise ValueError(f"{input_path} is missing required array: {key}")
    return archive[key]


def load_surface_grid(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the final surface from a browser-ready layer surface NPZ."""
    with np.load(input_path) as archive:
        height = require_array(archive, "height_mm", input_path)

        if height.ndim == 3:
            z_grid = height[-1].astype(np.float32)
        elif height.ndim == 2:
            z_grid = height.astype(np.float32)
        else:
            raise ValueError(
                f"{input_path} height_mm must have shape (layers, y, x) or (y, x); "
                f"got {height.shape}."
            )

        y_size, x_size = z_grid.shape
        x_values = (
            archive["x_mm"].astype(np.float32)
            if "x_mm" in archive.files
            else np.arange(x_size, dtype=np.float32)
        )
        y_values = (
            archive["y_mm"].astype(np.float32)
            if "y_mm" in archive.files
            else np.arange(y_size, dtype=np.float32)
        )

        if x_values.shape != (x_size,):
            raise ValueError(
                f"{input_path} x_mm shape {x_values.shape} does not match surface width {x_size}."
            )
        if y_values.shape != (y_size,):
            raise ValueError(
                f"{input_path} y_mm shape {y_values.shape} does not match surface height {y_size}."
            )

        if "active_mask" in archive.files:
            active_mask = archive["active_mask"].astype(bool)
            if active_mask.shape != z_grid.shape:
                raise ValueError(
                    f"{input_path} active_mask shape {active_mask.shape} "
                    f"does not match surface shape {z_grid.shape}."
                )
            z_grid = z_grid.copy()
            z_grid[~active_mask] = np.nan

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, z_grid


def build_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    clean_axes: bool,
    dark_mode: bool,
) -> go.Figure:
    """Create the interactive dome figure."""
    active_height = z_grid[np.isfinite(z_grid)]
    min_height = float(np.min(active_height))
    max_height = float(np.max(active_height))
    height_span = max_height - min_height

    # Add a little vertical headroom so the highest point is not visually clipped
    # by the scene bounds when the user rotates the dome.
    z_padding = max(height_span * 0.12, max_height * 0.025, 0.5)
    background_color = "#050505" if dark_mode else "white"
    text_color = "#f4efe6" if dark_mode else "#1f2937"
    grid_color = "rgba(255,255,255,0.20)" if dark_mode else "rgba(31,41,55,0.20)"
    axis_line_color = "rgba(255,255,255,0.45)" if dark_mode else "rgba(31,41,55,0.45)"
    contour_highlight = "#f6d36b" if dark_mode else "#4f2a0a"

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale="YlOrBr",
                colorbar={
                    "title": {"text": "Surface height (mm)", "font": {"color": text_color}},
                    "thickness": 18,
                    "len": 0.72,
                    "tickfont": {"color": text_color},
                },
                # Do not bridge over NaN cells; those gaps encode the circular
                # active growth domain from the underlying model output.
                connectgaps=False,
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "highlightcolor": contour_highlight,
                        "project_z": False,
                    }
                },
                hovertemplate=(
                    "x: %{x:.1f} mm<br>"
                    "y: %{y:.1f} mm<br>"
                    "height: %{z:.2f} mm"
                    "<extra></extra>"
                ),
                lighting={
                    "ambient": 0.48,
                    "diffuse": 0.72,
                    "roughness": 0.64,
                    "specular": 0.18,
                },
            )
        ]
    )

    # The same figure can be used for debugging with visible axes, or for a
    # cleaner publication view with axes hidden via --clean-axes.
    axis_style = {
        "showgrid": not clean_axes,
        "showline": not clean_axes,
        "showticklabels": not clean_axes,
        "zeroline": False,
        "title": "" if clean_axes else None,
        "backgroundcolor": "rgba(0,0,0,0)",
        "color": text_color,
        "gridcolor": grid_color,
        "linecolor": axis_line_color,
    }
    z_axis_style = {
        **axis_style,
        "range": [max(0.0, min_height - z_padding), max_height + z_padding],
        "title": "" if clean_axes else "Height above base (mm)",
    }
    if not clean_axes:
        axis_style["title"] = None

    fig.update_layout(
        title={
            "text": "Interactive 3-D circular domed stromatolite surface",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": text_color},
        },
        width=1100,
        height=780,
        paper_bgcolor=background_color,
        font={"color": text_color},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        scene={
            "xaxis": {**axis_style, "title": "" if clean_axes else "x-position (mm)"},
            "yaxis": {**axis_style, "title": "" if clean_axes else "y-position (mm)"},
            "zaxis": z_axis_style,
            "aspectmode": "manual",
            # Compress z slightly so the height field reads as a low stromatolite
            # dome rather than as an exaggerated spike.
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.55},
            "camera": {
                "eye": {"x": 1.15, "y": -1.25, "z": 0.68},
                "center": {"x": 0.0, "y": 0.0, "z": -0.08},
            },
        },
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT,
                        help="Layer surface NPZ to render")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="HTML file to write; defaults to input path with .html suffix")
    parser.add_argument("-c", "--clean-axes", action="store_true",
                        help="Hide axes, ticks and grid lines for publication-style presentation")
    parser.add_argument("-d", "--dark-mode", action="store_true",
                        help="Use a black background with light axis and colorbar text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else args.input.with_suffix(".html")

    x_grid, y_grid, z_grid = load_surface_grid(args.input)
    fig = build_figure(x_grid, y_grid, z_grid, clean_axes=args.clean_axes, dark_mode=args.dark_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Embed Plotly in the file so the render can be opened directly or published
    # as a standalone HTML artefact.
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved interactive dome render to {output_path}")


if __name__ == "__main__":
    main()
