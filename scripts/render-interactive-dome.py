#!/usr/bin/env python
"""Build an interactive Plotly render of the domed stromatolite surface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "output" / "3d-circular-domed-layer-surfaces.npz"


@dataclass(frozen=True)
class LayerMetadata:
    """Display metadata for one representative surface layer."""

    label: str
    layer_id: int
    start_step: int | None
    end_step: int | None
    max_height: float


def require_array(archive: np.lib.npyio.NpzFile, key: str, input_path: Path) -> np.ndarray:
    """Return a required NPZ array or raise a clear format error."""
    if key not in archive.files:
        raise ValueError(f"{input_path} is missing required array: {key}")
    return archive[key]


def optional_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected_length: int,
    dtype: type[np.generic],
) -> np.ndarray | None:
    """Load an optional one-dimensional metadata vector."""
    if key not in archive.files:
        return None

    values = archive[key].astype(dtype)
    if values.shape != (expected_length,):
        raise ValueError(
            f"{key} shape {values.shape} does not match layer count {expected_length}."
        )
    return values


def layer_label(layer_id: int, start_step: int | None, end_step: int | None) -> str:
    """Build a compact slider label for a layer."""
    label = f"Layer {layer_id}"
    if start_step is not None and end_step is not None:
        label += f" · steps {start_step}-{end_step}"
    elif end_step is not None:
        label += f" · step {end_step}"
    return label


def customdata_for_layer(metadata: LayerMetadata, shape: tuple[int, int]) -> np.ndarray:
    """Create hover metadata for a Plotly surface layer."""
    step_text = ""
    if metadata.start_step is not None and metadata.end_step is not None:
        step_text = f"steps {metadata.start_step}-{metadata.end_step}"
    elif metadata.end_step is not None:
        step_text = f"step {metadata.end_step}"

    customdata = np.empty((*shape, 3), dtype=object)
    customdata[:, :, 0] = metadata.layer_id
    customdata[:, :, 1] = step_text
    customdata[:, :, 2] = metadata.max_height
    return customdata


def finite_range(values: np.ndarray) -> tuple[float, float]:
    """Return a usable finite min/max range for a surface."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Surface layer does not contain any finite heights.")

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if min_value == max_value:
        padding = max(abs(max_value) * 0.01, 0.5)
        return min_value - padding, max_value + padding
    return min_value, max_value


def load_layer_surface_grids(
    input_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[LayerMetadata]]:
    """Load all representative surface layers from a browser-ready NPZ."""
    with np.load(input_path) as archive:
        height = require_array(archive, "height_mm", input_path)

        if height.ndim == 3:
            z_layers = height.astype(np.float32)
        elif height.ndim == 2:
            z_layers = height[np.newaxis, :, :].astype(np.float32)
        else:
            raise ValueError(
                f"{input_path} height_mm must have shape (layers, y, x) or (y, x); "
                f"got {height.shape}."
            )

        if z_layers.shape[0] == 0:
            raise ValueError(f"{input_path} does not contain any surface layers.")

        layer_count, y_size, x_size = z_layers.shape
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
            if active_mask.shape != (y_size, x_size):
                raise ValueError(
                    f"{input_path} active_mask shape {active_mask.shape} "
                    f"does not match surface shape {(y_size, x_size)}."
                )
            z_layers = z_layers.copy()
            z_layers[:, ~active_mask] = np.nan

        if not np.any(np.isfinite(z_layers)):
            raise ValueError(f"{input_path} does not contain any finite surface heights.")

        layer_ids = optional_vector(archive, "layer_ids", layer_count, np.int16)
        start_steps = optional_vector(archive, "layer_start_step", layer_count, np.int32)
        end_steps = optional_vector(archive, "layer_end_step", layer_count, np.int32)

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    metadata = []
    for index, z_layer in enumerate(z_layers):
        layer_id = int(layer_ids[index]) if layer_ids is not None else index
        start_step = int(start_steps[index]) if start_steps is not None else None
        end_step = int(end_steps[index]) if end_steps is not None else None
        max_height = float(np.nanmax(z_layer))
        metadata.append(
            LayerMetadata(
                label=layer_label(layer_id, start_step, end_step),
                layer_id=layer_id,
                start_step=start_step,
                end_step=end_step,
                max_height=max_height,
            )
        )
    return x_grid, y_grid, z_layers, metadata


def build_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_layers: np.ndarray,
    metadata: list[LayerMetadata],
    *,
    clean_axes: bool,
    dark_mode: bool,
) -> go.Figure:
    """Create the interactive layer-browsing dome figure."""
    final_layer_index = len(metadata) - 1
    initial_z_grid = z_layers[final_layer_index]
    min_height, max_height = finite_range(z_layers)
    initial_color_min, initial_color_max = finite_range(initial_z_grid)
    height_span = max_height - min_height

    # Add a little vertical headroom so the highest point is not visually clipped
    # by the scene bounds when the user rotates the dome.
    z_padding = max(height_span * 0.12, max_height * 0.025, 0.5)
    background_color = "#050505" if dark_mode else "white"
    text_color = "#f4efe6" if dark_mode else "#1f2937"
    grid_color = "rgba(255,255,255,0.20)" if dark_mode else "rgba(31,41,55,0.20)"
    axis_line_color = "rgba(255,255,255,0.45)" if dark_mode else "rgba(31,41,55,0.45)"
    contour_highlight = "#f6d36b" if dark_mode else "#4f2a0a"
    initial_metadata = metadata[final_layer_index]

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=initial_z_grid,
                cmin=initial_color_min,
                cmax=initial_color_max,
                customdata=customdata_for_layer(initial_metadata, initial_z_grid.shape),
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
                    "layer: %{customdata[0]}<br>"
                    "%{customdata[1]}<br>"
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
    fig.frames = [
        go.Frame(
            name=str(index),
            data=[
                go.Surface(
                    z=z_layer,
                    cmin=finite_range(z_layer)[0],
                    cmax=finite_range(z_layer)[1],
                    customdata=customdata_for_layer(layer_metadata, z_layer.shape),
                )
            ],
            traces=[0],
            layout={
                "title": {
                    "text": f"Interactive 3-D circular domed stromatolite surface · {layer_metadata.label}",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"color": text_color},
                }
            },
        )
        for index, (z_layer, layer_metadata) in enumerate(zip(z_layers, metadata, strict=True))
    ]

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
            "text": f"Interactive 3-D circular domed stromatolite surface · {initial_metadata.label}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": text_color},
        },
        width=1100,
        height=680,
        paper_bgcolor=background_color,
        font={"color": text_color},
        margin={"l": 20, "r": 20, "t": 60, "b": 70},
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
        sliders=[
            {
                "active": final_layer_index,
                "currentvalue": {
                    "prefix": "Selected ",
                    "font": {"color": text_color, "size": 15},
                    "xanchor": "left",
                },
                "len": 0.72,
                "x": 0.18,
                "y": 0.0,
                "pad": {"t": 28, "b": 6},
                "steps": [
                    {
                        "label": layer_metadata.label,
                        "method": "animate",
                        "args": [
                            [str(index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for index, layer_metadata in enumerate(metadata)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": -0.08,
                "xanchor": "left",
                "yanchor": "middle",
                "pad": {"t": 0, "r": 10},
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 450, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
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

    x_grid, y_grid, z_layers, metadata = load_layer_surface_grids(args.input)
    fig = build_figure(
        x_grid,
        y_grid,
        z_layers,
        metadata,
        clean_axes=args.clean_axes,
        dark_mode=args.dark_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Embed Plotly in the file so the render can be opened directly or published
    # as a standalone HTML artefact.
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved interactive dome render to {output_path}")


if __name__ == "__main__":
    main()
