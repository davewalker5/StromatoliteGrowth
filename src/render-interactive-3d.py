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
    """Display metadata for one representative surface layer.

    :param label: Human-readable label used by the Plotly slider.
    :param layer_id: Model layer identifier represented by this surface.
    :param start_step: First model step associated with this layer, if available.
    :param end_step: Last model step associated with this layer, if available.
    :param max_height: Maximum height of this layer surface in millimetres.
    """

    label: str
    layer_id: int
    start_step: int | None
    end_step: int | None
    max_height: float


@dataclass(frozen=True)
class ColorSurface:
    """Surface-colour values and labels derived from the layer archive.

    :param values: Per-layer raster stack used as Plotly surface colours.
    :param title: Colourbar title shown beside the render.
    :param hover_label: Short label used for the coloured value in hover text.
    :param colorscale: Plotly colour scale name.
    :param symmetric: Whether to use a zero-centred colour range per layer.
    """

    values: np.ndarray
    title: str
    hover_label: str
    colorscale: str
    symmetric: bool


def require_array(archive: np.lib.npyio.NpzFile, key: str, input_path: Path) -> np.ndarray:
    """Return a required NPZ array or raise a clear format error.

    :param archive: Open NPZ archive being read.
    :param key: Required array name.
    :param input_path: Source path used in error messages.
    :return: The requested NumPy array.
    """
    if key not in archive.files:
        raise ValueError(f"{input_path} is missing required array: {key}")
    return archive[key]


def optional_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected_length: int,
    dtype: type[np.generic],
) -> np.ndarray | None:
    """Load an optional one-dimensional metadata vector.

    :param archive: Open NPZ archive being read.
    :param key: Optional array name.
    :param expected_length: Required vector length when the array exists.
    :param dtype: NumPy dtype used to cast the vector.
    :return: Cast metadata vector, or None when absent.
    """
    if key not in archive.files:
        return None

    # Optional metadata arrays must still align one-to-one with the layer stack
    # so slider labels cannot drift away from the surfaces they describe.
    values = archive[key].astype(dtype)
    if values.shape != (expected_length,):
        raise ValueError(
            f"{key} shape {values.shape} does not match layer count {expected_length}."
        )
    return values


def layer_label(layer_id: int, start_step: int | None, end_step: int | None) -> str:
    """Build a compact slider label for a layer.

    :param layer_id: Model layer identifier.
    :param start_step: First model step associated with the layer, if available.
    :param end_step: Last model step associated with the layer, if available.
    :return: Slider label containing layer and step information.
    """
    label = f"Layer {layer_id}"
    if start_step is not None and end_step is not None:
        label += f" · steps {start_step}-{end_step}"
    elif end_step is not None:
        label += f" · step {end_step}"
    return label


def customdata_for_layer(
    metadata: LayerMetadata,
    z_layer: np.ndarray,
    color_layer: np.ndarray,
) -> np.ndarray:
    """Create hover metadata for a Plotly surface layer.

    :param metadata: Layer-level display metadata.
    :param z_layer: Height raster used for the 3-D geometry.
    :param color_layer: Raster used for the surface colour values.
    :return: Per-cell customdata array consumed by the Plotly hover template.
    """
    step_text = ""
    if metadata.start_step is not None and metadata.end_step is not None:
        step_text = f"steps {metadata.start_step}-{metadata.end_step}"
    elif metadata.end_step is not None:
        step_text = f"step {metadata.end_step}"

    # Plotly carries this per-cell array through hover events. Repeating the
    # layer-level values across the grid keeps the hover template simple.
    customdata = np.empty((*z_layer.shape, 4), dtype=object)
    customdata[:, :, 0] = metadata.layer_id
    customdata[:, :, 1] = step_text
    customdata[:, :, 2] = metadata.max_height
    customdata[:, :, 3] = color_layer
    return customdata


def finite_range(values: np.ndarray) -> tuple[float, float]:
    """Return a usable finite min/max range for a surface.

    :param values: Numeric array that may contain NaN mask cells.
    :return: Finite minimum and maximum, padded when the range is flat.
    """
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Surface layer does not contain any finite heights.")

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if min_value == max_value:
        # Plotly needs a non-zero colour/axis interval even for a perfectly flat layer.
        padding = max(abs(max_value) * 0.01, 0.5)
        return min_value - padding, max_value + padding
    return min_value, max_value


def symmetric_finite_range(values: np.ndarray) -> tuple[float, float]:
    """Return a zero-centred finite colour range.

    :param values: Numeric array that may contain NaN mask cells.
    :return: Negative and positive limits centred on zero.
    """
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Surface layer does not contain any finite colour values.")

    limit = float(np.max(np.abs(finite_values)))
    if limit == 0.0:
        limit = 0.5
    return -limit, limit


def normalize_layer_array(values: np.ndarray, z_layers: np.ndarray, key: str, input_path: Path) -> np.ndarray:
    """Validate and normalize a layer array to match the height stack.

    :param values: Candidate layer raster or raster stack.
    :param z_layers: Normalized height stack used as the reference shape.
    :param key: Source array name used in error messages.
    :param input_path: Source path used in error messages.
    :return: Float32 array with shape matching z_layers.
    """
    if values.shape == z_layers.shape:
        return values.astype(np.float32)
    if values.ndim == 2 and z_layers.shape[0] == 1:
        return values[np.newaxis, :, :].astype(np.float32)

    raise ValueError(
        f"{input_path} {key} shape {values.shape} does not match height_mm shape {z_layers.shape}."
    )


def growth_anomaly(growth_layers: np.ndarray) -> np.ndarray:
    """Compute per-layer growth thickness relative to the active-layer mean.

    :param growth_layers: Growth-thickness raster stack.
    :return: Raster stack with each layer mean subtracted.
    """
    anomalies = growth_layers.copy()
    for index, layer in enumerate(growth_layers):
        # Subtract the mean independently for each layer so the colours show
        # local over/under-growth rather than the cumulative upward trend.
        anomalies[index] = layer - np.nanmean(layer)
    return anomalies


def build_color_surface(
    color_mode: str,
    growth_layers: np.ndarray,
    layer_thickness_layers: np.ndarray | None,
) -> ColorSurface:
    """Select the field used for surface colouring.

    :param color_mode: Colour mode requested by the command line.
    :param growth_layers: Growth-thickness raster stack.
    :param layer_thickness_layers: Optional selected-layer thickness raster stack.
    :return: Colour values and display labels for the selected mode.
    """
    # Geometry always comes from height_mm; this selection only controls the
    # surface colours, colourbar label, and hover value.
    if color_mode == "growth":
        return ColorSurface(
            values=growth_layers,
            title="Growth thickness (mm)",
            hover_label="growth",
            colorscale="YlOrBr",
            symmetric=False,
        )
    if color_mode == "growth-anomaly":
        return ColorSurface(
            values=growth_anomaly(growth_layers),
            title="Growth thickness anomaly (mm)",
            hover_label="growth anomaly",
            colorscale="RdBu_r",
            symmetric=True,
        )
    if color_mode == "layer-thickness":
        if layer_thickness_layers is None:
            raise ValueError("layer-thickness colour mode requires layer_thickness_mm in the input NPZ.")
        return ColorSurface(
            values=layer_thickness_layers,
            title="Layer thickness (mm)",
            hover_label="layer thickness",
            colorscale="YlOrBr",
            symmetric=False,
        )

    raise ValueError(f"Unsupported colour mode: {color_mode}")


def load_layer_surface_grids(
    input_path: Path,
    color_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ColorSurface, list[LayerMetadata]]:
    """Load all representative surface layers from a browser-ready NPZ.

    :param input_path: Browser-ready layer-surface NPZ file.
    :param color_mode: Colour mode used to derive surface colours.
    :return: X grid, Y grid, height stack, colour surface, and layer metadata.
    """
    with np.load(input_path) as archive:
        height = require_array(archive, "height_mm", input_path)
        growth_thickness = require_array(archive, "growth_thickness_mm", input_path)

        # A single static surface is accepted by promoting it to a one-layer stack.
        # The normal browser-ready format is already (layer, y, x).
        if height.ndim == 3:
            z_layers = height.astype(np.float32)
        elif height.ndim == 2:
            z_layers = height[np.newaxis, :, :].astype(np.float32)
        else:
            raise ValueError(
                f"{input_path} height_mm must have shape (layers, y, x) or (y, x); "
                f"got {height.shape}."
            )

        growth_layers = normalize_layer_array(
            growth_thickness,
            z_layers,
            "growth_thickness_mm",
            input_path,
        )
        layer_thickness_layers = (
            normalize_layer_array(
                archive["layer_thickness_mm"],
                z_layers,
                "layer_thickness_mm",
                input_path,
            )
            if "layer_thickness_mm" in archive.files
            else None
        )
        color_surface = build_color_surface(color_mode, growth_layers, layer_thickness_layers)

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
            color_values = color_surface.values.copy()
            # Keep inactive cells as NaN so Plotly leaves the masked exterior empty
            # instead of drawing a rectangular base plane around the circular mat.
            z_layers[:, ~active_mask] = np.nan
            color_values[:, ~active_mask] = np.nan
            color_surface = ColorSurface(
                values=color_values,
                title=color_surface.title,
                hover_label=color_surface.hover_label,
                colorscale=color_surface.colorscale,
                symmetric=color_surface.symmetric,
            )

        if not np.any(np.isfinite(z_layers)):
            raise ValueError(f"{input_path} does not contain any finite surface heights.")
        if not np.any(np.isfinite(color_surface.values)):
            raise ValueError(f"{input_path} does not contain any finite colour values.")

        layer_ids = optional_vector(archive, "layer_ids", layer_count, np.int16)
        start_steps = optional_vector(archive, "layer_start_step", layer_count, np.int32)
        end_steps = optional_vector(archive, "layer_end_step", layer_count, np.int32)

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    metadata = []
    for index, z_layer in enumerate(z_layers):
        # Build a compact label for each slider position from whatever metadata
        # the NPZ provides, falling back to the stack index when needed.
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
    return x_grid, y_grid, z_layers, color_surface, metadata


def build_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_layers: np.ndarray,
    color_surface: ColorSurface,
    metadata: list[LayerMetadata],
    *,
    clean_axes: bool,
    dark_mode: bool,
) -> go.Figure:
    """Create the interactive layer-browsing dome figure.

    :param x_grid: 2-D x-coordinate grid for Plotly.
    :param y_grid: 2-D y-coordinate grid for Plotly.
    :param z_layers: Height raster stack used for 3-D geometry.
    :param color_surface: Surface-colour values and display labels.
    :param metadata: Layer metadata used for slider labels and hover text.
    :param clean_axes: Whether to hide axes, ticks, and grid lines.
    :param dark_mode: Whether to use the dark visual theme.
    :return: Configured Plotly figure with slider frames.
    """
    color_layers = color_surface.values
    final_layer_index = len(metadata) - 1
    initial_z_grid = z_layers[final_layer_index]
    initial_color_grid = color_layers[final_layer_index]
    # The z-axis range stays global so the structure does not visually rescale
    # while browsing; the colour range is recalculated per selected layer below.
    min_height, max_height = finite_range(z_layers)
    initial_color_min, initial_color_max = (
        symmetric_finite_range(initial_color_grid)
        if color_surface.symmetric
        else finite_range(initial_color_grid)
    )
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
                surfacecolor=initial_color_grid,
                # Start on the final layer, matching the original single-surface render.
                cmin=initial_color_min,
                cmax=initial_color_max,
                customdata=customdata_for_layer(
                    initial_metadata,
                    initial_z_grid,
                    initial_color_grid,
                ),
                colorscale=color_surface.colorscale,
                colorbar={
                    "title": {"text": color_surface.title, "font": {"color": text_color}},
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
                    "height: %{z:.2f} mm<br>"
                    f"{color_surface.hover_label}: "
                    "%{customdata[3]:.2f} mm"
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
                    surfacecolor=color_layer,
                    # Each frame updates only the mutable surface fields; x/y,
                    # lighting, colourbar, and scene settings come from the base trace.
                    cmin=(
                        symmetric_finite_range(color_layer)[0]
                        if color_surface.symmetric
                        else finite_range(color_layer)[0]
                    ),
                    cmax=(
                        symmetric_finite_range(color_layer)[1]
                        if color_surface.symmetric
                        else finite_range(color_layer)[1]
                    ),
                    customdata=customdata_for_layer(layer_metadata, z_layer, color_layer),
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
        for index, (z_layer, color_layer, layer_metadata) in enumerate(
            zip(z_layers, color_layers, metadata, strict=True)
        )
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
                        # Zero-duration animation makes slider clicks behave like
                        # direct layer selection rather than a time interpolation.
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
                        # Play uses the same frames as the slider, starting from
                        # the currently selected layer.
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
    parser.add_argument("-cm", "--color-mode", choices=("growth-anomaly", "growth", "layer-thickness"),
                        default="layer-thickness", help="Surface colouring field")
    parser.add_argument("-c", "--clean-axes", action="store_true",
                        help="Hide axes, ticks and grid lines for publication-style presentation")
    parser.add_argument("-d", "--dark-mode", action="store_true",
                        help="Use a black background with light axis and colorbar text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # When --output is omitted, write beside the input NPZ with an HTML suffix.
    output_path = args.output if args.output is not None else args.input.with_suffix(".html")

    x_grid, y_grid, z_layers, color_surface, metadata = load_layer_surface_grids(
        args.input,
        args.color_mode,
    )
    fig = build_figure(
        x_grid,
        y_grid,
        z_layers,
        color_surface,
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
