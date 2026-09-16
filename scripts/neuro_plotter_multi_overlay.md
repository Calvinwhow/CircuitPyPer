# Multi-overlay plots

A figure becomes multi-map when it contains `overlays`. The normal `plot` key
still chooses the geometry; there is no separate multi-overlay plot type.

```python
{
    "name": "triple_dissociation",
    "plot": "parcel_mesh",
    "mesh": "aal3_suit_parcels",
    "atlas": None,
    "cerebellum": None,
    "overlays": [
        {
            "path": "/path/to/motor.nii.gz",
            "label": "Motor Network",
            "color": "#c15656",
            "threshold": 1.96,
            "alpha": 0.85,
        },
        {
            "path": "/path/to/cognitive.nii.gz",
            "label": "Cognitive Network",
            "color": "#5071a0",
            "threshold": 1.96,
            "alpha": 0.85,
        },
        {
            "path": "/path/to/emotional.nii.gz",
            "label": "Emotional Network",
            "color": "#9a8ed1",
            "threshold": 1.96,
            "alpha": 0.85,
        },
    ],
    "plot_kwargs": {"zoom": 1.25},
}
```

| Key | Purpose |
|---|---|
| `name` | Output name only |
| `plot` | Geometry and renderer |
| `overlays` | Maps and their visual properties |

Supported multi-map geometries are:

| `plot` | Result |
|---|---|
| `parcel_mesh` | Colors the regional set selected by `mesh` |
| `mesh` | Draws NIfTIs as isosurfaces and `.mat`, `.fib.npy`, or `.tck` inputs as streamlines inside the selected `mesh` |

Older `volume` and `tracts` plot values still normalize to `mesh`.

For `mesh`, use `mesh="glass_wholebrain"` for the carved hull or
`mesh="pial_wholebrain"` for cortical/SUIT/brainstem context. For
`parcel_mesh`, change only `mesh` to switch between the custom AAL3+SUIT set
and yabplot sets such as `aseg` or `tian2020_s1`.

## Overlay fields

| Field | Meaning | Default |
|---|---|---|
| `path` | Input NIfTI, `.mat`, `.fib.npy`, or `.tck` | required |
| `label` | Legend text | filename |
| `color` | Matplotlib color or hex | required |
| `threshold` | Magnitude cutoff or percentile such as `"95%"` | `"95%"` |
| `alpha` | Overlay opacity | `0.85` for parcels; `0.65` for surfaces |
| `sign` | `absolute`, `positive`, or `negative` | `absolute` |
| `n_levels` | Nested volumetric surfaces | figure value |
| `vmax` | Upper value for nested surfaces | surviving 99th percentile |

Relative paths are resolved beside `NIFTI_PATH`. All maps must already be in
the same physical coordinate system; the renderer does not register them.

For `parcel_mesh`, each NIfTI's contribution is scaled by the fraction of mesh
points above threshold, while tract inputs remain streamlines over the parcels.
For `mesh`, NIfTIs are transparent isosurfaces and tract inputs remain
streamlines, including when both types occur in one figure. Start with
`n_levels=1`; additional levels show stronger NIfTI cores but make the image
busier.

Screen-space overlap is view-dependent. Confirm apparent intersections in more
than one anatomical view.

## Fiber overlays

Use the same `plot` and `mesh` keys as NIfTI inputs. The file extension selects
streamline rendering automatically; there is no separate `tract_mesh`. Each MAT
or `.fib.npy` file is converted to a temporary TCK, rendered, and deleted after
all requested views and formats are complete.

`threshold` is an absolute statistic cutoff for fiber overlays. `top_percent`
can be set globally with `TRACT_TOP_PERCENT` or per overlay. A geometry-bearing
`.fib.npy` needs nothing else. A one-dimensional `.fib.npy` value vector needs
`TRACT_ATLAS_PATH` (or an overlay-level `fiber_atlas_path`) pointing to its
canonical `.npz`/`.npy` fiber atlas. Lead-DBS FTR and discriminative-fiber MAT
files are read directly.
