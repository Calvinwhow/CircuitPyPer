import numpy as np
import nibabel as nib
import os
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hdbscan
import plotly.io as pio 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from calvin_utils.ml_utils.torus_utils import project_points_to_torus_auto
from calvin_utils.ml_utils.sphere_utils import project_points_to_sphere_auto


class BrainUmap:
    """
    Spatially-aware UMAP + HDBSCAN workflow for 3-D brain volumes.

    The class converts a collection of volumetric maps (one per subject) into a
    low-dimensional embedding that jointly encodes voxel intensities and their
    spatial coordinates, then clusters the embedding to identify coherent
    anatomical/functional regions.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Each row is a voxel, each column is a map (subject or condition).  
        Shape = (n_voxels, n_maps).
    n_components : int, default 3
        Target dimensionality of the UMAP embedding.
    n_neighbors : int, default 15
        Size of the local neighborhood UMAP uses to balance local vs. global
        structure.
    min_dist : float, default 0.1
        Minimum distance between points in the embedded space (UMAP *min_dist*).
    metric : str or callable, default "correlation"
        Distance metric passed to UMAP when *cluster_voxels=True*. When
        *cluster_voxels=False* the metric is only used for pair-wise distance
        calculation after projection. See ``umap.UMAP`` docs for options.
    projection : {"sphere", None}, default "sphere"
        • "sphere" — radially projects a 3-D embedding onto the unit sphere  
        • None      — no projection
    min_cluster_size : int, default 5
        Minimum cluster size for HDBSCAN.
    mask : str or None, default None
        Path to a NIfTI mask; voxels with 0 are excluded. If None, all voxels
        are used.
    verbose : bool, default True
        If True, prints status messages.
    cluster_voxels : bool, default True
        If True, clusters individual voxels in embedding space.  
        If False, clusters whole maps using a pre-computed distance matrix.

    Attributes
    ----------
    mask : ndarray or None
        Boolean flat mask applied to the voxel axis.
    brain_array : ndarray, shape (n_maps, n_selected_voxels)
        Masked data matrix.
    features : ndarray, shape (n_maps, n_selected_voxels)
        Z-scored intensity features fed to UMAP.
    embedding : ndarray, shape (n_samples, n_components)
        Low-dimensional representation after optional projection.
    distances : ndarray or None
        Pair-wise distance matrix (only when *cluster_voxels=False*).
    cluster_labels : ndarray, shape (n_samples,)
        Label of the HDBSCAN cluster assigned to each point; -1 for noise.
    cluster_probabilities : ndarray
        Soft cluster membership probabilities.
    cluster_persistence : ndarray
        HDBSCAN cluster persistence scores.

    Methods
    -------
    plot_embedding(point_size=3, opacity=0.6)
        Interactive Plotly scatter/Scatter3d of the embedding with cluster
        labels.
    run()
        Convenience wrapper that calls ``plot_embedding()``.

    Notes
    -----
    • Data are expected row-wise as voxels (long format) to avoid reshaping the
      4-D array in RAM.  
    • When *projection="sphere"* the embedding must be 3-D; the class rescales
      each point onto the unit sphere before clustering.  
    • Setting *cluster_voxels=False* yields a subject-level clustering based on
      pre-computed distances in embedding space.

    Examples
    --------
    >>> umap_runner = BrainUmap(df, mask="brainmask.nii.gz",
    ...                         n_components=3, projection="sphere",
    ...                         metric="correlation", cluster_voxels=True)
    >>> umap_runner.plot_embedding()
    """
    def __init__(self, data, n_components=2, n_neighbors=15, min_dist=0.1, metric='correlation', projection="sphere", min_cluster_size=5, mask=None, verbose=True, cluster_voxels=False, visualize_failed_clusters=False, cols_to_flip=[]):
        """
        metric: 'euclidean', 'manhattan', 'cosine', 'correlation', 'haversine', etc. (see umap.UMAP docs)
        min_dist recommendation: 0.1 (low for tight clusters, higher for more spread out embedding)
        n_neighbors recommendation: 15 (smaller for local structure, larger for global structure)
        n_components recommendation: 2 or 3 (for visualization; higher for downstream ML tasks)
        """
        self.verbose = verbose
        self.projection = projection
        self.visualize_failed_clusters = visualize_failed_clusters
        arr, self.cols = self._get_arr(data)
        arr = self._flip_selected_columns(arr, cols_to_flip)
        self.mask = self._get_mask(mask)
        self.brain_array = self._mask_arr(arr, cluster_voxels) # (N maps, N voxels)
        self.features = self._get_features()
        self.embedding = self._run_umap(n_components, n_neighbors, min_dist, metric)
        self.embedding = self._project_embedding()
        self.distances = self._compute_distance(metric, cluster_voxels)
        self.cluster_labels, self.cluster_probabilities, self.cluster_persistence = self._run_hdbscan(min_cluster_size, cluster_voxels)
        
    ### Setters and Getters ###
    def _get_arr(self, data):
        if isinstance(data, pd.DataFrame):
            arr = data.values
            cols = data.columns
        else:
            arr = np.asarray(data)
            cols = None
        return arr.T, cols

    def _flip_selected_columns(self, arr, cols_to_flip):
        """Flip maps whose column names contain any requested string."""
        self.cols_to_flip = list(cols_to_flip or [])
        self.flipped_cols = []

        if len(self.cols_to_flip) == 0:
            return arr
        if self.cols is None:
            raise ValueError("cols_to_flip requires DataFrame input with column names.")

        col_names = [str(col) for col in self.cols]
        arr = arr.copy()
        for pattern in self.cols_to_flip:
            matches = [idx for idx, col in enumerate(col_names) if str(pattern) in col]
            if len(matches) == 0 and self.verbose:
                print(f"[BrainUmap] No columns matched cols_to_flip pattern: {pattern}")
            for idx in matches:
                arr[idx, :] *= -1
                self.flipped_cols.append(self.cols[idx])

        if self.verbose and self.flipped_cols:
            print(f"[BrainUmap] Flipped {len(self.flipped_cols)} input map(s): {list(self.flipped_cols)}")
        return arr
        
    def _get_mask(self, path):
        '''Imports a nifti mask'''
        return nib.load(path).get_fdata().flatten() > 0 if path else None
    
    def _mask_arr(self, arr, cluster_voxels):
        '''Masks the array'''
        if self.mask is not None:
            arr = arr[:, self.mask]
        return arr.T if cluster_voxels else arr
            
    def _get_features(self):
        """Creates voxel-wise feature vectors combining spatial and intensity information."""
        return StandardScaler().fit_transform(self.brain_array) # standardizes expecting (N maps, N voxels)
    
    ### Geometry API ###
    def _calculate_location_on_circle(self, R=1):
        '''assumes the embeddings define the angles on a unit circle'''
        theta = np.mod(self.embedding[:,0], 2*np.pi)
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        return x, y
    
    def _calculate_location_on_sphere(self, R=1):
        '''assumes the embeddings define the angles of a spherical surface'''
        phi = np.mod(self.embedding[:,0], 2*np.pi)    # azimuth
        theta = np.mod(self.embedding[:,1], np.pi)    # polar angle
        
        x = R * np.cos(phi)*np.sin(theta)
        y = R * np.sin(phi)*np.sin(theta)
        z = R * np.cos(theta)
        return x, y, z
    
    def _calculate_location_on_torus(self, R=2, r=1):
        '''assumes the embeddings define the angles of a toroidal surface'''
        phi = np.mod(self.embedding[:,0], 2*np.pi)
        theta = np.mod(self.embedding[:,1], 2*np.pi)
        x = (R + r*np.cos(phi))*np.cos(theta)
        y = (R + r*np.cos(phi))*np.sin(theta)
        z = r*np.sin(phi)
        return x, y, z
        
    ### Internal API ###
    def _run_umap(self, n_components, n_neighbors, min_dist, metric):
        """Runs UMAP dimensionality reduction."""
        if (metric == 'haversine' or metric == 'geodesic') and (self.features.shape[1] != 2):
            print(f"Cannot pass haversine/geodesic distance umapped with {self.features.shape[1]} features. Substituting in cosine distance.")
            metric = 'cosine'
        umapper = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
        return umapper.fit_transform(self.features) # learns on (N maps, N voxels)
    
    def _project_embedding(self):
        '''Projects the embedding if the embedding space is created'''
        if self.projection=="circle":
            arr = np.zeros((self.embedding.shape[0], 2))
            if self.embedding.shape[1] != 2:
                raise ValueError("Embedding must have 2 dimensions. Set n_components=2.")
            arr[:,0], arr[:,1] = self._calculate_location_on_circle()
            return arr
        
        elif self.projection=="sphere": 
            arr = np.zeros((self.embedding.shape[0], 3))
            if self.embedding.shape[1] == 2:
                arr[:,0], arr[:,1], arr[:,2] = self._calculate_location_on_sphere()
            elif self.embedding.shape[1] == 3:
                arr[:,0], arr[:,1], arr[:,2] = project_points_to_sphere_auto(self.embedding[:,0], self.embedding[:,1], self.embedding[:,2])
            else:
                raise ValueError("Embedding must have 2 or 3 dimensions for=r spherical projection. Set n_components=2or 3.")
            return arr
        
        elif self.projection=="torus":
            arr = np.zeros((self.embedding.shape[0], 3))
            if self.embedding.shape[1] == 2:
                arr[:,0], arr[:,1], arr[:,2] = self._calculate_location_on_torus()
            elif self.embedding.shape[1] == 3:
                arr[:,0], arr[:,1], arr[:,2] = project_points_to_torus_auto(self.embedding[:,0], self.embedding[:,1],  self.embedding[:,2])
            else:
                raise ValueError("Toroidal embedding must have 2 or 3 dimensions for toroidal. Set n_components=2 or 3.")
            return arr
        
        elif self.projection is None:
            return self.embedding
        else:
            raise ValueError(f"projection={self.projection} not implemented.")
        
    def _compute_distance(self, metric: str, cluster_voxels: bool, force_euclidean=False):
        """Return an (n, n) distance matrix for HDBSCAN when cluster_voxels=False. Receuves enbedding shaped [obs, nlatent]"""
        if cluster_voxels:
            return None
        
        X = self.embedding.astype(np.float64) # shape (obs, latent_dims)
        if metric == "cosine":  # Cosine Similarity
            norm = np.linalg.norm(X, axis=-1, keepdims=True) # (obs, 1) <- (obs, n_latent)
            sim = (X @ X.T) / (norm @ norm.T)                # (obs, obs) <- (obs, n_latent) @ (n_latent)
            d = 1.0 - sim
            
        elif metric == "haversine":  # Haversine Distance
            if X.shape[1] != 2:
                raise ValueError("Haversine is only defined for 2 dimensional data. Call geodesic if working with spherical projection of 3 dimensions.")
            lat = np.radians(X[:, 0])[:, None]  # (n,1)
            lon = np.radians(X[:, 1])[:, None]  # (n,1)
            dlat = lat - lat.T                  # (n,n)
            dlon = lon - lon.T                  # (n,n)
            a = np.sin(dlat*0.5)**2 + np.cos(lat)*np.cos(lat.T) * np.sin(dlon*0.5)**2
            a = np.clip(a, 0.0, 1.0)
            d = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))  # radians on unit sphere

        elif metric == "geodesic" or "spherical":   # distance between two points travelling along a sphere's surfaces
            X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
            dot = np.clip(X @ X.T, -1.0, 1.0)
            cross = np.linalg.norm(np.cross(X[:, None, :], X[None, :, :]), axis=-1)
            d = np.arctan2(cross, dot)
                
        elif (metric == "euclidean") or (force_euclidean):                                                       # Euclidean Distance     
            d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            
        else: 
            raise ValueError(f"Metric {metric} not yet implemented")
        return d
    
    def _run_hdbscan(self, min_cluster_size, cluster_voxels):
        '''Clusters using distance metrics calculated from the embedding'''
        metric = 'euclidean' if cluster_voxels else 'precomputed'
        arr = self.embedding if cluster_voxels else self.distances
        kwargs = {}
        kwargs['cluster_selection_method'] = 'eom'      # leaf or eom. eom is typical HDBSCAN, while leaf gets smaller homogeneous clusters. 
        kwargs['allow_single_cluster'] = False          # False is standard HDBSCAN operation 
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, **kwargs).fit(arr)
        return clusterer.labels_, clusterer.probabilities_, clusterer.cluster_persistence_
    
    def _get_opacity(self, override_probabilities=None, noise_alpha=0.0):
        probs  = override_probabilities if override_probabilities is not None else self.cluster_probabilities
        labels = self.cluster_labels
        labels_norm = (labels - np.min(labels)) / (np.ptp(labels) + 1e-6)
        colors = cm.get_cmap('viridis')(labels_norm)  # RGBA in [0,1]
        rgba_colors = [
            f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{(noise_alpha if lbl==-1 else (p + (1-p)/2))})'
            for (r,g,b,_), lbl, p in zip(colors, labels, probs)
        ]
        return rgba_colors

    ### Public API ###
    def export_cluster_report(self, out_path):
        """
        Writes a CSV summarizing the UMAP + HDBSCAN clustering results.

        Parameters
        ----------
        out_path : str
            Full path (including filename) where the CSV will be saved.
        """
        n_samples = self.embedding.shape[0]

        df = pd.DataFrame({
            "sample_id": np.arange(n_samples),
            "umap_x": self.embedding[:, 0],
            "umap_y": self.embedding[:, 1] if self.embedding.shape[1] > 1 else np.nan,
            "umap_z": self.embedding[:, 2] if self.embedding.shape[1] > 2 else np.nan,
            "cluster_label": self.cluster_labels,
            "cluster_probability": self.cluster_probabilities,
            "cluster_persistence": [
                self.cluster_persistence[label] if label >= 0 else np.nan
                for label in self.cluster_labels
            ],
            "image_path": self.cols
        })

        df.to_csv(out_path, index=False)
        if self.verbose:
            print(f"[BrainUmap] Cluster report saved to: {out_path}")
        return df
    
    def plot_embedding(self, *, point_size: int = 4, override_probabilities = None, outlines=True, verbose = True, continuous_colour=True):
        """Interactive scatter of the embedding using Plotly. Points coloured by cluster label, opacity by probability of belonging to a cluster."""
        emb = self.embedding
        rgba_colors = self._get_opacity(override_probabilities)
        customdata = np.stack([self.cluster_labels, self.cluster_probabilities], axis=-1)
        labels = self.cluster_labels
        if not self.visualize_failed_clusters:
            mask               = self.cluster_labels != -1
            emb                = emb[mask]
            rgba_colors        = [c for c, keep in zip(rgba_colors, mask) if keep]
            customdata         = customdata[mask]
            labels             = labels[mask]

        if continuous_colour:
            col = rgba_colors
        else:
            col = labels
        
        if (emb.shape[1] == 2) and (self.projection != "torus"):
            fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=col,
                             size=np.full(emb.shape[0], point_size),
                             labels={"x": "UMAP 1", "y": "UMAP 2"})
            fig.update_layout(scene=dict(xaxis_title="UMAP 1",
                                         yaxis_title="UMAP 2"))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        elif emb.shape[1] == 3:
            zero = np.zeros(emb.shape[0])
            dim_buttons = []
            dim_options = [
                ("UMAP 1+2+3", (0, 1, 2)),
                ("UMAP 1+2", (0, 1)),
                ("UMAP 1+3", (0, 2)),
                ("UMAP 2+3", (1, 2)),
                ("UMAP 1", (0,)),
                ("UMAP 2", (1,)),
                ("UMAP 3", (2,)),
            ]
            for label, dims in dim_options:
                coords = [
                    emb[:, axis] if axis in dims else zero
                    for axis in range(3)
                ]
                dim_buttons.append(dict(
                    label=label,
                    method="update",
                    args=[
                        {"x": [coords[0]], "y": [coords[1]], "z": [coords[2]]},
                        {
                            "scene.xaxis.title.text": "UMAP 1" if 0 in dims else "",
                            "scene.yaxis.title.text": "UMAP 2" if 1 in dims else "",
                            "scene.zaxis.title.text": "UMAP 3" if 2 in dims else "",
                        },
                    ],
                ))
            axes_buttons = [
                dict(
                    label="Hide axes",
                    method="relayout",
                    args=[{
                        "scene.xaxis.visible": False,
                        "scene.yaxis.visible": False,
                        "scene.zaxis.visible": False,
                    }],
                ),
                dict(
                    label="Show axes",
                    method="relayout",
                    args=[{
                        "scene.xaxis.visible": True,
                        "scene.yaxis.visible": True,
                        "scene.zaxis.visible": True,
                    }],
                ),
            ]

            fig = go.Figure(go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode='markers', 
                marker=dict(size=point_size, color=col, line=dict(width=0.5 if outlines else 0, color='black')),
                customdata=customdata,
                hovertemplate=(
                    'UMAP 1: %{x}<br>'
                    'UMAP 2: %{y}<br>'
                    'UMAP 3: %{z}<br>'
                    'Cluster: %{customdata[0]}<br>'
                    'Probability: %{customdata[1]:.2f}<br>'
                    '<extra></extra>'
                )
            ))
            fig.update_layout(scene=dict(xaxis_title="UMAP 1",
                                         yaxis_title="UMAP 2",
                                         zaxis_title="UMAP 3",
                                         aspectmode="cube"),
                              updatemenus=[dict(
                                  type="dropdown",
                                  buttons=dim_buttons,
                                  direction="down",
                                  x=0.0,
                                  xanchor="left",
                                  y=1.08,
                                  yanchor="top",
                                  showactive=True,
                              ), dict(
                                  type="buttons",
                                  buttons=axes_buttons,
                                  direction="right",
                                  x=0.36,
                                  xanchor="left",
                                  y=1.08,
                                  yanchor="top",
                                  showactive=True,
                              )],
                              font=dict(family="Helvetica",
                                        color="black"))
        else:
            raise ValueError("Embedding must be 2D or 3D for plotting.")

        if emb.shape[1] == 3:
            axis_style = dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=True,
                showticklabels=False,
                zeroline=True,
                ticks="",
                linecolor="rgba(40,40,40,0.7)",
                zerolinecolor="rgba(40,40,40,0.9)",
            )
            fig.update_layout(
                title="Interactive UMAP Embedding",
                template="plotly_white",
                scene=dict(
                    xaxis=dict(axis_style, title="UMAP 1"),
                    yaxis=dict(axis_style, title="UMAP 2"),
                    zaxis=dict(axis_style, title="UMAP 3"),
                ),
            )
        else:
            fig.update_layout(title="Interactive UMAP Embedding", template="plotly_white")
        if verbose: fig.show()
        return fig

    def export_plotly_view(
        self,
        out_path,
        *,
        camera=None,
        width=1600,
        height=1600,
        scale=2,
        point_size=6,
        outlines=False,
        axes=False,
        title=None,
        show_controls=False,
        transparent=False,
        continuous_colour=True,
        preserve_aspect=False,
    ):
        """
        Export the interactive Plotly view to a static image.

        Notes
        -----
        Requires the optional ``kaleido`` package. For a view you can reproduce,
        pass a Plotly ``scene_camera`` dict.
        """
        fig = self.plot_embedding(
            point_size=point_size,
            outlines=outlines,
            verbose=False,
            continuous_colour=continuous_colour,
        )
        if camera is not None:
            fig.update_layout(scene_camera=camera)
        if not show_controls:
            fig.layout.updatemenus = ()
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=40 if title else 0, b=0),
            paper_bgcolor="rgba(0,0,0,0)" if transparent else "white",
            plot_bgcolor="rgba(0,0,0,0)" if transparent else "white",
        )
        if self.embedding.shape[1] == 3:
            data_min = self.embedding.min(axis=0)
            data_max = self.embedding.max(axis=0)
            data_range = data_max - data_min
            data_range[data_range == 0] = 1
            padding = data_range * 0.08
            if preserve_aspect:
                center = self.embedding.mean(axis=0)
                radius = np.max(np.linalg.norm(self.embedding - center, axis=1))
                if radius == 0:
                    radius = 1
                ranges = [
                    [center[0] - radius, center[0] + radius],
                    [center[1] - radius, center[1] + radius],
                    [center[2] - radius, center[2] + radius],
                ]
                aspectmode = "cube"
            else:
                ranges = [
                    [data_min[0] - padding[0], data_max[0] + padding[0]],
                    [data_min[1] - padding[1], data_max[1] + padding[1]],
                    [data_min[2] - padding[2], data_max[2] + padding[2]],
                ]
                aspectmode = "data"
            fig.update_layout(scene=dict(
                xaxis=dict(visible=axes, range=ranges[0]),
                yaxis=dict(visible=axes, range=ranges[1]),
                zaxis=dict(visible=axes, range=ranges[2]),
                aspectmode=aspectmode,
                domain=dict(x=[0, 1], y=[0, 1]),
            ))
        try:
            fig.write_image(out_path, width=width, height=height, scale=scale)
        except ValueError as exc:
            raise RuntimeError(
                "Static Plotly export requires kaleido. Install it with: pip install kaleido"
            ) from exc
        return out_path

    def export_matplotlib_view(
        self,
        out_path,
        *,
        elevation=24,
        azimuth=-55,
        roll=0,
        point_size=42,
        outlines=False,
        dpi=600,
        figsize=(5.5, 5.5),
        axes=True,
        grid=True,
        legend=False,
        transparent=False,
    ):
        """Export a fixed-orientation Matplotlib figure for publication."""
        fig = self.plot_embedding_matplotlib(
            point_size=point_size,
            outlines=outlines,
            dpi=dpi,
            elevation=elevation,
            azimuth=azimuth,
            roll=roll,
            figsize=figsize,
            axes=axes,
            grid=grid,
            legend=legend,
        )
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches='tight',
            transparent=transparent,
            facecolor='none' if transparent else 'white',
        )
        plt.close(fig)
        return out_path

    def plot_embedding_matplotlib(self, *, point_size=100, outlines=True, dpi=300, out_path=None, elevation=60, azimuth=30, roll=0, figsize=(8, 8), axes=True, grid=False, legend=True):
        emb   = self.embedding.copy()
        labs  = self.cluster_labels.copy()
        probs = self.cluster_probabilities.copy()

        if not self.visualize_failed_clusters:
            keep = labs != -1
            emb, labs, probs = emb[keep], labs[keep], probs[keep]

        uniq = np.unique(labs)
        # nice categorical palette
        base = list(plt.cm.tab10.colors)
        palette = {lab: base[i % len(base)] for i, lab in enumerate(uniq)}

        if emb.shape[1] == 2:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            for lab in uniq:
                m = (labs == lab)
                rgb = np.tile(palette[lab], (m.sum(), 1))                    # (n,3)
                rgba = np.hstack([rgb, np.clip(probs[m], 0.15, 1.0)[:, None]])  # (n,4)
                ax.scatter(
                    emb[m, 0], emb[m, 1],
                    s=point_size,
                    c=rgba,
                    edgecolors='k' if outlines else 'none',
                    linewidths=0.35 if outlines else 0
                )
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_aspect('equal', adjustable='box')
            ax.grid(grid, color='0.88', linewidth=0.6)
            if not axes:
                ax.axis('off')
            # legend
            handles = [Line2D([0],[0], marker='o', linestyle='',
                            markerfacecolor=palette[lab], markeredgecolor='k' if outlines else palette[lab],
                            markersize=6) for lab in uniq]
            if axes and legend:
                ax.legend(handles, [f"Cluster {lab}" for lab in uniq], frameon=False, title="Cluster", loc='best')

        elif emb.shape[1] == 3:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho')
            for lab in uniq:
                m = (labs == lab)
                rgb = np.tile(palette[lab], (m.sum(), 1))
                rgba = np.hstack([rgb, np.clip(probs[m], 0.15, 1.0)[:, None]])
                ax.scatter(
                    emb[m, 0], emb[m, 1], emb[m, 2],
                    s=point_size,
                    c=rgba,
                    depthshade=False,
                    edgecolors='k' if outlines else 'none',
                    linewidths=0.25 if outlines else 0
                )
            data_min = emb.min(axis=0)
            data_max = emb.max(axis=0)
            ranges = data_max - data_min
            ranges[ranges == 0] = 1
            pad = ranges * 0.08
            ax.set_xlim(data_min[0] - pad[0], data_max[0] + pad[0])
            ax.set_ylim(data_min[1] - pad[1], data_max[1] + pad[1])
            ax.set_zlim(data_min[2] - pad[2], data_max[2] + pad[2])
            ax.set_box_aspect(ranges / ranges.max())
            ax.view_init(elev=elevation, azim=azimuth, roll=roll)
            if axes:
                ax.set_xlabel("UMAP 1", labelpad=8)
                ax.set_ylabel("UMAP 2", labelpad=8)
                ax.set_zlabel("UMAP 3", labelpad=8)
                ax.tick_params(axis='both', which='major', labelsize=8, pad=2)
                ax.grid(grid)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.pane.fill = False
                    axis.pane.set_edgecolor('0.90')
                    axis._axinfo["grid"].update({"linewidth": 0.35, "color": (0.78, 0.78, 0.78, 0.55)})
                    axis._axinfo["axisline"].update({"linewidth": 0.8, "color": (0.15, 0.15, 0.15, 1.0)})
            else:
                ax.set_axis_off()
            # legend (2D proxy)
            handles = [Line2D([0],[0], marker='o', linestyle='',
                            markerfacecolor=palette[lab], markeredgecolor='k' if outlines else palette[lab],
                            markersize=6) for lab in uniq]
            if axes and legend:
                ax.legend(handles, [f"Cluster {lab}" for lab in uniq], frameon=False, title="Cluster", loc='upper left', bbox_to_anchor=(0.0, 1.0))

        else:
            raise ValueError("Embedding must be 2D or 3D.")

        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        return fig

    
    ### Orchestration ###
    def run(self, out_dir=None):
        '''Orchestrator method. Leave out_dir as None if you do not want to save results.'''
        fig = self.plot_embedding()
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            self.export_cluster_report(os.path.join(out_dir, 'cluster_results.csv'))
            fig.write_html(os.path.join(out_dir, 'umap_embedding_full.html'))
            if self.embedding.shape[1] == 3:
                views = {
                    'three_quarter': dict(elevation=24, azimuth=-55, roll=0),
                    'front_right': dict(elevation=15, azimuth=-25, roll=0),
                    'front_left': dict(elevation=15, azimuth=-145, roll=0),
                    'top_oblique': dict(elevation=58, azimuth=-45, roll=0),
                }
                for name, view in views.items():
                    for ext in ('svg', 'png'):
                        self.export_matplotlib_view(
                            os.path.join(out_dir, f'umap_publication_matplotlib_{name}_axes.{ext}'),
                            **view,
                            point_size=34,
                            outlines=False,
                            dpi=600,
                            figsize=(5.2, 5.2),
                            axes=True,
                            grid=True,
                            legend=False,
                            transparent=False,
                        )
            else:
                for ext in ('svg', 'png'):
                    self.export_matplotlib_view(
                        os.path.join(out_dir, f'umap_publication_matplotlib_axes.{ext}'),
                        point_size=34,
                        outlines=False,
                        dpi=600,
                        figsize=(5.2, 5.2),
                        axes=True,
                        grid=True,
                        legend=False,
                        transparent=False,
                    )
        return fig
