neural_lam.datastore.npyfilesmeps.store
=======================================

.. py:module:: neural_lam.datastore.npyfilesmeps.store

.. autoapi-nested-parse::

   Numpy-files based datastore to support the MEPS example dataset introduced in
   neural-lam v0.1.0.







Module Contents
---------------

.. py:class:: NpyFilesDatastoreMEPS(config_path)

   Bases: :py:obj:`neural_lam.datastore.base.BaseRegularGridDatastore`


   Represents a dataset stored as numpy files on disk. The dataset is assumed
   to be stored in a directory structure where each sample is stored in a
   separate file. The file-name format is assumed to be described by
   ``STATE_FILENAME_FORMAT``.

   The MEPS dataset is organised into three splits: train, val, and test. Each
   split has a set of files which are:

   - ``STATE_FILENAME_FORMAT``:
       The state variables for a forecast started at `analysis_time` with
       member id `member_id`. The dimensions of the array are
       `[forecast_timestep, y, x, feature]`.

   - ``TOA_SW_DOWN_FLUX_FILENAME_FORMAT``:
       The top-of-atmosphere downwelling shortwave flux at `time`. The
       dimensions of the array are `[forecast_timestep, y, x]`.

   - ``OPEN_WATER_FILENAME_FORMAT``:
       The open water fraction at `time`. The dimensions of the array are
       `[y, x]`.


   Folder structure:

   meps_example_reduced
   ├── data_config.yaml
   ├── samples
   │   ├── test
   │   │   ├── nwp_2022090100_mbr000.npy
   │   │   ├── nwp_2022090100_mbr001.npy
   │   │   ├── nwp_2022090112_mbr000.npy
   │   │   ├── nwp_2022090112_mbr001.npy
   │   │   ├── ...
   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022090100.npy
   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022090112.npy
   │   │   ├── ...
   │   │   ├── wtr_2022090100.npy
   │   │   ├── wtr_2022090112.npy
   │   │   └── ...
   │   ├── train
   │   │   ├── nwp_2022040100_mbr000.npy
   │   │   ├── nwp_2022040100_mbr001.npy
   │   │   ├── ...
   │   │   ├── nwp_2022040112_mbr000.npy
   │   │   ├── nwp_2022040112_mbr001.npy
   │   │   ├── ...
   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040100.npy
   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040112.npy
   │   │   ├── ...
   │   │   ├── wtr_2022040100.npy
   │   │   ├── wtr_2022040112.npy
   │   │   └── ...
   │   └── val
   │       ├── nwp_2022060500_mbr000.npy
   │       ├── nwp_2022060500_mbr001.npy
   │       ├── ...
   │       ├── nwp_2022060512_mbr000.npy
   │       ├── nwp_2022060512_mbr001.npy
   │       ├── ...
   │       ├── nwp_toa_downwelling_shortwave_flux_2022060500.npy
   │       ├── nwp_toa_downwelling_shortwave_flux_2022060512.npy
   │       ├── ...
   │       ├── wtr_2022060500.npy
   │       ├── wtr_2022060512.npy
   │       └── ...
   └── static
       ├── border_mask.npy
       ├── diff_mean.pt
       ├── diff_std.pt
       ├── flux_stats.pt
       ├── grid_features.pt
       ├── nwp_xy.npy
       ├── parameter_mean.pt
       ├── parameter_std.pt
       ├── parameter_weights.npy
       └── surface_geopotential.npy

   For the MEPS dataset:
   N_t' = 65
   N_t = 65//subsample_step (= 21 for 3h steps)
   dim_y = 268
   dim_x = 238
   N_grid = 268x238 = 63784
   d_features = 17 (d_features' = 18)
   d_forcing = 5

   For the MEPS reduced dataset:
   N_t' = 65
   N_t = 65//subsample_step (= 21 for 3h steps)
   dim_y = 134
   dim_x = 119
   N_grid = 134x119 = 15946
   d_features = 8
   d_forcing = 1

   Create a new NpyFilesDatastore using the configuration file at the
   given path. The config file should be a YAML file and will be loaded
   into an instance of the `NpyDatastoreConfig` dataclass.

   Internally, the datastore uses dask.delayed to load the data from the
   numpy files, so that the data isn't actually loaded until it's needed.

   :param config_path: The path to the configuration file for the datastore.
   :type config_path: str


   .. py:method:: coords_projection() -> cartopy.crs.Projection

      The projection of the spatial coordinates.

      :returns: The projection of the spatial coordinates.
      :rtype: ccrs.Projection



   .. py:method:: get_dataarray(category: str, split: Optional[str], standardize: bool = False) -> xarray.core.dataarray.DataArray

      Get the data array for the given category and split of data. If the
      category is 'state', the data array will be a concatenation of the data
      arrays for all ensemble members. The data will be loaded as a dask
      array, so that the data isn't actually loaded until it's needed.

      :param category: The category of the data to load. One of 'state', 'forcing', or
                       'static'.
      :type category: str
      :param split: The dataset split to load the data for. One of 'train', 'val', or
                    'test'.
      :type split: str
      :param standardize: If the dataarray should be returned standardized
      :type standardize: bool

      :returns: The data array for the given category and split, with dimensions
                per category:
                state:     `[elapsed_forecast_duration, analysis_time, grid_index,
                            feature, ensemble_member]`
                forcing:   `[elapsed_forecast_duration, analysis_time, grid_index,
                            feature]`
                static:    `[grid_index, feature]`
      :rtype: xr.DataArray



   .. py:method:: get_num_data_vars(category: str) -> int

      Return the number of variables available in ``category``.



   .. py:method:: get_standardization_dataarray(category: str) -> xarray.Dataset

      Return the standardization dataarray for the given category. This
      should contain a `{category}_mean` and `{category}_std` variable for
      each variable in the category.
      For `category=="state"`, the dataarray should also contain a
      `state_diff_mean_standardized` and `state_diff_std_standardized`
      variable for the one-step differences of the state variables.

      :param category: The category of the dataset (state/forcing/static).
      :type category: str

      :returns: The standardization dataarray for the given category, with
                variables for the mean and standard deviation of the variables (and
                differences for state variables).
      :rtype: xr.Dataset



   .. py:method:: get_vars_long_names(category: str) -> List[str]

      Return descriptive names for the variables in ``category``.



   .. py:method:: get_vars_names(category: str) -> List[str]

      Return canonical short names for the variables in ``category``.



   .. py:method:: get_vars_units(category: str) -> List[str]

      Return unit strings for the variables in ``category``.



   .. py:method:: get_xy(category: str, stacked: bool) -> numpy.ndarray

      Return the x, y coordinates of the dataset.

      :param category: The category of the dataset (state/forcing/static).
      :type category: str
      :param stacked: Whether to stack the x, y coordinates.
      :type stacked: bool

      :returns: The x, y coordinates of the dataset (with x first then y second),
                returned differently based on the value of `stacked`:
                - `stacked==True`: shape `(n_grid_points, 2)` where
                                          n_grid_points=N_x*N_y.
                - `stacked==False`: shape `(N_x, N_y, 2)`
      :rtype: np.ndarray



   .. py:attribute:: SHORT_NAME
      :value: 'npyfilesmeps'



   .. py:property:: boundary_mask
      :type: xarray.DataArray


      The boundary mask for the dataset. This is a binary mask that is 1
      where the grid cell is on the boundary of the domain, and 0 otherwise.

      :returns: The boundary mask for the dataset, with dimensions `[grid_index]`.
      :rtype: xr.DataArray


   .. py:property:: config
      :type: neural_lam.datastore.npyfilesmeps.config.NpyDatastoreConfig


      The configuration for the datastore.

      :returns: The configuration for the datastore.
      :rtype: NpyDatastoreConfig


   .. py:property:: grid_shape_state
      :type: neural_lam.datastore.base.CartesianGridShape


      The shape of the cartesian grid for the state variables.

      :returns: The shape of the cartesian grid for the state variables.
      :rtype: CartesianGridShape


   .. py:attribute:: is_ensemble
      :value: True



   .. py:attribute:: is_forecast
      :value: True



   .. py:property:: root_path
      :type: pathlib.Path


      The root path of the datastore on disk. This is the directory relative
      to which graphs and other files can be stored.

      :returns: The root path of the datastore
      :rtype: Path


   .. py:property:: step_length
      :type: datetime.timedelta


      The length of each time step as a time interval.

      :returns: The length of each time step as a datetime.timedelta object.
      :rtype: timedelta


.. py:data:: OPEN_WATER_FILENAME_FORMAT
   :value: 'wtr_{analysis_time:%Y%m%d%H}.npy'


.. py:data:: STATE_FILENAME_FORMAT
   :value: 'nwp_{analysis_time:%Y%m%d%H}_mbr{member_id:03d}.npy'


.. py:data:: TOA_SW_DOWN_FLUX_FILENAME_FORMAT
   :value: 'nwp_toa_downwelling_shortwave_flux_{analysis_time:%Y%m%d%H}.npy'


