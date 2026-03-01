neural_lam.vis
==============

.. py:module:: neural_lam.vis

.. autoapi-nested-parse::

   Visualization helpers for analysing Neural-LAM predictions and errors.





Module Contents
---------------

.. py:function:: plot_error_map(errors, datastore: neural_lam.datastore.base.BaseRegularGridDatastore, title=None)

   Plot a heatmap of per-variable errors across prediction horizons.

   :param errors: Error values for each horizon and feature.

                  * **Shape**: ``(pred_steps, d_f)``
   :type errors: torch.Tensor
   :param datastore: Datastore providing metadata for labels and units.
   :type datastore: BaseRegularGridDatastore
   :param title: Optional plot title.
   :type title: str or None, optional

   :returns: Figure handle containing the rendered heatmap.
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_prediction(datastore: neural_lam.datastore.base.BaseRegularGridDatastore, da_prediction: xarray.DataArray, da_target: xarray.DataArray, title=None, vrange=None)

   Plot a prediction alongside the corresponding ground truth field.

   :param datastore: Datastore providing coordinate metadata and projection details.
   :type datastore: BaseRegularGridDatastore
   :param da_prediction: Predicted field flattened over the grid.

                         * **Shape**: ``(N_grid,)``
   :type da_prediction: xarray.DataArray
   :param da_target: Ground-truth field flattened over the grid.

                     * **Shape**: ``(N_grid,)``
   :type da_target: xarray.DataArray
   :param title: Optional figure title.
   :type title: str or None, optional
   :param vrange: Explicit value range ``(vmin, vmax)`` for the color scale.
   :type vrange: tuple[float, float] or None, optional

   :returns: Figure handle containing the two subplots.
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_spatial_error(error, datastore: neural_lam.datastore.base.BaseRegularGridDatastore, title=None, vrange=None)

   Plot spatial error magnitudes on the datastore grid.

   :param error: Error magnitudes on the flattened grid.

                 * **Shape**: ``(N_grid,)``
   :type error: torch.Tensor
   :param datastore: Datastore providing coordinate metadata and boundary masks.
   :type datastore: BaseRegularGridDatastore
   :param title: Optional figure title.
   :type title: str or None, optional
   :param vrange: Explicit value range ``(vmin, vmax)`` for the color scale.
   :type vrange: tuple[float, float] or None, optional

   :returns: Figure handle containing the plotted map.
   :rtype: matplotlib.figure.Figure


