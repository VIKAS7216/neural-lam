neural_lam.metrics
==================

.. py:module:: neural_lam.metrics

.. autoapi-nested-parse::

   Evaluation metrics shared across training and validation routines.







Module Contents
---------------

.. py:function:: crps_gauss(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Compute the (negative) Continuous Ranked Probability Score (CRPS).

   A closed-form expression for a Gaussian predictive distribution is used.

   :param pred: Distribution mean predictions.

                * **Shape**: ``(..., N, d_state)``
   :type pred: torch.Tensor
   :param target: Ground-truth values.

                  * **Shape**: ``(..., N, d_state)``
   :type target: torch.Tensor
   :param pred_std: Predicted standard deviation parameter of the Gaussian.

                    * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
   :type pred_std: torch.Tensor
   :param mask: Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None, optional
   :param average_grid: If ``True``, average over the grid dimension. Default is ``True``.
   :type average_grid: bool, optional
   :param sum_vars: If ``True``, sum over the variable dimension. Default is ``True``.
   :type sum_vars: bool, optional

   :returns: Negative CRPS values with shape determined by ``average_grid`` and
             ``sum_vars``.
   :rtype: torch.Tensor


.. py:function:: get_metric(metric_name)

   Retrieve a registered metric function by name.

   :param metric_name: Name of the metric to load (case-insensitive).
   :type metric_name: str

   :returns: Metric function implementing the requested metric.
   :rtype: callable

   :raises AssertionError: If ``metric_name`` is not part of :data:`DEFINED_METRICS`.


.. py:function:: mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Compute the unweighted Mean Absolute Error (MAE).

   :param pred: Model predictions.

                * **Shape**: ``(..., N, d_state)``
   :type pred: torch.Tensor
   :param target: Ground-truth values.

                  * **Shape**: ``(..., N, d_state)``
   :type target: torch.Tensor
   :param pred_std: Unused argument for compatibility with :func:`wmae`.

                    * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
   :type pred_std: torch.Tensor
   :param mask: Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None, optional
   :param average_grid: If ``True``, average over the grid dimension. Default is ``True``.
   :type average_grid: bool, optional
   :param sum_vars: If ``True``, sum over the variable dimension. Default is ``True``.
   :type sum_vars: bool, optional

   :returns: MAE with shape determined by ``average_grid`` and ``sum_vars``.
   :rtype: torch.Tensor


.. py:function:: mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars)

   Apply a spatial mask and optionally reduce a per-entry metric tensor.

   :param metric_entry_vals: Entry-wise metric values.

                             * **Shape**: ``(..., N, d_state)`` where ``...`` are broadcastable
                               leading dimensions.
   :type metric_entry_vals: torch.Tensor
   :param mask: Boolean mask selecting which grid nodes to include. Pass ``None`` to
                use all nodes.

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None
   :param average_grid: If ``True``, reduce the grid dimension ``N`` by taking the mean,
                        producing ``(..., d_state)``.
   :type average_grid: bool
   :param sum_vars: If ``True``, reduce the variable dimension ``d_state`` by summing,
                    producing ``(..., N)`` or ``(...,)`` depending on ``average_grid``.
   :type sum_vars: bool

   :returns: Reduced metric tensor.

             * **Shape**: one of ``(...,)``, ``(..., d_state)``, ``(..., N)``, or
               ``(..., N, d_state)`` depending on the reduction flags.
   :rtype: torch.Tensor


.. py:function:: mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Compute the unweighted Mean Squared Error (MSE).

   :param pred: Model predictions.

                * **Shape**: ``(..., N, d_state)``
   :type pred: torch.Tensor
   :param target: Ground-truth values.

                  * **Shape**: ``(..., N, d_state)``
   :type target: torch.Tensor
   :param pred_std: Unused argument for API parity with :func:`wmse`.

                    * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
   :type pred_std: torch.Tensor
   :param mask: Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None, optional
   :param average_grid: If ``True``, average over the grid dimension. Default is ``True``.
   :type average_grid: bool, optional
   :param sum_vars: If ``True``, sum over the variable dimension. Default is ``True``.
   :type sum_vars: bool, optional

   :returns: MSE with shape determined by ``average_grid`` and ``sum_vars``.
   :rtype: torch.Tensor


.. py:function:: nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Compute the Negative Log Likelihood for an isotropic Gaussian likelihood.

   :param pred: Distribution mean predictions.

                * **Shape**: ``(..., N, d_state)``
   :type pred: torch.Tensor
   :param target: Ground-truth values.

                  * **Shape**: ``(..., N, d_state)``
   :type target: torch.Tensor
   :param pred_std: Predicted standard deviation parameter of the Gaussian.

                    * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
   :type pred_std: torch.Tensor
   :param mask: Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None, optional
   :param average_grid: If ``True``, average over the grid dimension. Default is ``True``.
   :type average_grid: bool, optional
   :param sum_vars: If ``True``, sum over the variable dimension. Default is ``True``.
   :type sum_vars: bool, optional

   :returns: Negative log-likelihood with shape determined by ``average_grid`` and
             ``sum_vars``.
   :rtype: torch.Tensor


.. py:function:: wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Compute the Weighted Mean Absolute Error (wMAE).

   :param pred: Model predictions.

                * **Shape**: ``(..., N, d_state)``
   :type pred: torch.Tensor
   :param target: Ground-truth values.

                  * **Shape**: ``(..., N, d_state)``
   :type target: torch.Tensor
   :param pred_std: Predicted standard deviation used as the per-entry weighting.

                    * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
   :type pred_std: torch.Tensor
   :param mask: Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None, optional
   :param average_grid: If ``True``, average over the grid dimension. Default is ``True``.
   :type average_grid: bool, optional
   :param sum_vars: If ``True``, sum over the variable dimension. Default is ``True``.
   :type sum_vars: bool, optional

   :returns: Weighted MAE with shape determined by ``average_grid`` and
             ``sum_vars``.
   :rtype: torch.Tensor


.. py:function:: wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Compute the Weighted Mean Squared Error (wMSE).

   Scales the squared error at each grid node and variable by the inverse
   variance ``1 / pred_std**2``, then applies masking and reduction via
   :func:`mask_and_reduce_metric`.

   :param pred: Model predictions.

                * **Shape**: ``(..., N, d_state)``
   :type pred: torch.Tensor
   :param target: Ground-truth values.

                  * **Shape**: ``(..., N, d_state)``
   :type target: torch.Tensor
   :param pred_std: Predicted standard deviation used as the per-entry weighting.

                    * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
   :type pred_std: torch.Tensor
   :param mask: Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

                * **Shape**: ``(N,)``
   :type mask: torch.Tensor or None, optional
   :param average_grid: If ``True``, average over the grid dimension. Default is ``True``.
   :type average_grid: bool, optional
   :param sum_vars: If ``True``, sum over the variable dimension. Default is ``True``.
   :type sum_vars: bool, optional

   :returns: Weighted MSE after masking and reduction (see
             :func:`mask_and_reduce_metric`).

             * **Shape**: determined by ``average_grid`` and ``sum_vars``.
   :rtype: torch.Tensor


.. py:data:: DEFINED_METRICS

