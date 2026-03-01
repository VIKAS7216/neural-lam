neural_lam.models.ar_model
==========================

.. py:module:: neural_lam.models.ar_model

.. autoapi-nested-parse::

   Auto-regressive LightningModule implementations for Neural-LAM.





Module Contents
---------------

.. py:class:: ARModel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`pytorch_lightning.LightningModule`


   Generic auto-regressive weather model.
   Abstract class that can be extended.

   :param args: Parsed training arguments controlling rollout length, loss, etc.
   :type args: argparse.Namespace
   :param config: Experiment configuration containing datastore/training settings.
   :type config: NeuralLAMConfig
   :param datastore: Datastore supplying state/forcing/static arrays.
   :type datastore: BaseDatastore


   .. py:method:: aggregate_and_plot_metrics(metrics_dict, prefix)

      Aggregate metric tensors and create error-map visualisations.

      :param metrics_dict: Mapping from metric name to per-batch tensors of evaluations.
      :type metrics_dict: dict[str, list[torch.Tensor]]
      :param prefix: Prefix to use for logger keys.
      :type prefix: str



   .. py:method:: all_gather_cat(tensor_to_gather)

      Gather tensors across ranks and concatenate along dim-0.

      :param tensor_to_gather: Tensor distributed across ``K`` ranks.

                               * **Shape**: ``(d1, d2, ...)`` per rank
      :type tensor_to_gather: torch.Tensor

      :returns: Concatenated tensor gathered from all ranks.

                * **Shape**: ``(K * d1, d2, ...)``
      :rtype: torch.Tensor



   .. py:method:: common_step(batch)

      Run a forward pass shared by train/val/test steps.

      :param batch: Tuple of ``(init_states, target_states, forcing_features,
                    batch_times)`` produced by :class:`WeatherDataset`.

                    * **init_states**: ``(B, 2, num_grid_nodes, d_features)``
                    * **target_states**: ``(B, pred_steps, num_grid_nodes, d_features)``
                    * **forcing_features**: ``(B, pred_steps, num_grid_nodes,
                      d_forcing)``
                    * **batch_times**: ``(B, pred_steps)`` timestamps
      :type batch: tuple

      :returns: ``(prediction, target_states, pred_std, batch_times)``.

                * **prediction**: ``(B, pred_steps, num_grid_nodes, d_f)``
                * **target_states**: ``(B, pred_steps, num_grid_nodes, d_f)``
                * **pred_std**: ``(B, pred_steps, num_grid_nodes, d_f)`` or
                  ``(d_f,)``
                * **batch_times**: ``(B, pred_steps)``
      :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]



   .. py:method:: configure_optimizers()

      Construct the :class:`torch.optim.AdamW` optimizer for training.



   .. py:method:: create_metric_log_dict(metric_tensor, prefix, metric_name)

      Assemble logging artefacts for a single metric tensor.

      :param metric_tensor: Metric values per time step and variable.

                            * **Shape**: ``(pred_steps, d_f)``
      :type metric_tensor: torch.Tensor
      :param prefix: Prefix used for logger keys (e.g., ``"val"`` or ``"test"``).
      :type prefix: str
      :param metric_name: Human-readable metric name.
      :type metric_name: str

      :returns: Mapping from log keys to figures or scalar tensors.
      :rtype: dict[str, object]



   .. py:method:: expand_to_batch(x, batch_size)
      :staticmethod:


      Broadcast a tensor by prepending a batch dimension.

      :param x: Tensor to expand.
      :type x: torch.Tensor
      :param batch_size: Batch size to broadcast to.
      :type batch_size: int

      :returns: Tensor with a leading batch dimension added via ``expand``.
      :rtype: torch.Tensor



   .. py:method:: on_load_checkpoint(checkpoint)

      Perform any changes to state dict before loading checkpoint



   .. py:method:: on_test_epoch_end()

      Compute test metrics and make plots at the end of test epoch. Will
      gather stored tensors and perform plotting and logging on rank 0.



   .. py:method:: on_validation_epoch_end()

      Compute val metrics at the end of val epoch



   .. py:method:: plot_examples(batch, n_examples, split, prediction=None)

      Plot the first ``n_examples`` forecasts from ``batch``.

      :param batch: Batch tuple produced by the dataloader.
      :type batch: tuple
      :param n_examples: Number of forecasts to visualise.
      :type n_examples: int
      :param split: Dataset split name used for metadata lookups.
      :type split: str
      :param prediction: Pre-computed predictions to plot. If ``None`` the method runs
                         :meth:`common_step` to obtain predictions.

                         * **Shape**: ``(B, pred_steps, num_grid_nodes, d_f)``
      :type prediction: torch.Tensor or None, optional



   .. py:method:: predict_step(prev_state, prev_prev_state, forcing)
      :abstractmethod:


      Advance the state by one step using the prediction model.

      :param prev_state: Current state ``X_t``.

                         * **Shape**: ``(B, num_grid_nodes, feature_dim)``
      :type prev_state: torch.Tensor
      :param prev_prev_state: Previous state ``X_{t-1}``.

                              * **Shape**: ``(B, num_grid_nodes, feature_dim)``
      :type prev_prev_state: torch.Tensor
      :param forcing: Forcing inputs applied at the prediction step.

                      * **Shape**: ``(B, num_grid_nodes, forcing_dim)``
      :type forcing: torch.Tensor

      :returns: Tuple ``(new_state, pred_std)`` describing the next state and
                optional uncertainty estimate.
      :rtype: tuple[torch.Tensor, torch.Tensor | None]



   .. py:method:: test_step(batch, batch_idx)

      Evaluate ``batch`` during testing and log diagnostics.

      :param batch: Batch sampled from the test dataloader.
      :type batch: tuple
      :param batch_idx: Index of the current batch.
      :type batch_idx: int



   .. py:method:: training_step(batch)

      Execute a single optimization step on ``batch``.

      :param batch: Batch sampled from the training dataloader.
      :type batch: tuple



   .. py:method:: unroll_prediction(init_states, forcing_features, true_states)

      Roll out predictions autoregressively over multiple time steps.

      :param init_states: Initial states providing ``X_{t-1}`` and ``X_t``.

                          * **Shape**: ``(B, 2, num_grid_nodes, d_f)``
      :type init_states: torch.Tensor
      :param forcing_features: Forcing inputs aligned with each rollout step.

                               * **Shape**: ``(B, pred_steps, num_grid_nodes, d_static_f)``
      :type forcing_features: torch.Tensor
      :param true_states: Ground-truth states used for boundary replacement.

                          * **Shape**: ``(B, pred_steps, num_grid_nodes, d_f)``
      :type true_states: torch.Tensor

      :returns: Tuple ``(prediction, pred_std)``.

                * **prediction**: ``(B, pred_steps, num_grid_nodes, d_f)``
                * **pred_std**: ``(B, pred_steps, num_grid_nodes, d_f)`` or
                  ``(d_f,)`` when a constant per-feature value is used
      :rtype: tuple[torch.Tensor, torch.Tensor]



   .. py:method:: validation_step(batch, batch_idx)

      Evaluate ``batch`` during validation.

      :param batch: Batch sampled from the validation dataloader.
      :type batch: tuple
      :param batch_idx: Index of the current batch.
      :type batch_idx: int



   .. py:attribute:: args


   .. py:attribute:: feature_weights


   .. py:attribute:: grid_dim


   .. py:property:: interior_mask_bool

      Boolean interior mask identifying non-boundary grid nodes.

      :returns: Boolean mask.

                * **Shape**: ``(N,)``
      :rtype: torch.Tensor


   .. py:attribute:: loss


   .. py:attribute:: n_example_pred


   .. py:attribute:: output_std


   .. py:attribute:: plotted_examples
      :value: 0



   .. py:attribute:: restore_opt


   .. py:attribute:: spatial_loss_maps
      :type:  List[Any]
      :value: []



   .. py:attribute:: test_metrics
      :type:  Dict[str, List]


   .. py:attribute:: val_metrics
      :type:  Dict[str, List]


