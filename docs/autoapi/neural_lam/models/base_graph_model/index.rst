neural_lam.models.base_graph_model
==================================

.. py:module:: neural_lam.models.base_graph_model

.. autoapi-nested-parse::

   Base classes for Neural-LAM graph models.





Module Contents
---------------

.. py:class:: BaseGraphModel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.ar_model.ARModel`


   Base (abstract) class for graph-based models building on
   the encode-process-decode idea.

   Initialize the graph-model scaffolding shared by concrete variants.

   :param args: Training/runtime arguments describing graph paths and widths.
   :type args: argparse.Namespace
   :param config: Experiment configuration for clamping and weighting.
   :type config: NeuralLAMConfig
   :param datastore: Datastore providing static features and metadata (e.g. graph path).
   :type datastore: BaseDatastore


   .. py:method:: embedd_mesh_nodes()
      :abstractmethod:


      Embed static mesh features for downstream processing.

      :returns: Embedded mesh node representations.

                * **Shape**: ``(num_mesh_nodes, d_h)``
      :rtype: torch.Tensor



   .. py:method:: get_clamped_new_state(state_delta, prev_state)

      Clamp predicted deltas and add them to the previous state.

      The clamped values follow
      ``f(f^{-1}(X_t) + model({X_t, X_{t-1}, ...}, forcing))`` so that the
      model learns to emit outputs in the range of the inverse clamping
      function.

      :param state_delta: Predicted change to apply to the previous state.

                          * **Shape**: ``(B, num_grid_nodes, feature_dim)``
      :type state_delta: torch.Tensor
      :param prev_state: Previous state ``X_t``.

                         * **Shape**: ``(B, num_grid_nodes, feature_dim)``
      :type prev_state: torch.Tensor

      :returns: Clamped next state ``X_{t+1}``.

                * **Shape**: ``(B, num_grid_nodes, feature_dim)``
      :rtype: torch.Tensor



   .. py:method:: get_num_mesh()
      :abstractmethod:


      Compute mesh node counts used for encoding and decoding.

      :returns: Total number of mesh nodes and the number that should be ignored
                during encoding/decoding.
      :rtype: tuple[int, int]



   .. py:method:: predict_step(prev_state, prev_prev_state, forcing)

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

      :returns: Tuple ``(new_state, pred_std)`` where ``pred_std`` is ``None`` when
                the model does not emit uncertainty estimates.

                * **Shape**: ``(B, num_grid_nodes, feature_dim)`` for ``new_state``
                  and ``(B, num_grid_nodes, d_f)`` for ``pred_std`` when present.
      :rtype: tuple[torch.Tensor, torch.Tensor | None]



   .. py:method:: prepare_clamping_params(config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

      Prepare per-feature parameters for clamping model outputs.

      :param config: Model and training configuration containing clamping settings.
      :type config: NeuralLAMConfig
      :param datastore: Datastore that provides the ordering of state variables.
      :type datastore: BaseDatastore



   .. py:method:: process_step(mesh_rep)
      :abstractmethod:


      Run the processor portion of the encode-process-decode framework.

      :param mesh_rep: Mesh node representations prior to the processor.

                       * **Shape**: ``(B, num_mesh_nodes, d_h)``
      :type mesh_rep: torch.Tensor

      :returns: Updated mesh representations after processing.

                * **Shape**: ``(B, num_mesh_nodes, d_h)``
      :rtype: torch.Tensor



   .. py:attribute:: encoding_grid_mlp


   .. py:attribute:: g2m_embedder


   .. py:attribute:: g2m_gnn


   .. py:attribute:: grid_embedder


   .. py:attribute:: m2g_embedder


   .. py:attribute:: m2g_gnn


   .. py:attribute:: mlp_blueprint_end


   .. py:attribute:: output_map


