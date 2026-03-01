neural_lam.models.hi_lam
========================

.. py:module:: neural_lam.models.hi_lam

.. autoapi-nested-parse::

   Sequential up/down hierarchical Neural-LAM model (Hi-LAM).





Module Contents
---------------

.. py:class:: HiLAM(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_hi_graph_model.BaseHiGraphModel`


   Hierarchical graph model with message passing that goes sequentially down
   and up the hierarchy during processing.
   The Hi-LAM model from Oskarsson et al. (2023)

   Initialize the sequential up/down hierarchical processor.


   .. py:method:: hi_processor_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)

      Execute one full processor iteration (down + up sweeps).

      :param mesh_rep_levels: Mesh representations for each level.

                              * **Shape**: ``(B, N_mesh[l], d_h)``
      :type mesh_rep_levels: list[torch.Tensor]
      :param mesh_same_rep: Same-level edge representations.

                            * **Shape**: ``(B, M_same[l], d_h)``
      :type mesh_same_rep: list[torch.Tensor]
      :param mesh_up_rep: Upward edge representations.

                          * **Shape**: ``(B, M_up[l], d_h)``
      :type mesh_up_rep: list[torch.Tensor]
      :param mesh_down_rep: Downward edge representations.

                            * **Shape**: ``(B, M_down[l], d_h)``
      :type mesh_down_rep: list[torch.Tensor]

      :returns: * *tuple[* -- list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
                  list[torch.Tensor]
                * *]* -- Updated representations ``(mesh_rep_levels, mesh_same_rep,
                  mesh_up_rep, mesh_down_rep)`` after both sweeps.



   .. py:method:: make_down_gnns(args)

      Make GNNs for processing steps down through the hierarchy.



   .. py:method:: make_same_gnns(args)

      Make intra-level GNNs.



   .. py:method:: make_up_gnns(args)

      Make GNNs for processing steps up through the hierarchy.



   .. py:method:: mesh_down_step(mesh_rep_levels, mesh_same_rep, mesh_down_rep, down_gnns, same_gnns)

      Run the downward half of the hierarchical processing sweep.

      :param mesh_rep_levels: Mesh representations for each level.

                              * **Shape**: ``(B, N_mesh[l], d_h)``
      :type mesh_rep_levels: list[torch.Tensor]
      :param mesh_same_rep: Same-level edge representations.

                            * **Shape**: ``(B, M_same[l], d_h)``
      :type mesh_same_rep: list[torch.Tensor]
      :param mesh_down_rep: Downward edge representations.

                            * **Shape**: ``(B, M_down[l], d_h)``
      :type mesh_down_rep: list[torch.Tensor]
      :param down_gnns: Message-passing networks applied to downward edges.
      :type down_gnns: Sequence[InteractionNet]
      :param same_gnns: Message-passing networks for same-level processing.
      :type same_gnns: Sequence[InteractionNet]

      :returns: Updated ``(mesh_rep_levels, mesh_same_rep, mesh_down_rep)``.
      :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]



   .. py:method:: mesh_up_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, same_gnns)

      Run the upward half of the hierarchical processing sweep.

      :param mesh_rep_levels: Mesh representations for each level.

                              * **Shape**: ``(B, N_mesh[l], d_h)``
      :type mesh_rep_levels: list[torch.Tensor]
      :param mesh_same_rep: Same-level edge representations.

                            * **Shape**: ``(B, M_same[l], d_h)``
      :type mesh_same_rep: list[torch.Tensor]
      :param mesh_up_rep: Upward edge representations.

                          * **Shape**: ``(B, M_up[l], d_h)``
      :type mesh_up_rep: list[torch.Tensor]
      :param up_gnns: Message-passing networks applied to upward edges.
      :type up_gnns: Sequence[InteractionNet]
      :param same_gnns: Message-passing networks for same-level processing.
      :type same_gnns: Sequence[InteractionNet]

      :returns: Updated ``(mesh_rep_levels, mesh_same_rep, mesh_up_rep)``.
      :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]



   .. py:attribute:: mesh_down_gnns


   .. py:attribute:: mesh_down_same_gnns


   .. py:attribute:: mesh_up_gnns


   .. py:attribute:: mesh_up_same_gnns


