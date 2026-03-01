neural_lam.models.hi_lam_parallel
=================================

.. py:module:: neural_lam.models.hi_lam_parallel

.. autoapi-nested-parse::

   Parallel message-passing variant of the Hi-LAM architecture.





Module Contents
---------------

.. py:class:: HiLAMParallel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_hi_graph_model.BaseHiGraphModel`


   Version of HiLAM where all message passing in the hierarchical mesh (up,
   down, inter-level) is ran in parallel.

   This is a somewhat simpler alternative to the sequential message passing
   of Hi-LAM.

   Initialize the parallel hierarchical message-passing processor.


   .. py:method:: hi_processor_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)

      Internal processor step executed between mesh init and read-out.

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
                  mesh_up_rep, mesh_down_rep)`` after the parallel pass.



   .. py:attribute:: edge_split_sections


