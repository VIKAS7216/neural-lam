neural_lam.models.base_hi_graph_model
=====================================

.. py:module:: neural_lam.models.base_hi_graph_model

.. autoapi-nested-parse::

   Base implementations for hierarchical (multi-level) graph models.





Module Contents
---------------

.. py:class:: BaseHiGraphModel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_graph_model.BaseGraphModel`


   Base class for hierarchical graph models.

   Extend :class:`BaseGraphModel` with hierarchical mesh structures.


   .. py:method:: embedd_mesh_nodes()

      Embed static mesh features for the bottom level of the hierarchy.

      :returns: Embedded representations for the base-level mesh nodes.

                * **Shape**: ``(num_mesh_nodes[0], d_h)``
      :rtype: torch.Tensor



   .. py:method:: get_num_mesh()

      Compute mesh node counts used for encoding and decoding.

      :returns: Total number of mesh nodes and the number to ignore during
                encoding/decoding.
      :rtype: tuple[int, int]



   .. py:method:: hi_processor_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)
      :abstractmethod:


      Internal processor step executed between mesh init and read-out.

      :param mesh_rep_levels: Mesh representations for each level.

                              * **Shape**: ``(B, num_mesh_nodes[l], d_h)``
      :type mesh_rep_levels: list[torch.Tensor]
      :param mesh_same_rep: Same-level edge representations per level.

                            * **Shape**: ``(B, M_same[l], d_h)``
      :type mesh_same_rep: list[torch.Tensor]
      :param mesh_up_rep: Edge representations from level ``l`` to ``l+1``.

                          * **Shape**: ``(B, M_up[l -> l+1], d_h)``
      :type mesh_up_rep: list[torch.Tensor]
      :param mesh_down_rep: Edge representations from level ``l+1`` down to ``l``.

                            * **Shape**: ``(B, M_down[l <- l+1], d_h)``
      :type mesh_down_rep: list[torch.Tensor]

      :returns: * *tuple[* -- list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
                  list[torch.Tensor]
                * *]* -- Updated representations for (mesh, same-level, up edges, down edges)
                  in that order.



   .. py:method:: process_step(mesh_rep)

      Run the processor portion of the hierarchical encode-process-decode.

      :param mesh_rep: Base-level mesh representations prior to the processor.

                       * **Shape**: ``(B, num_mesh_nodes, d_h)``
      :type mesh_rep: torch.Tensor

      :returns: Updated base-level mesh representations.

                * **Shape**: ``(B, num_mesh_nodes, d_h)``
      :rtype: torch.Tensor



   .. py:attribute:: level_mesh_sizes


   .. py:attribute:: mesh_down_embedders


   .. py:attribute:: mesh_embedders


   .. py:attribute:: mesh_init_gnns


   .. py:attribute:: mesh_read_gnns


   .. py:attribute:: mesh_same_embedders


   .. py:attribute:: mesh_up_embedders


   .. py:attribute:: num_levels


