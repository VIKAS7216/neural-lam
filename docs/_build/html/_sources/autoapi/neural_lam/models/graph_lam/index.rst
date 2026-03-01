neural_lam.models.graph_lam
===========================

.. py:module:: neural_lam.models.graph_lam

.. autoapi-nested-parse::

   GraphLAM: the non-hierarchical Neural-LAM architecture.





Module Contents
---------------

.. py:class:: GraphLAM(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_graph_model.BaseGraphModel`


   Full graph-based LAM model that can be used with different
   (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
   Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
   Oskarsson et al. (2023).

   Initialize the non-hierarchical GraphLAM variant.


   .. py:method:: embedd_mesh_nodes()

      Embed static mesh features.

      :returns: Embedded mesh node representations.

                * **Shape**: ``(N_mesh, d_h)``
      :rtype: torch.Tensor



   .. py:method:: get_num_mesh()

      Compute mesh node counts used for encoding and decoding.

      :returns: Total number of mesh nodes and the number to ignore during
                encoding/decoding.
      :rtype: tuple[int, int]



   .. py:method:: process_step(mesh_rep)

      Run the processor portion of the encode-process-decode framework.

      :param mesh_rep: Mesh node representations before processing.

                       * **Shape**: ``(B, N_mesh, d_h)``
      :type mesh_rep: torch.Tensor

      :returns: Updated mesh representations.

                * **Shape**: ``(B, N_mesh, d_h)``
      :rtype: torch.Tensor



   .. py:attribute:: m2m_embedder


   .. py:attribute:: mesh_embedder


   .. py:attribute:: processor


