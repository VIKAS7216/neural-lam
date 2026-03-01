neural_lam.interaction_net
==========================

.. py:module:: neural_lam.interaction_net

.. autoapi-nested-parse::

   Interaction Network layers and helper modules used by Neural-LAM.





Module Contents
---------------

.. py:class:: InteractionNet(edge_index, input_dim, update_edges=True, hidden_layers=1, hidden_dim=None, edge_chunk_sizes=None, aggr_chunk_sizes=None, aggr='sum')

   Bases: :py:obj:`torch_geometric.nn.MessagePassing`


   Implementation of a generic Interaction Network,
   from Battaglia et al. (2016)

   Initialise an InteractionNet message-passing layer.

   :param edge_index: Edge connectivity tensor in PyG format.

                      * **Shape**: ``(2, M)`` where ``M`` is the number of edges.
   :type edge_index: torch.Tensor
   :param input_dim: Dimensionality of both node and edge input representations.
   :type input_dim: int
   :param update_edges: If ``True``, compute and return updated edge representations in
                        addition to node representations. Default is ``True``.
   :type update_edges: bool, optional
   :param hidden_layers: Number of hidden layers in each MLP. Default is ``1``.
   :type hidden_layers: int, optional
   :param hidden_dim: Width of hidden layers. If ``None``, defaults to ``input_dim``.
   :type hidden_dim: int or None, optional
   :param edge_chunk_sizes: Chunk sizes for splitting edge representations across separate
                            MLPs. ``None`` uses a single shared MLP.
   :type edge_chunk_sizes: list[int] or None, optional
   :param aggr_chunk_sizes: Chunk sizes for splitting aggregated node representations across
                            separate MLPs. ``None`` uses a single shared MLP.
   :type aggr_chunk_sizes: list[int] or None, optional
   :param aggr: Message aggregation method. Default is ``"sum"``.
   :type aggr: {"sum", "mean"}, optional

   :raises AssertionError: If ``aggr`` is not one of ``"sum"`` or ``"mean"``.


   .. py:method:: aggregate(inputs, index, ptr, dim_size)

      Aggregate messages while also returning the per-edge values.



   .. py:method:: forward(send_rep, rec_rep, edge_rep)

      Update receiver (and optionally edge) representations via message
      passing.

      :param send_rep: Vector representations of sender nodes.

                       * **Shape**: ``(N_send, d_h)``
      :type send_rep: torch.Tensor
      :param rec_rep: Vector representations of receiver nodes.

                      * **Shape**: ``(N_rec, d_h)``
      :type rec_rep: torch.Tensor
      :param edge_rep: Edge representations used during message passing.

                       * **Shape**: ``(M, d_h)``
      :type edge_rep: torch.Tensor

      :returns: Updated receiver representations. If ``self.update_edges`` is
                ``True``, the tuple ``(rec_rep, edge_rep)`` containing the updated
                receiver and edge representations is returned.

                * **Shape**: ``(N_rec, d_h)`` for receivers and ``(M, d_h)`` for
                  edges.
      :rtype: torch.Tensor or tuple[torch.Tensor, torch.Tensor]



   .. py:method:: message(x_j, x_i, edge_attr)

      Compute messages from node ``j`` to ``i`` using edge features.



   .. py:attribute:: num_rec


   .. py:attribute:: update_edges
      :value: True



.. py:class:: SplitMLPs(mlps, chunk_sizes)

   Bases: :py:obj:`torch.nn.Module`


   Module that feeds chunks of input through different MLPs.
   Split up input along dim -2 using given chunk sizes and feeds
   each chunk through separate MLPs.

   Create a module that dispatches chunks of the input to separate MLPs.

   :param mlps: Sequence of MLPs to apply to each chunk.
   :type mlps: Iterable[nn.Module]
   :param chunk_sizes: Sizes used when splitting the input along ``dim=-2``.
   :type chunk_sizes: Sequence[int]

   :raises AssertionError: If the number of ``mlps`` and ``chunk_sizes`` differ.


   .. py:method:: forward(x)

      Chunk up input tensor and feed each slice through its MLP.

      :param x: Input tensor to split and process.

                * **Shape**: ``(..., N, d)`` where ``N = sum(chunk_sizes)``.
      :type x: torch.Tensor

      :returns: Concatenated MLP outputs assembled along the chunk dimension.

                * **Shape**: ``(..., N, d)``
      :rtype: torch.Tensor



   .. py:attribute:: chunk_sizes


   .. py:attribute:: mlps


