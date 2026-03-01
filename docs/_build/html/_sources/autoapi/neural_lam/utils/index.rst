neural_lam.utils
================

.. py:module:: neural_lam.utils

.. autoapi-nested-parse::

   Utility helpers shared across Neural-LAM training and evaluation.







Module Contents
---------------

.. py:class:: BufferList(buffer_tensors, persistent=True)

   Bases: :py:obj:`torch.nn.Module`


   A list of torch buffer tensors that sit together as a Module with no
   parameters and only buffers.

   This should be replaced by a native torch BufferList once implemented.
   See: https://github.com/pytorch/pytorch/issues/37386

   Register a collection of tensors as buffers inside a module.

   :param buffer_tensors: Buffers to register in the order they should be indexed.
   :type buffer_tensors: Sequence[torch.Tensor]
   :param persistent: If ``True``, buffers are saved in checkpoints. Default ``True``.
   :type persistent: bool, optional


   .. py:attribute:: n_buffers


.. py:function:: fractional_plot_bundle(fraction)

   Return a ``tueplots`` bundle scaled to a fraction of the page width.

   :param fraction: Denominator applied to the default NeurIPS figure width.
   :type fraction: float

   :returns: Matplotlib rcParams bundle with updated ``figure.figsize``.
   :rtype: dict


.. py:function:: get_integer_time(tdelta) -> tuple[int, str]

   Express a :class:`datetime.timedelta` as an integer number of time units.

   :param tdelta: Time interval to convert.
   :type tdelta: datetime.timedelta

   :returns: Integer value and the corresponding unit (e.g. ``"hours"``). If no
             unit yields an integer count, ``(1, "unknown")`` is returned.
   :rtype: tuple[int, str]

   .. rubric:: Examples

   >>> from datetime import timedelta
   >>> get_integer_time(timedelta(days=14))
   (2, 'weeks')
   >>> get_integer_time(timedelta(hours=5))
   (5, 'hours')
   >>> get_integer_time(timedelta(milliseconds=1000))
   (1, 'seconds')
   >>> get_integer_time(timedelta(days=0.001))
   (1, 'unknown')


.. py:function:: has_working_latex()

   Check whether a LaTeX toolchain is available on the system.

   :returns: ``True`` if ``latex`` and the required auxiliary tools are callable.
   :rtype: bool


.. py:function:: init_training_logger_metrics(training_logger, val_steps)

   Configure validation metric aggregation for the active training logger.

   :param training_logger: Logger instance used during training.
   :type training_logger: pytorch_lightning.loggers.Logger
   :param val_steps: Autoregressive rollout lengths to log as separate metrics.
   :type val_steps: Iterable[int]


.. py:function:: inverse_sigmoid(x)

   Compute the logit (inverse sigmoid) while clamping to ``(0, 1)``.

   :param x: Input tensor assumed to contain logits after a sigmoid.
   :type x: torch.Tensor

   :returns: Tensor containing ``log(x / (1 - x))`` after clamping away from the
             saturation limits.
   :rtype: torch.Tensor


.. py:function:: inverse_softplus(x, beta=1, threshold=20)

   Approximate the inverse of :func:`torch.nn.functional.softplus`.

   :param x: Input tensor whose softplus inverse should be computed.
   :type x: torch.Tensor
   :param beta: Softplus ``beta`` parameter that controls the sharpness. Default ``1``.
   :type beta: float, optional
   :param threshold: Threshold applied to the input for numerical stability. Default ``20``.
   :type threshold: float, optional

   :returns: Tensor containing the inverse-softplus values.
   :rtype: torch.Tensor


.. py:function:: load_graph(graph_dir_path, device='cpu')

   Load all tensors representing the graph from `graph_dir_path`.

   Needs the following files for all graphs:
   - m2m_edge_index.pt
   - g2m_edge_index.pt
   - m2g_edge_index.pt
   - m2m_features.pt
   - g2m_features.pt
   - m2g_features.pt
   - mesh_features.pt

   And in addition for hierarchical graphs:
   - mesh_up_edge_index.pt
   - mesh_down_edge_index.pt
   - mesh_up_features.pt
   - mesh_down_features.pt

   :param graph_dir_path: Path to directory containing the graph files.
   :type graph_dir_path: str
   :param device: Device to load tensors to.
   :type device: str

   :returns: * **hierarchical** (*bool*) -- Whether the graph is hierarchical.
             * **graph** (*dict*) -- Dictionary containing the graph tensors, with keys as follows:
               - g2m_edge_index
               - m2g_edge_index
               - m2m_edge_index
               - mesh_up_edge_index
               - mesh_down_edge_index
               - g2m_features
               - m2g_features
               - m2m_features
               - mesh_up_features
               - mesh_down_features
               - mesh_static_features


.. py:function:: make_mlp(blueprint, layer_norm=True)

   Construct a multilayer perceptron from a blueprint of layer widths.

   :param blueprint: Sequence of layer dimensions where ``blueprint[0]`` is the input size
                     and ``blueprint[-1]`` is the output size.
   :type blueprint: list[int]
   :param layer_norm: If ``True``, append a ``LayerNorm`` to the output as in GraphCast.
   :type layer_norm: bool, optional

   :returns: Sequential module implementing the specified MLP.
   :rtype: torch.nn.Sequential


.. py:function:: rank_zero_print(*args, **kwargs)

   Print arguments only from the rank-zero process in distributed runs.


.. py:function:: setup_training_logger(datastore, args, run_name)

   Instantiate the configured experiment logger.

   :param datastore: Datastore providing metadata for logging configuration.
   :type datastore: BaseDatastore
   :param args: Parsed training arguments controlling the logger backend.
   :type args: argparse.Namespace
   :param run_name: Name of the run.
   :type run_name: str

   :returns: **logger** -- Logger object.
   :rtype: pytorch_lightning.loggers.base


