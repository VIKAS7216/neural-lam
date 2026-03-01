neural_lam.datastore.npyfilesmeps.compute_standardization_stats
===============================================================

.. py:module:: neural_lam.datastore.npyfilesmeps.compute_standardization_stats

.. autoapi-nested-parse::

   Utilities for computing MEPS datastore standardization statistics.







Module Contents
---------------

.. py:class:: PaddedWeatherDataset(base_dataset, world_size, batch_size)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Wrap :class:`WeatherDataset` to pad samples for distributed runners.

   :param base_dataset: Dataset to pad.
   :type base_dataset: WeatherDataset
   :param world_size: Total number of distributed ranks participating.
   :type world_size: int
   :param batch_size: Per-rank batch size.
   :type batch_size: int


   .. py:method:: get_original_indices()

      Return indices of the non-padded samples.



   .. py:method:: get_original_window_indices(step_length)

      Return index mapping for sub-sampled windows at ``step_length``.



   .. py:attribute:: base_dataset


   .. py:attribute:: batch_size


   .. py:attribute:: original_indices


   .. py:attribute:: padded_indices


   .. py:attribute:: padded_samples


   .. py:attribute:: total_samples


   .. py:attribute:: world_size


.. py:function:: cli()

   Parse CLI arguments and trigger :func:`main`.


.. py:function:: get_rank()

   Return the rank inferred from SLURM or default to 0.


.. py:function:: get_world_size()

   Return the world size inferred from SLURM or default to 1.


.. py:function:: main(datastore_config_path, batch_size, step_length, n_workers, distributed)

   Pre-compute and persist standardization statistics from the datastore.

   :param datastore_config_path: Path to the MEPS datastore configuration file.
   :type datastore_config_path: str or pathlib.Path
   :param batch_size: Batch size used while iterating through the dataset.
   :type batch_size: int
   :param step_length: Temporal sampling interval for the difference statistics.
   :type step_length: datetime.timedelta
   :param n_workers: Number of dataloader workers.
   :type n_workers: int
   :param distributed: If ``True``, run using torch.distributed with SLURM settings.
   :type distributed: bool


.. py:function:: save_stats(static_dir_path, means, squares, flux_means, flux_squares, filename_prefix)

   Aggregate running statistics and persist them to ``static_dir_path``.

   :param static_dir_path: Directory where ``*.pt`` files should be written.
   :type static_dir_path: str or pathlib.Path
   :param means: Batch-wise means with shape ``(N_batch, d_features)``.
   :type means: Sequence[torch.Tensor]
   :param squares: Batch-wise second moments with shape ``(N_batch, d_features)``.
   :type squares: Sequence[torch.Tensor]
   :param flux_means: Optional flux means of shape ``(N_batch,)``.
   :type flux_means: Sequence[torch.Tensor]
   :param flux_squares: Optional flux second moments of shape ``(N_batch,)``.
   :type flux_squares: Sequence[torch.Tensor]
   :param filename_prefix: Prefix (e.g., ``"parameter"`` or ``"diff"``) for saved tensors.
   :type filename_prefix: str


.. py:function:: setup(rank, world_size)

   Initialize the distributed group.


