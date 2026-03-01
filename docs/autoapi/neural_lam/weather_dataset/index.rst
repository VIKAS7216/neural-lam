neural_lam.weather_dataset
==========================

.. py:module:: neural_lam.weather_dataset

.. autoapi-nested-parse::

   Dataset helpers wrapping Neural-LAM datastores for PyTorch Lightning.





Module Contents
---------------

.. py:class:: WeatherDataModule(datastore: neural_lam.datastore.base.BaseDatastore, ar_steps_train: int = 3, ar_steps_eval: int = 25, standardize: bool = True, num_past_forcing_steps: int = 1, num_future_forcing_steps: int = 1, batch_size: int = 4, num_workers: int = 16, eval_split: str = 'test')

   Bases: :py:obj:`pytorch_lightning.LightningDataModule`


   DataModule for weather data.

   :param datastore: Datastore used for all splits.
   :type datastore: BaseDatastore
   :param ar_steps_train: Number of AR steps for training batches. Default ``3``.
   :type ar_steps_train: int, optional
   :param ar_steps_eval: Number of AR steps for validation/test batches. Default ``25``.
   :type ar_steps_eval: int, optional
   :param standardize: If ``True``, datasets are returned standardized. Default ``True``.
   :type standardize: bool, optional
   :param num_past_forcing_steps: Number of past forcing steps to include. Default ``1``.
   :type num_past_forcing_steps: int, optional
   :param num_future_forcing_steps: Number of future forcing steps to include. Default ``1``.
   :type num_future_forcing_steps: int, optional
   :param batch_size: Mini-batch size for dataloaders. Default ``4``.
   :type batch_size: int, optional
   :param num_workers: Number of background workers per dataloader. Default ``16``.
   :type num_workers: int, optional
   :param eval_split: Dataset split to use for ``test_dataloader``. Default ``"test"``.
   :type eval_split: str, optional


   .. py:method:: setup(stage=None)

      Instantiate datasets for the requested trainer stage.

      :param stage: Trainer stage identifier (``"fit"``/``"test"``/``None``). When
                    ``None``, both train and evaluation datasets are created.
      :type stage: str or None, optional



   .. py:method:: test_dataloader()

      Load test dataset.



   .. py:method:: train_dataloader()

      Load train dataset.



   .. py:method:: val_dataloader()

      Load validation dataset.



   .. py:attribute:: ar_steps_eval
      :value: 25



   .. py:attribute:: ar_steps_train
      :value: 3



   .. py:attribute:: batch_size
      :value: 4



   .. py:attribute:: eval_split
      :value: 'test'



   .. py:attribute:: multiprocessing_context
      :type:  Union[str, None]
      :value: None



   .. py:attribute:: num_future_forcing_steps
      :value: 1



   .. py:attribute:: num_past_forcing_steps
      :value: 1



   .. py:attribute:: num_workers
      :type:  int
      :value: 16



   .. py:attribute:: standardize
      :value: True



   .. py:attribute:: test_dataset
      :value: None



   .. py:attribute:: train_dataset
      :value: None



   .. py:attribute:: val_dataset
      :value: None



.. py:class:: WeatherDataset(datastore: neural_lam.datastore.base.BaseDatastore, split: str = 'train', ar_steps: int = 3, num_past_forcing_steps: int = 1, num_future_forcing_steps: int = 1, standardize: bool = True)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset class for weather data.

   This class loads and processes weather data from a given datastore.

   :param datastore: The datastore to load the data from (e.g. mdp).
   :type datastore: BaseDatastore
   :param split: The data split to use ("train", "val" or "test"). Default is "train".
   :type split: str, optional
   :param ar_steps: The number of autoregressive steps. Default is 3.
   :type ar_steps: int, optional
   :param num_past_forcing_steps: Number of past time steps to include in forcing input. If set to i,
                                  forcing from times t-i, t-i+1, ..., t-1, t (and potentially beyond,
                                  given num_future_forcing_steps) are included as forcing inputs at time t
                                  Default is 1.
   :type num_past_forcing_steps: int, optional
   :param num_future_forcing_steps: Number of future time steps to include in forcing input. If set to j,
                                    forcing from times t, t+1, ..., t+j-1, t+j (and potentially times before
                                    t, given num_past_forcing_steps) are included as forcing inputs at time
                                    t. Default is 1.
   :type num_future_forcing_steps: int, optional
   :param standardize: Whether to standardize the data. Default is True.
   :type standardize: bool, optional
   :param :
   :type : param datastore: Datastore providing access to state/forcing/static arrays.
   :param :
   :type : type datastore: BaseDatastore
   :param : Default ``"train"``.
   :type : param split: Data split (``"train"``, ``"val"``, or ``"test"``).
   :param :
   :type : type split: str, optional
   :param :
   :type : param ar_steps: Number of autoregressive steps per training sample. Default ``3``.
   :param :
   :type : type ar_steps: int, optional
   :param : are concatenated. Default ``1``.
   :type : param num_past_forcing_steps: Past forcing window length ``i`` so that ``[t-i, ..., t]`` forcings
   :param :
   :type : type num_past_forcing_steps: int, optional
   :param : forcings are available. Default ``1``.
   :type : param num_future_forcing_steps: Future forcing window length ``j`` so that ``[t, ..., t+j]``
   :param :
   :type : type num_future_forcing_steps: int, optional
   :param :
   :type : param standardize: If ``True``, normalize state/forcing arrays via datastore stats.
   :param :
   :type : type standardize: bool, optional


   .. py:method:: create_dataarray_from_tensor(tensor: torch.Tensor, time: Union[datetime.datetime, list[datetime.datetime]], category: str)

      Construct a xarray.DataArray from a `pytorch.Tensor` with coordinates
      for `grid_index`, `time` and `{category}_feature` matching the shape
      and number of times provided and add the x/y coordinates from the
      datastore.

      The number if times provided is expected to match the shape of the
      tensor. For a 2D tensor, the dimensions are assumed to be (grid_index,
      {category}_feature) and only a single time should be provided. For a 3D
      tensor, the dimensions are assumed to be (time, grid_index,
      {category}_feature) and a list of times should be provided.

      :param tensor: The tensor to construct the DataArray from, this assumed to have
                     the same dimension ordering as returned by the __getitem__ method
                     (i.e. time, grid_index, {category}_feature). The tensor will be
                     copied to the CPU before constructing the DataArray.
      :type tensor: torch.Tensor
      :param time: The time or times of the tensor.
      :type time: datetime.datetime or list[datetime.datetime]
      :param category: The category of the tensor, either "state", "forcing" or "static".
      :type category: str

      :returns: **da** -- The constructed DataArray.
      :rtype: xr.DataArray



   .. py:attribute:: ar_steps
      :value: 3



   .. py:attribute:: da_forcing


   .. py:attribute:: da_state


   .. py:attribute:: datastore


   .. py:attribute:: num_future_forcing_steps
      :value: 1



   .. py:attribute:: num_past_forcing_steps
      :value: 1



   .. py:attribute:: split
      :value: 'train'



   .. py:attribute:: standardize
      :value: True



