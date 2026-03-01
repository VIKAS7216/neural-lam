neural_lam.custom_loggers
=========================

.. py:module:: neural_lam.custom_loggers

.. autoapi-nested-parse::

   Custom logging utilities (e.g., MLFlow wrappers) used in Neural-LAM.





Module Contents
---------------

.. py:class:: CustomMLFlowLogger(experiment_name, tracking_uri, run_name)

   Bases: :py:obj:`pytorch_lightning.loggers.MLFlowLogger`


   Custom MLFlow logger that adds the `log_image()` functionality not
   present in the default implementation from pytorch-lightning as
   of version `2.0.3` at least.

   Initialize the logger and start an MLflow run.

   :param experiment_name: Target MLflow experiment.
   :type experiment_name: str
   :param tracking_uri: MLflow tracking server URI.
   :type tracking_uri: str
   :param run_name: Human-readable run name stored as ``mlflow.runName``.
   :type run_name: str


   .. py:method:: log_image(key, images, step=None)

      Log one or more Matplotlib figures as images in MLflow.

      :param key: Identifier under which to log the image.
      :type key: str
      :param images: Figures to export; only the first element is logged.
      :type images: Sequence[matplotlib.figure.Figure]
      :param step: Optional training step index appended to ``key``.
      :type step: int or None, optional

      :raises SystemExit: If AWS credentials for the MLflow artifact store are missing.



   .. py:property:: save_dir

      Returns the directory where the MLFlow artifacts are saved.
      Used to define the path to save output when using the logger.

      :returns: Path to the directory where the artifacts are saved.
      :rtype: str


