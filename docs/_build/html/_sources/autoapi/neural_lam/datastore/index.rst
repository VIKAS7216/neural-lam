neural_lam.datastore
====================

.. py:module:: neural_lam.datastore

.. autoapi-nested-parse::

   Datastore backends for loading and serving weather model data.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/neural_lam/datastore/base/index
   /autoapi/neural_lam/datastore/mdp/index
   /autoapi/neural_lam/datastore/npyfilesmeps/index
   /autoapi/neural_lam/datastore/plot_example/index






Package Contents
----------------

.. py:function:: init_datastore(datastore_kind, config_path)

   Instantiate a datastore based on its short-name identifier.

   :param datastore_kind: Key corresponding to one of :data:`DATASTORES`.
   :type datastore_kind: str
   :param config_path: Path to the datastore-specific configuration file.
   :type config_path: str | pathlib.Path

   :returns: Concrete datastore instance configured for ``config_path``.
   :rtype: BaseDatastore

   :raises NotImplementedError: If ``datastore_kind`` is not registered.


.. py:data:: DATASTORES

.. py:data:: DATASTORE_CLASSES

