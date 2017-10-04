.. _json_format:

JSON Format Specification
=========================

Example
-------

The following is an example JSON configuration file for mdcgenpy:

.. code-block :: json

	{
	    "n_samples": 3000,
	    "n_feats": 2,
	    "k": 3,
	    "possible_distributions":["gaussian", "uniform"],
	    "mv": true,
	    "corr": 0.0,
	    "compactness_factor": 0.1,
	    "alpha_n": 1,
	    "outliers": 50,
	    "rotate": true,
	    "clusters": [
	        {"distributions": "uniform", "corr": 0.5},
	        {},
	        {"mv": null, "rotate": false}
	    ]
	}

As is usual in mdcgenpy, all parameters are optional.

General Format
--------------

In general, the format of an input JSON file must be something of this type:

.. code-block :: none

	{
	  GENERATOR_PARAM_1: VAL,
	  GENERATOR_PARAM_2: VAL,
	  ...
	  GENERATOR_PARAM_N: VAL,
	  "clusters": [
	    {CLUSTER_1_PARAM_1: VAL, ..., CLUSTER_1_PARAM_N: VAL},
	    {CLUSTER_2_PARAM_1: VAL, ..., CLUSTER_2_PARAM_N: VAL},
	    ...
	    {CLUSTER_M_PARAM_1: VAL, ..., CLUSTER_M_PARAM_N: VAL},
	  ]
	}

The generator parameters are as defined in the :py:class:`Cluster Generator class <mdcgenpy.clusters.ClusterGenerator>`
.
The ``"clusters"`` keyword is for overriding the generator parameters for specific clusters.

All the parameters in the JSON file are optional (including the ``"clusters"``  keyword).

Overriding Parameters for Specific Clusters
-------------------------------------------

After the generator parameters, there is an (optional) ``"clusters"`` keyword.
If the ``"clusters"`` keyword is supplied, a list of at most the same length as the number of clusters must be supplied.
Each element of this list contains cluster-specific parameters, which overrule the general parameters of the cluster
generator, for that cluster.

For a complete list of parameters which are acceptable for specific clusters, check
:py:data:`~mdcgenpy.clusters.Cluster.settables`.
