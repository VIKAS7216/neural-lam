# Neural-LAM

**Graph-native neural weather prediction for Limited Area Modeling.**
Powered by PyTorch, PyTorch Lightning, and PyG.

Neural-LAM brings graph learning to kilometer-scale regional weather 
prediction. The toolkit focuses on Limited Area Modeling workloads, 
providing strong defaults for constrained domains while staying easily 
extensible for new sensors, datastores, and architectures.

- Built for **research velocity** — curated tutorials, minimal boilerplate, 
  and batteries-included configs for common experiments.
- Tuned for **operational realism** — explicit boundary handling, 
  datastore validation, and reproducible preprocessing pipelines.
- Rooted in **open science** — exhaustive API docs and transparent training 
  recipes.
```{admonition} Quick install
:class: tip
pip install neural_lam
```
```{admonition} Paper Reference
:class: note
For the scientific background behind the models implemented here, see
**"Building Machine Learning Limited Area Models: Kilometer-Scale Weather 
Forecasting in Realistic Settings"** (Adamov et al., 2025) — 
[arXiv:2504.09340](https://arxiv.org/abs/2504.09340)
```

---

## Getting Started

| Goal | Resource |
|---|---|
| Spin up the workflow end-to-end | [Hello World: Training on DANRA](notebooks/hello_world_danra) |
| Understand the canonical datastore | [Datastore Explorer](notebooks/datastore_explorer) |
| Integrate a custom dataset | [Adding a New Datastore](notebooks/adding_new_datastore) |
| Dive into every module | [Full API Reference](autoapi/index) |

Each tutorial layers on the previous one: Hello World produces the zarr 
datastore that Datastore Explorer inspects, and the datastore deep dive 
underpins Adding a New Datastore.

---

## Tutorials

### [Hello World: Training on DANRA](notebooks/hello_world_danra)

Fastest route from install to a trained checkpoint:

- Bootstrap the environment with `uv` and fetch a public DANRA subset via 
  `mllam-data-prep` — no local mirror required.
- Generate the single-level `1level` graph, prepare features, and train a 
  `graph_lam` baseline for one CPU epoch.
- Inspect metrics and checkpoints to validate the full pipeline.

Ideal for smoke-testing your setup before scaling up or adding GPUs.

---

### [Datastore Explorer](notebooks/datastore_explorer)

Build intuition for the zarr datastore format that underpins neural-lam:

- Decode the flattened `grid_index` back to true 2D geometry for any 
  non-square domain.
- Review canonical dimensions, coordinates, time splits, and 
  normalisation statistics.
- Visualize every state, forcing, and static feature with production-ready 
  routines.

Spend time here before adapting a new dataset; it prevents downstream 
errors.

---

### [Adding a New Datastore](notebooks/adding_new_datastore)

A rigorous playbook for onboarding external data:

- Walk through the `BaseDatastore` contract and the config-driven 
  `MDPDatastore` path.
- Extend the `BaseRegularGridDatastore` template for bespoke projections or 
  irregular formats.
- Avoid common pitfalls: naming conventions, projections, boundary masks, 
  and standardisation variables.
- Validate implementations with `validate_datastore()` before training.

Required reading for contributors integrating novel data sources.

---

### [Creating meps_example_reduced](notebooks/create_reduced_meps_dataset)

Documented recipe for the bundled MEPS sample dataset. Follow the same 
coordinate subsetting, time-window selection, and export steps when 
preparing reduced regional datasets for development or CI workflows.

---

## API Reference

**[Explore the full API](autoapi/index)** — every public class, method, 
function, and module is rendered with NumPy-style docstrings and cross 
links.

- [`neural_lam.models`](autoapi/neural_lam/models/index) — GraphLAM, HiLAM, HiLAMParallel
- [`neural_lam.datastore`](autoapi/neural_lam/datastore/index) — BaseDatastore, MDPDatastore, NpyFilesDatastoreMEPS
- [`neural_lam.config`](autoapi/neural_lam/config/index) — configuration loading and validation
- [`neural_lam.metrics`](autoapi/neural_lam/metrics/index) — WMSE, MAE, and metric utilities

---

## Community & Support

- **[Slack](https://ml-lam.slack.com/)** — join real-time research and operations discussions.
- **[Issues](https://github.com/mllam/neural-lam/issues)** — report bugs, request features, and track progress.
- **[GitHub](https://github.com/mllam/neural-lam)** — browse the source, open pull requests, and follow releases.
