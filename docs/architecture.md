# neural-lam · Architecture

```{note}
All tensor shapes use the notation `(B, T, N_grid, d_h)` —
B = batch · T = timesteps · N_grid = flattened grid nodes · N_mesh = mesh nodes · d_h = hidden width.
```

Neural-LAM ingests gridded analyses, maps them onto a graph, and steps autoregressively
through time using encode-process-decode message-passing networks.

---

## 01 · System Overview

```{mermaid}
%%{init: {'theme':'dark','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8'}}}%%
graph TB
    raw["🗄 Raw Data<br/>Zarr / NpyFiles"]
    base["BaseDataStore<br/>datastore/base.py"]
    impl["MDPDatastore / NpyFilesDatastoreMEPS"]
    wds["WeatherDataset<br/>weather_dataset.py"]
    dl["PyTorch DataLoader"]
    ar["ARModel<br/>models/ar_model.py"]
    bgm["BaseGraphModel<br/>models/base_graph_model.py"]
    variants["GraphLAM / HiLAM / HiLAMParallel"]
    pred["🎯 Predictions"]
    metrics["📊 Metrics & Visualisation"]

    raw --> base --> impl --> wds --> dl --> ar --> bgm --> variants --> pred --> metrics

    classDef data   fill:#1e3a5f,stroke:#38bdf8,color:#bae6fd;
    classDef ds     fill:#2e1065,stroke:#a78bfa,color:#e9d5ff;
    classDef model  fill:#431407,stroke:#fb923c,color:#fed7aa;
    classDef output fill:#052e16,stroke:#34d399,color:#a7f3d0;

    class raw,base,impl data;
    class wds,dl ds;
    class ar,bgm,variants model;
    class pred,metrics output;
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `init_states` | `(B, 2, N_grid, d_state)` | Two consecutive normalised states to warm-start each rollout. |
| `forcing` | `(B, T+2, N_grid, d_forcing)` | Forcing slices with static covariates already concatenated. |
| `target_states` | `(B, T, N_grid, d_state)` | Future trajectory used by `ARModel` to compute the training loss. |

```{note}
Static covariates are concatenated into `forcing` inside `WeatherDataset` — they are **not** returned as a separate tensor.
The batch contract is the 4-tuple `(init_states, target_states, forcing, target_times)`.
```

---

## 02 · Datastore Class Hierarchy

```{mermaid}
%%{init: {'theme':'dark','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8'}}}%%
classDiagram
    BaseDataStore <|-- BaseRegularGridDatastore
    BaseRegularGridDatastore <|-- MDPDatastore
    BaseRegularGridDatastore <|-- NpyFilesDatastoreMEPS

    class BaseDataStore {
        +root_path()
        +config()
        +step_length()
        +get_dataarray(category, split)
        +boundary_mask
        +expected_dim_order()
    }

    class BaseRegularGridDatastore {
        +grid_shape_state
        +get_xy()
        +stack_grid_coords()
        +unstack_grid_coords()
        +num_grid_points()
    }

    class MDPDatastore {
        +__init__()
        +get_dataarray()
        +get_standardization_dataarray()
        +boundary_mask
        +coords_projection()
    }

    class NpyFilesDatastoreMEPS {
        +__init__()
        +get_dataarray()
        +_get_single_timeseries_dataarray()
        +_calc_datetime_forcing_features()
        +get_standardization_dataarray()
        +boundary_mask
    }
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `state_cube` | `(T, N_grid, d_state)` | Stacked state fields from `get_dataarray(category="state")`. |
| `forcing_cube` | `(T, N_grid, d_forcing)` | Forcing arrays including datetime encodings in MEPS. |
| `static_covariates` | `(N_grid, d_static)` | Time-invariant grid features, concatenated into forcing channels by `WeatherDataset`. |

---

## 03 · Autoregressive Unrolling

```{mermaid}
%%{init: {'theme':'dark','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8','activationBkgColor':'#1e3a5f','activationBorderColor':'#38bdf8'}}}%%
sequenceDiagram
    participant WD  as WeatherDataset
    participant DL  as DataLoader
    participant AR  as ARModel
    participant BG  as BaseGraphModel
    participant MT  as Metrics

    WD->>DL:  init_states, target_states, forcing, target_times
    DL->>AR:  batched (init_states, target_states, forcing, target_times)
    loop t = 1 .. T
        AR->>BG:  current_state + forcing_t
        BG-->>AR: delta (residual update)
        AR->>AR:  next_state = current_state + delta
        AR->>MT:  loss(next_state, target_state_t)
    end
    MT-->>AR: total_loss (wMSE)
    AR->>AR: optimizer.step()
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `init_states` | `(B, 2, N_grid, d_state)` | Two consecutive timesteps to warm-start the first AR step. |
| `target_states` | `(B, T, N_grid, d_state)` | Future trajectory the loss compares against. |
| `forcing` | `(B, T+2, N_grid, d_forcing)` | Windowed forcing — 1 past + T current + 1 future step, static covariates included. |
| `target_times` | `(B, T)` | Datetime timestamps for each target step, used for logging and visualisation. |
| `current_state` | `(B, N_grid, d_state)` | Latest prediction fed into the next step. |
| `delta` | `(B, N_grid, d_state)` | **Residual** output — added to `current_state` to produce `next_state`. |

```{note}
The model predicts a **residual delta**, not the next state directly.
`next_state = current_state + delta` is central to training stability during long rollouts.
```

---

## 04 · Encode → Process → Decode

```{mermaid}
%%{init: {'theme':'dark','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#94a3b8'}}}%%
flowchart LR
    classDef stateNode fill:#78350f,stroke:#fbbf24,color:#fef3c7,font-weight:bold;
    classDef encNode   fill:#1e3a5f,stroke:#38bdf8,color:#bae6fd,font-weight:bold;
    classDef procNode  fill:#052e16,stroke:#34d399,color:#a7f3d0,font-weight:bold;
    classDef decNode   fill:#431407,stroke:#fb923c,color:#fed7aa,font-weight:bold;

    prevState["prev_state<br/>(B, N_grid, d_state)"]

    subgraph ENCODE["ENCODE"]
        grid["Grid nodes<br/>(B, N_grid, d_h)"]
        g2m["g2m edges"]
        meshIn["Mesh nodes<br/>(B, N_mesh, d_h)"]
    end

    subgraph PROCESS["PROCESS  ×  N layers"]
        meshLoop["Mesh nodes"]
        m2m["m2m edges"]
        meshOut["Updated mesh<br/>(B, N_mesh, d_h)"]
    end

    subgraph DECODE["DECODE"]
        meshDec["Mesh nodes"]
        m2g["m2g edges"]
        gridOut["Grid nodes<br/>(B, N_grid, d_h)"]
        delta["Residual delta<br/>(B, N_grid, d_state)"]
    end

    nextState["next_state = prev_state + delta<br/>(B, N_grid, d_state)"]

    prevState --> grid
    grid --> g2m --> meshIn --> meshLoop
    meshLoop --> m2m --> meshOut --> meshLoop
    meshOut --> meshDec
    meshDec --> m2g --> gridOut --> delta
    delta --> nextState
    prevState --> nextState

    class prevState,nextState stateNode;
    class grid,g2m,meshIn encNode;
    class meshLoop,m2m,meshOut procNode;
    class meshDec,m2g,gridOut,delta decNode;
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `grid_embed` | `(B, N_grid, d_h)` | State + forcing projected by `grid_embedder`. |
| `mesh_latent` | `(B, N_mesh, d_h)` | Mesh features after `g2m_gnn` message passing (encode). |
| `mesh_updated` | `(B, N_mesh, d_h)` | Mesh features after N rounds of `m2m_gnn` (process). |
| `grid_decoded` | `(B, N_grid, d_h)` | Grid features after `m2g_gnn` message passing (decode). |
| `delta` | `(B, N_grid, d_state)` | Residual output — added to `prev_state` to form `next_state`. |

---

## 05 · HiLAM — Hierarchical Processing

```{mermaid}
%%{init: {'theme':'dark','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#94a3b8'}}}%%
flowchart LR
    classDef gridNode fill:#1e3a5f,stroke:#38bdf8,color:#bae6fd,font-weight:bold;
    classDef meshNode fill:#2e1065,stroke:#a78bfa,color:#e9d5ff,font-weight:bold;
    classDef opNode   fill:#431407,stroke:#fb923c,color:#fed7aa,font-weight:bold;
    classDef procNode fill:#052e16,stroke:#34d399,color:#a7f3d0,font-weight:bold;

    subgraph UPSWEEP["↑ Up Sweep  (finest → coarsest)"]
        direction LR
        GIN["Grid"]
        ENC["Encode<br/>g2m edges"]
        L0U["Mesh L0<br/>finest"]
        MU0["mesh_up<br/>edge_index"]
        L1U["Mesh L1"]
        MU1["mesh_up<br/>edge_index"]
        L2U["Mesh L2<br/>coarsest"]
        GIN --> ENC --> L0U --> MU0 --> L1U --> MU1 --> L2U
    end

    PROC["Process<br/>m2m edges<br/>at top level"]

    subgraph DOWNSWEEP["↓ Down Sweep  (coarsest → finest)"]
        direction LR
        L2D["Mesh L2<br/>coarsest"]
        MD1["mesh_down<br/>edge_index"]
        L1D["Mesh L1"]
        MD2["mesh_down<br/>edge_index"]
        L0D["Mesh L0<br/>finest"]
        DEC["Decode<br/>m2g edges"]
        GOUT["Grid<br/>+ residual"]
        L2D --> MD1 --> L1D --> MD2 --> L0D --> DEC --> GOUT
    end

    L2U --> PROC --> L2D

    class GIN,GOUT gridNode;
    class L0U,L1U,L2U,L2D,L1D,L0D meshNode;
    class ENC,MU0,MU1,MD1,MD2,DEC opNode;
    class PROC procNode;
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `mesh_L0` | `(B, N_mesh_L0, d_h)` | Finest mesh activations produced by the encoder (g2m). |
| `mesh_L1` | `(B, N_mesh_L1, d_h)` | Intermediate latent features after the first up-step. |
| `mesh_L2` | `(B, N_mesh_L2, d_h)` | Coarsest representation iterated by the `process` block (m2m). |
| `grid_out` | `(B, N_grid, d_state)` | Final decoded residual after the full down-sweep and m2g step. |

```{note}
`HiLAMParallel` uses the same graph files but runs the up-sweep, process,
and down-sweep **in parallel** rather than sequentially. Architecture is identical;
execution order is the only difference.
```

---

## 06 · Extension Points

| What to add | Where to look | Key base class |
|-------------|---------------|----------------|
| Alternative file-backed datastore (e.g. NetCDF) | `datastore/base.py`, mirror hooks from `datastore/mdp.py` | `BaseDataStore` or `BaseRegularGridDatastore` |
| Custom `WeatherDataset` sampling strategy | `weather_dataset.py` (override slicing / normalisation) | `torch.utils.data.Dataset` |
| New graph encoder / decoder | `models/base_graph_model.py` (reuse graph buffers, clamping helpers) | `BaseGraphModel` |
| Additional hierarchical variant | `models/hi_lam.py`, `hi_lam_parallel.py` | `BaseHiGraphModel` + `InteractionNet` |
| New loss or metric | `metrics.py`, `loss_weighting.py` | `metrics.get_metric` helpers |
| New graph topology | [weather-model-graphs](https://github.com/mllam/weather-model-graphs) | — |

---

## 07 · File Map

| File | Description |
|------|-------------|
| `neural_lam/__init__.py` | Package marker, version metadata. |
| `neural_lam/config.py` | Typed Pydantic config objects (`NeuralLAMConfig`). |
| `neural_lam/create_graph.py` | CLI to build grid/mesh graphs from a datastore. |
| `neural_lam/custom_loggers.py` | W&B and MLflow Lightning logger adapters. |
| `neural_lam/interaction_net.py` | Edge-conditioned `InteractionNet` message-passing module. |
| `neural_lam/loss_weighting.py` | Per-variable and spatial loss-weight utilities. |
| `neural_lam/metrics.py` | Metric factory — MSE, MAE, wMSE, NLL, CRPS. |
| `neural_lam/plot_graph.py` | Visualise grid/mesh connectivity from the `graphs/` folder. |
| `neural_lam/train_model.py` | Lightning training entry point: wires configs, data, and models. |
| `neural_lam/utils.py` | Shared helpers: MLP builders, graph loading, rank-zero printing. |
| `neural_lam/vis.py` | Prediction and diagnostic visualisation utilities. |
| `neural_lam/weather_dataset.py` | `WeatherDataset` — bridges datastores and AR rollouts. |
| `neural_lam/datastore/base.py` | `BaseDataStore` and `BaseRegularGridDatastore` abstract interfaces. |
| `neural_lam/datastore/mdp.py` | `MDPDatastore` — mllam-data-prep zarr wrapper. |
| `neural_lam/datastore/plot_example.py` | Quick-look plotting utilities for datastore samples. |
| `neural_lam/datastore/npyfilesmeps/__init__.py` | Entry points for the MEPS numpy-file datastore. |
| `neural_lam/datastore/npyfilesmeps/config.py` | Dataclass schema for MEPS file layout. |
| `neural_lam/datastore/npyfilesmeps/store.py` | `NpyFilesDatastoreMEPS` with dask-backed loading. |
| `neural_lam/datastore/npyfilesmeps/compute_standardization_stats.py` | Precompute MEPS normalisation statistics. |
| `neural_lam/models/ar_model.py` | `ARModel` — base `LightningModule` for autoregressive training. |
| `neural_lam/models/base_graph_model.py` | Shared encode-process-decode scaffold and output clamping. |
| `neural_lam/models/base_hi_graph_model.py` | Base class utilities for hierarchical mesh processing. |
| `neural_lam/models/graph_lam.py` | Single-level `GraphLAM` implementation. |
| `neural_lam/models/hi_lam.py` | `HiLAM` — sequential multilevel message passing. |
| `neural_lam/models/hi_lam_parallel.py` | `HiLAMParallel` — parallelised hierarchical variant. |
