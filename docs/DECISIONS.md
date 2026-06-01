# Decisions

## Locked Decisions

1. Canonical compute substrate is `ndarray`.
2. Public APIs are pure numerical APIs over ndarray types.
3. `nabled` does not depend on Arrow types.
4. Workspace structure is required for long-term scale.
5. No hidden data conversion in hot compute paths.
6. Quality gates remain strict: pedantic linting, CI parity, and coverage >= 90%.
7. Backend selection is compile-time only; no runtime backend dispatch.
8. Default execution path is internal ndarray-native implementations.
9. Backend-specific behavior must not leak into public API names.
10. No legacy/backward-compatibility shims for unreleased APIs.
11. Decomposition-style APIs use concise domain naming (for example, `svd::decompose`).
12. Performance-critical kernels expose explicit allocation-control APIs (`*_into`) and optional reusable workspace types.
13. View/convenience APIs must not hide heap allocations without explicit rustdoc disclosure.
14. Dense-kernel tolerance and iteration defaults are centralized in one shared policy (`nabled-linalg::internal::DenseKernelPolicy`).
15. Delivery strategy is domain-first vertical slices: each domain is finalized with API, tests, benchmarks, and docs before expanding horizontally.
16. P2 kickoff is incremental: establish rank-3 tensor/cube APIs and compile-time accelerator contracts first, then add concrete GPU and future multi-node kernels.
17. Execution terminology is explicit and stable:
   - `Provider`: decomposition implementation source (for example, internal vs OpenBLAS-backed).
   - `Backend`: primitive-kernel execution target (for example, CPU, GPU).
   - `Kernel`: operation-family contract implemented by backends (for example, matrix-matrix multiply).
18. Provider and backend are orthogonal axes and may both be used within one public algorithm flow.
19. Kernel implementations do not directly invoke provider selection; orchestration of provider-backed decomposition and backend-backed kernels lives in domain APIs.
20. Provider selection remains compile-time via feature gating (`#[cfg]`) in domain code; no runtime provider-dispatch API is required.
21. Kernel-family scope is explicitly cataloged (`docs/KERNEL_CATALOG.md`) and treated as finite/planned work, not ad hoc expansion.
22. V1 tensor and GPU capability scope is explicitly bounded and version-locked in `docs/V1_STABILITY.md`; out-of-scope behavior must be explicit/typed, not implicit fallback.
23. Runtime workload-size policy is allowed inside a selected backend implementation (for example, `GpuBackend` deciding not to attempt GPU for tiny workloads); this does not violate compile-time backend selection.
24. `D-MOD-1` Facade default features: the `nabled` facade defaults to `["linalg"]` only. Physical AI / ML / signal / sim / geometry / model / kinematics / dynamics / control / sensor are per-domain opt-in features, and `physical-ai` is the umbrella feature that re-enables the full Physical AI vertical. This keeps `nabled = "0.x"` slim for users who only need linear algebra and avoids transitive ndarray-linalg / pyo3 / numpy build costs.
25. `D-MOD-2` URDF / DH boundary lockdown: URDF-loaded `RobotModel`s never silently synthesize DH parameters. `BodySpec::dh_params` is `Option<DhParams<T>>`. URDF parsing leaves it as `None`. `model::dh::to_chain_spec` / `extract_chain_spec` / `extract_chain_spec_for_dynamics` fail loudly with `ModelError::InvalidInput` when any body on the requested branch is missing DH parameters. URDF-derived models can still be used through the kinematic tree FK / Jacobian / IK and tree dynamics APIs.
26. `D-MOD-3` Tree dynamics (lite): `rnea_tree` / `mass_matrix_tree` / `forward_dynamics_tree` (with `_into` siblings) operate per-branch by extracting a serial sub-`ChainSpec` via `extract_chain_spec_for_dynamics`, running the existing serial RNEA / CRBA / FD on a constructed sub-`RobotModel`, and scattering results back into the full-model actuated ordering. Whole-tree coupled dynamics is intentionally out of scope; callers compose branch calls.
27. `D-MOD-4` Python physical_ai ingress is view-first: every `physical_ai` Python entry point that has a Rust `_view` variant calls it directly via `PyReadonlyArrayN::as_array()` (no `.to_owned()` on the FFI boundary). Egress paths that have `_into` siblings expose an `out=` numpy buffer keyword that is written in place. This is the documented "view-first" policy for `PYNABLED_ARCHITECTURE`.
28. `D-EMB-1` Embeddings crate is bring-your-own-vectors, numerics-only: `nabled-embeddings` accepts dense `Array2<T>` / `ArrayView2<T>` (`f32`/`f64` via `NabledReal`) and composes existing `nabled-linalg::vector` + `nabled-ml::pca` kernels. It does not implement model inference, tokenizers, ONNX/candle, an ANN index (HNSW/IVF), or storage. ANN recall and persistence are a vector store's job (e.g. LanceDB); the crate provides the exact rerank/normalize/compress/kNN compute step that runs after ANN. The crate stays Arrow-free; Arrow ingress lives only at the `nabled` facade and `pynabled`. LanceDB is an optional, I/O-free plug-in / batch producer, never a crate-graph dependency (no `lance`/`lancedb`/`tokio`/async).
29. `D-EMB-2` Metric is an explicit caller choice: `query_corpus_scores`/`rerank`/`brute_force_knn` take an explicit `Metric { Cosine, Dot, L2 }`, and each metric records its ranking polarity (`higher_is_better`) so direction-aware `top_k` selection is correct for both similarities (cosine/dot, higher is better) and distances (L2, lower is better). The math cannot enforce two usage rules, so they are documented contract: (1) query and corpus must come from the same model and share the same `dim`; (2) the user picks the metric to match how the model was trained (cosine default; dot for MIPS-style models where un-normalized dot favors larger-norm rows by design; L2 where applicable, which matches normalized-cosine ranking).
30. `D-EMB-3` Embeddings int8 quantization scheme: `nabled-embeddings::quantize` uses **per-row symmetric int8** quantization. For each row, `scale = max(|x|) / 127`, each element is `round(x / scale)` clamped to `[-127, 127]` (the `-128` code is intentionally unused so `dequantize(quantize(x))` has no sign bias), and a fully-zero row gets `scale = 0` with all-zero codes (decoding back to zeros). `QuantizedMatrix` stores `data: Array2<i8>` plus `scales: Array1<f32>`. Scoring is **dequantize-then-existing-kernel**: `query_corpus_scores_quantized` decodes both operands to `f32` and reuses `query_corpus_scores`. There is deliberately no native int8 kernel and no new `nabled-linalg` surface; quantization is a storage/transfer optimization (~4x smaller) traded against a small, bounded precision loss, not a faster compute path.
31. `D-EMB-4` Arrow embeddings neighbor result contract: the `nabled::arrow` embeddings adapters (features `embeddings` + `arrow`) return Arrow-native results with a locked layout. Score matrices and normalized rows are `FixedSizeListArray` (one inner list per query/row). Single-query rerank neighbors are a `StructArray` with a non-nullable `int64` `index` field and a metric-typed (`Float32`/`Float64`) `score` field, in best-first order. Per-query brute-force kNN neighbors are a `ListArray` whose values use that same neighbor `StructArray` layout, one list element per query row. Inbound Arrow → ndarray bridging is zero-copy via `fixed_size_list_view`; neighbor indices that exceed the Arrow `int64` range are a typed error, not a silent truncation.

## API Purity Model

1. A function accepts ndarray inputs and returns ndarray outputs (or scalar/error outputs).
2. Any additional controls are explicit function arguments.
3. No assumptions about calling context (database, Arrow, SQL, transport layer).

## Data Types

1. Vector: `Array1<T>`, `ArrayView1<'a, T>`, `ArrayViewMut1<'a, T>`.
2. Matrix: `Array2<T>`, `ArrayView2<'a, T>`, `ArrayViewMut2<'a, T>`.
3. Higher-rank tensors (future): `ArrayD<T>` and fixed-rank aliases as needed.
4. Complex support: `num_complex::Complex32/Complex64` where algorithms are mathematically valid.

## Near-Term Non-Goals

1. Cross-library interop adapters.
2. Arrow integration inside `nabled`.

These are deferred until the ndarray-first core is complete and stable.

## Python Bindings

Python bindings are pursued via the `pynabled` crate (PyO3-based). NumPy arrays are the canonical
dense Python data type. Branch-local parity and PyPI readiness are not inferred from this document;
use `docs/PYNABLED_GAPS_AUDIT.md`, `docs/EXECUTION_TRACKER.md`, and `docs/STATUS.md` for current
merge/release truth when working on `feat/pynabled-bindings`.

Locked Python-boundary decisions:
1. `pynabled` does not use one universal Python carrier across all admitted domains.
2. Dense/vector/matrix/tensor CPU-facing APIs use NumPy arrays as the canonical Python carrier.
3. Sparse APIs require first-class sparse carriers (`pynabled` wrappers and/or SciPy-compatible
   objects); raw CSR buffer tuples are transitional only, not the release-grade contract.
4. Arrow-admitted APIs use canonical PyArrow/`ndarrow` carriers; Arrow-native Rust flows must not
   be degraded back to NumPy by default.
5. Rich Rust result structs must become typed Python result objects where metadata matters; tuples
   are not the final release contract for those families.
6. Callback-driven Python APIs are convenience APIs unless their hot loops remain in Rust; they may
   not be documented as no-compromise performance equivalents by default.
7. Copy/allocation behavior at the Python boundary must follow the same no-hidden-copy discipline as
   the Rust workspace.

For the detailed per-domain contract, use `docs/PYNABLED_ARCHITECTURE.md`.

## Provider and Backend Contract

1. `blas` is a baseline feature for enabling BLAS-accelerated ndarray paths where available.
2. LAPACK acceleration is provider-driven, not a separate runtime backend layer.
3. Provider scope includes four LAPACK provider features: `openblas-system`, `openblas-static`, `netlib-system`, and `netlib-static`.
4. Provider features imply `blas` so users do not have to compose low-level flags manually.
5. LAPACK-accelerated code should be gated by feature selection, not by hardcoded `target_os` branching.
6. Current platform intent is macOS and Linux first; Windows support is deferred.
7. Backend acceleration is compile-time feature-gated and operation-specific.
8. Backend acceleration does not imply provider acceleration, and provider acceleration does not imply backend acceleration.
