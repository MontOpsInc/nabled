# Pynabled Architecture Contract

Last updated: 2026-04-06

## Purpose

This document defines the boundary architecture for `pynabled`.

It exists because Python parity is not only a question of function coverage. `pynabled` must also
preserve `nabled`'s admitted carrier contracts, copy/allocation discipline, result fidelity, and
performance expectations.

Use this document together with:

1. `docs/PYNABLED_PARITY_MATRIX.md` for scope and parity status.
2. `docs/PYNABLED_GAPS_AUDIT.md` for the original branch audit findings.
3. `docs/EXECUTION_TRACKER.md` for ordered execution state.

## Repository Requirements Inherited By Python

`pynabled` inherits the same hard requirements that govern the Rust workspace:

1. Canonical compute substrate remains `ndarray`.
2. No hidden copy-heavy conversions in hot paths.
3. View-first execution is preferred where possible.
4. Allocation-control semantics matter on performance-critical workflows.
5. Carrier-specific interop belongs at explicit boundaries, not inside core numerical domains.
6. Release claims must match actual behavior; convenience wrappers may not be presented as
   no-compromise equivalents when they narrow data contracts or force avoidable copies.

## Architecture Questions This Document Answers

Before continuing Python breadth work, four questions must be explicit:

1. What is the canonical Python carrier for each domain?
2. What copy/allocation contract does each domain expose?
3. Which APIs preserve Rust-side execution locality, and which cross back into Python?
4. Which Rust result structs must remain structured objects in Python instead of flattened tuples?

## Locked Decisions

### 1. No Universal Python Carrier

`pynabled` does not use one Python data type for all admitted `nabled` domains.

The correct design is domain-native carrier selection, not "NumPy everywhere".

### 2. Canonical Python Carriers By Domain

#### Dense host arrays: vector / matrix / tensor

Canonical carrier:
1. `numpy.ndarray`

Why:
1. It is the standard homogeneous N-dimensional array type in the Python scientific ecosystem.
2. It interoperates naturally with SciPy, scikit-learn, JAX/PyTorch conversion points, and common
   Python numerical workflows.
3. The maintained Rust bridge (`rust-numpy`) maps NumPy arrays cleanly onto `ndarray` views and
   owned arrays.

Non-goal:
1. Introducing a custom dense Python array type as the primary public contract.

#### Sparse structures

Canonical carrier:
1. First-class `pynabled` sparse objects and/or SciPy-compatible sparse matrices, depending on the
   domain surface.

Why:
1. Sparse matrices have structural semantics that raw dense arrays do not capture.
2. Raw `(nrows, ncols, indptr, indices, data)` tuples are not a release-grade user contract.
3. Users expect sparse interop with SciPy-capable libraries.

Current status:
1. Raw CSR-buffer entrypoints are transitional only and do not satisfy the release contract.

#### Arrow-admitted domains

Canonical carrier:
1. `pyarrow` carriers aligned to `ndarrow` / `nabled::arrow` contracts.

Why:
1. Arrow-native workflows must stay Arrow-native when the Rust facade already admits Arrow-native
   carriers.
2. Converting Arrow workflows back to NumPy narrows the interop contract and breaks carrier
   fidelity.

#### Rich decomposition/workflow results

Canonical carrier:
1. Typed Python result objects.

Why:
1. Rust returns meaningful structs, not anonymous tuples.
2. Stable metadata, named fields, and future extension points require structured results.

Examples:
1. decomposition outputs,
2. PCA results,
3. regression results,
4. CP diagnostics/reporting,
5. Tucker/TT result families,
6. reusable factorization/workspace objects.

#### Reusable plans, workspaces, and factorization reuse paths

Canonical carrier:
1. Typed Python objects, not bare arrays.

Why:
1. These are lifecycle-bearing resources, not just data blobs.
2. Python needs stable object identity for repeated solves / repeated execution plans.

### 3. Interop Carriers Are Allowed, But Canonical Carriers Stay Explicit

`pynabled` should accept ecosystem-friendly interop forms when the cost model is explicit, but
interop carriers do not replace the canonical carrier for a domain.

Examples:
1. Dense APIs may accept array-like Python inputs that normalize into NumPy semantics.
2. Sparse APIs may accept SciPy objects and/or `pynabled` sparse wrappers.
3. Arrow APIs may accept PyArrow arrays/tables/record batches only where that matches admitted
   `ndarrow` carriers.

## Copy And Allocation Contract

### Contract Classes

Python boundary behavior must fall into one of four explicit classes:

1. Borrowed zero-copy ingress:
   - Python memory is borrowed directly into an `ndarray` view with no wrapper-level copy.
2. Explicit normalization copy:
   - Copy/cast/layout normalization happens because the user chose an input that is not directly
     admissible.
   - This must be explicit in API behavior or docs; it may not be silently framed as no-cost.
3. Result allocation by API contract:
   - The algorithm produces a new array/result object.
   - This is normal and not a performance bug by itself.
4. Reusable/in-place execution:
   - Performance-sensitive paths should expose `out=` / reusable object / workspace equivalents
     where Rust already exposes material allocation-control benefits.

### Dense Contract

1. Dense/vector/matrix/tensor ingress should target borrowed NumPy-backed views first.
2. Blanket "C-contiguous only" rejection is not the release-grade final contract for all dense
   APIs.
3. If a layout restriction exists for a given kernel family, one of the following must be true:
   - the Rust path truly requires it and it is documented explicitly, or
   - the binding offers an explicit normalization path whose copy behavior is visible to the user.
4. Hidden `np.ascontiguousarray(...)`-style normalization inside hot wrappers is not acceptable.
5. Where Rust already has `*_into` or reusable workspace semantics, Python must eventually expose
   equivalent allocation-control behavior if the path is performance-relevant.

### Sparse Contract

1. Sparse carriers must preserve sparse structure as first-class data.
2. Copying CSR buffers into owned Rust vectors on every call is not a release-grade final design.
3. Sparse APIs must distinguish:
   - borrowed/adapter paths,
   - explicit normalization paths,
   - and factorization/workspace reuse paths.

### Arrow Contract

1. Arrow/PyArrow ingress and egress must stay aligned to admitted `ndarrow` carriers.
2. If a Rust Arrow path is Arrow-native, the Python surface should stay Arrow-native rather than
   degrading to NumPy.
3. Zero-copy claims for Arrow interop must be made only where the carrier contract actually
   supports it.

## Execution Locality Contract

Not every Python API has the same execution semantics.

### Performance-preserving APIs

These are APIs where:
1. the Python call crosses into Rust,
2. the hot loop stays in Rust,
3. and the carrier boundary is crossed once per call or once per result object.

This is the model required for "preserve `nabled` performance" claims.

### Convenience callback APIs

Python-callable-driven APIs are still useful, but they must be classified correctly.

Current examples:
1. callback-driven Jacobian helpers,
2. callback-driven optimization routines.

Contract:
1. These APIs are convenience-oriented unless and until a Rust-resident objective/gradient story
   exists.
2. They may not be presented as no-compromise performance equivalents to Rust-native execution.
3. Docs must distinguish them from performance-preserving APIs.
4. Python should expose typed config objects for these workflows instead of treating raw positional
   tuning parameters as the production-facing contract.
5. Per-invocation carrier materialization at the Python callback boundary is acceptable here when
   required for Python-owned object lifetimes; it must be documented explicitly rather than
   described as zero-copy or hot-path-preserving behavior.

## Result Fidelity Contract

Flattening rich Rust structs into tuples/arrays is not release-grade where metadata matters.

### Required Direction

1. Result families with meaningful named fields must become typed Python result objects.
2. Stable attribute names should mirror Rust struct semantics closely.
3. Reconstruct/diagnostics/report workflows should operate on these result objects directly where
   practical.

### Examples Of Rows Requiring Structured Results

1. SVD / QR / LU / Eigen / Schur / Polar decomposition outputs
2. PCA and regression outputs
3. CP diagnostics and convergence reports
4. HOSVD / Tucker / TT decomposition families
5. Sparse factorization and reusable solver/preconditioner handles

## Recommended Implementation Model

### Dense/vector/matrix/tensor CPU-facing bindings

1. Keep `numpy.ndarray` as the canonical Python carrier.
2. Use `rust-numpy` as the bridge into `ndarray`.
3. Prefer borrowed-array ingress over wrapper-level owned conversion.
4. Add explicit allocation-control semantics where Rust hot paths justify them.

### Sparse bindings

1. Replace raw buffer tuples as the release contract.
2. Introduce first-class sparse carrier objects and SciPy interop.
3. Add reusable factorization/preconditioner objects instead of repeated setup-per-call APIs only.

### Arrow bindings

1. Rebuild around canonical PyArrow/`ndarrow` carriers.
2. Keep Arrow-native outputs where the admitted Rust surface is Arrow-native.
3. Avoid NumPy fallback surfaces that narrow the Arrow contract.

### Result objects

1. Move from tuple-heavy returns to typed Python result classes.
2. Use these result classes as the stable API contract for higher-level workflows and reuse paths.

## Immediate Consequences For The Tracker

These decisions change execution order:

1. Python parity implementation may continue, but only under the locked carrier/copy/result
   contracts above.
2. Foundation fixes now outrank feature-count expansion when the old implementation model violates
   the contract.
3. Remaining work should be sequenced around:
   - dense boundary contract hardening,
   - sparse carrier redesign,
   - result-object fidelity,
   - Arrow/PyArrow contract repair,
   - and only then broader convenience expansion where needed.

## Definition Of Done For This Document

This document is current only if:

1. canonical carrier by domain is explicit,
2. copy/allocation classes are explicit,
3. callback-vs-performance execution semantics are explicit,
4. result-object fidelity requirements are explicit,
5. and `docs/EXECUTION_TRACKER.md` uses this contract to drive the next implementation steps.
