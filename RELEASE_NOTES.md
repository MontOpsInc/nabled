## [0.0.5] - 2026-03-09

### Bug Fixes

- Expands gpu support for f32 ([adda852](https://github.com/MontOpsInc/nabled/commit/adda852b10ef214e095f2c4e7e508bff71867ec6))
- Finalizes GPU backend support across all supported kernels ([47f726b](https://github.com/MontOpsInc/nabled/commit/47f726bc38796929e1f334a028b22f81dc825b5a))
- Addresses complex magma kernels for lu, cholesky, qr, svd, and non-symmetirc eigen ([4b34cd7](https://github.com/MontOpsInc/nabled/commit/4b34cd71e55066716e49eb7654070fc36e3f172f))
- Batched MAGMA decomposition support (LU/Cholesky/QR) ([f3d0b6f](https://github.com/MontOpsInc/nabled/commit/f3d0b6f3c61ecaf15360d314731084cabf616b3a))
- Implements MAGMA sparse provider phase 1, more ([c559fa3](https://github.com/MontOpsInc/nabled/commit/c559fa319089855a91c3a3fd24d6ac523c91d140))
- Implements phase-2 of MAGMA sparse iterative/preconditional solves, complex tensor dispatch ([2226699](https://github.com/MontOpsInc/nabled/commit/2226699c6b2e23683a62efde96c4c33c7639b975))
- Adds fallback paths when MAGMA fails, adjusts scripts tmux ([50b8bfa](https://github.com/MontOpsInc/nabled/commit/50b8bfa0fb312c2757dd26991a2edf798420be4e))
- MAGMA validated against single and multi-threaded across existing api surface ([c98dbf7](https://github.com/MontOpsInc/nabled/commit/c98dbf7006cee7269630e896a061c9b9775dbb33))
- Introduces additional magma decomposition gate based on matrix size ([68deca2](https://github.com/MontOpsInc/nabled/commit/68deca2155daf8a216e12c93251bb0756b22ac88))
- Building out additional features across tensor-network ergonomics, additional decompositions ([a697aff](https://github.com/MontOpsInc/nabled/commit/a697aff343851071cf036adae8575041a8c64206))
- Builds out further TT tensor apis ([e5b4fe4](https://github.com/MontOpsInc/nabled/commit/e5b4fe41532f9d3a5543dcc0477d996168e02fa4))
- Completes tensor v1 rubric ([24106bb](https://github.com/MontOpsInc/nabled/commit/24106bb91f3f5258fead9e73cf6d124300076bb4))
- Patches some routing behavior around workload size ([ff6e696](https://github.com/MontOpsInc/nabled/commit/ff6e696574efe4b575433d583617675af9cc720f))

### Documentation

- Updates trackers around MAGMA ([d865f1f](https://github.com/MontOpsInc/nabled/commit/d865f1f6ffd0668ead62893012d93df7c9e9165b))
- Updates tracker docs ([a19258b](https://github.com/MontOpsInc/nabled/commit/a19258b472a63841723129af7f4204955950fb1a))
- Updates docs and READMEs ([43425ae](https://github.com/MontOpsInc/nabled/commit/43425ae5314e86b285374584ed98b3e6c01e4a27))

### Features

- Introduces f64 support for GPU ([8d99ead](https://github.com/MontOpsInc/nabled/commit/8d99eadc480ef32e2f2da774dd00f7da9b30283f))
- Introduces minimal NVIDIA MAGMA support ([c4819a0](https://github.com/MontOpsInc/nabled/commit/c4819a0108f1d3a9a47888c23b047f700b478ee7))

### Miscellaneous Tasks

- Cleans up gpu remote execution scripts and dockerfile ([a139eb5](https://github.com/MontOpsInc/nabled/commit/a139eb511606805c43ed2108b5f4c4b4ebdda799))
- Addresses bench in ci ([2531f26](https://github.com/MontOpsInc/nabled/commit/2531f26a20c231b080d326a1e7a4fac1b45dade8))
- Stabilizing benches ([f566754](https://github.com/MontOpsInc/nabled/commit/f566754f860bdd96c180fd53279c9c9987bc8aae))

### Testing

- Introduces MAGMA strict checks to validate MAGMA usage when available ([aebb09e](https://github.com/MontOpsInc/nabled/commit/aebb09ec696863b18f40f41a5bc29c235426f9e9))

### Build

- Includes scripts for remote gpu setup and verification ([a8eeda7](https://github.com/MontOpsInc/nabled/commit/a8eeda7b000af2950cabea2ac4ad18e6e6a988f3))


