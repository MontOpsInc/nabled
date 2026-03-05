# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Bug Fixes

- Patches incorrect feature flag usage ([b33124f](https://github.com/MontOpsInc/nabled/commit/b33124fdf3bb852f79c45893b1948eac4ef19ba1))

### Documentation

- Updates readme ([1bc0ae1](https://github.com/MontOpsInc/nabled/commit/1bc0ae12a71ad63d7012d0720fca17d4c6735a63))

### Features

- Updates the public API to support f32/f64, introduced CsrMatrixView ([9e7b04c](https://github.com/MontOpsInc/nabled/commit/9e7b04c6d8bb706c5034f291a537c08391646028))

### Build

- Updates just prepare-release command ([6af9b46](https://github.com/MontOpsInc/nabled/commit/6af9b464a4910521b07ea853ce4896e209d776e0))

## [0.0.3] - 2026-03-04

### Documentation

- Updates root README ([#15](https://github.com/MontOpsInc/nabled/issues/15)) ([298fd11](https://github.com/MontOpsInc/nabled/commit/298fd11c5b35f0d58f2615e8794ed887eb4ef98c))

### Features

- Introduces remaining lapack providers ([8e40881](https://github.com/MontOpsInc/nabled/commit/8e40881142d08f9ebb781a77c3d45a7f507f2fe7))

### Miscellaneous Tasks

- Release v0.0.3 ([#16](https://github.com/MontOpsInc/nabled/issues/16)) ([f3b59ff](https://github.com/MontOpsInc/nabled/commit/f3b59ff72627c2bad29cc1ba74c9e28b281aa602))

### Refactor

- Renames cuda to gpu to be more accurate, cleans up some docs ([0b277f9](https://github.com/MontOpsInc/nabled/commit/0b277f9ea97fd928831f2d6f423fcbd40595a273))

### Testing

- Introduces cpu/gpu test basic ([7539eaa](https://github.com/MontOpsInc/nabled/commit/7539eaa669252de9f4c681df8607a86179b13115))

## [0.0.2] - 2026-03-03

### Miscellaneous Tasks

- Prepare release v0.0.2 ([#14](https://github.com/MontOpsInc/nabled/issues/14)) ([e80300c](https://github.com/MontOpsInc/nabled/commit/e80300ce565dd9a0744b518275a06e5e41dbd04e))

## [0.0.1] - 2026-03-03

### Bug Fixes

- Adds additional tests, complex parity ([85d79b8](https://github.com/MontOpsInc/nabled/commit/85d79b829731b70736f29cef93f1e815dd25b1dc))
- Expands factorization reuse sparse solver APIs ([6c1e3f1](https://github.com/MontOpsInc/nabled/commit/6c1e3f1e50d4918349e3473812be2613dae47c34))
- Addresses remaining gap to base release supported functionality ([4cedff8](https://github.com/MontOpsInc/nabled/commit/4cedff87f533c59ea08e8773b0fb290cab05b312))
- Improves benches, focuses on cholesky first ([f9e266e](https://github.com/MontOpsInc/nabled/commit/f9e266e967d576d2453f0f00491291eaf4db86eb))
- Addresses some refactors needed with CpuBackend and other areas ([3d02ec2](https://github.com/MontOpsInc/nabled/commit/3d02ec22b73f57902aaa022f9407eec8904f232c))

### Documentation

- Adds badges ([0f2a10a](https://github.com/MontOpsInc/nabled/commit/0f2a10aea88cfa8301e6c98745bbc4c076754636))
- Updates docs ([24630cb](https://github.com/MontOpsInc/nabled/commit/24630cbc09c7a9d747acd884181f3ca0df8c3594))
- Housekeeping ([2ffb441](https://github.com/MontOpsInc/nabled/commit/2ffb44161fe2b489829adf13d491c8b636308757))
- Updates docs and release information ([9791a7c](https://github.com/MontOpsInc/nabled/commit/9791a7c8608ce06e631c126acee794a24a2fe160))

### Features

- Introduces backend kernel traits, some housekeeping ([2eb0bcb](https://github.com/MontOpsInc/nabled/commit/2eb0bcb08f1f6f8fb0ed2c1976414061f24464ca))
- Introduces LAPACK for linux, updates ci ([0e37da4](https://github.com/MontOpsInc/nabled/commit/0e37da4ba6414acb16e7dfde8a36f3567ceb6093))
- Eigen leverages backend kernels ([e38fc1d](https://github.com/MontOpsInc/nabled/commit/e38fc1d79dab4781c1b7ec03c99f5c1af0763b15))
- Triangular, cholesky, and schur pulled into benches and traits ([59ce7d1](https://github.com/MontOpsInc/nabled/commit/59ce7d161c493c68a4d68b2d870cb0f9c472329c))
- Pca, regression, and sylvester pulled into benches and traits ([3d9ed15](https://github.com/MontOpsInc/nabled/commit/3d9ed1569be85496329104f1b1cf9da00fff28aa))
- Matrix functions pulled into benches and traits ([b5f0363](https://github.com/MontOpsInc/nabled/commit/b5f036395a0c148ac6fce652637f3d6cca893f41))
- Adds additional primitives, addresses semantics ([3bd6765](https://github.com/MontOpsInc/nabled/commit/3bd6765767b836f53cd15425bb75fa93bb829e64))
- Adds ndarray view support ([6d5cbb2](https://github.com/MontOpsInc/nabled/commit/6d5cbb2adc33c9bcc72515b8e6fea3611258466c))
- Builds out complex parity further esp around provider backed methods ([654b049](https://github.com/MontOpsInc/nabled/commit/654b0499d76cdc4148b077a0ad7f0d24a4e0e82f))
- Adds provider backed support for schur and sylvester ([c9be857](https://github.com/MontOpsInc/nabled/commit/c9be857eff35303637c917a18fcc8afd1ab39b6d))
- Adds provider backed support for matrix functions ([d301186](https://github.com/MontOpsInc/nabled/commit/d3011867d711ddc1f507d7a327d3602949b7d462))
- Completes complex provider backed support for matrix functions ([57c69ad](https://github.com/MontOpsInc/nabled/commit/57c69ad5a199f2b41d73c9a97d92ddff44f3cad2))
- Completes complex paths w/ no provider for lu, cholesky, schur, sylvester/lyapunov, polar, and svd ([68634b1](https://github.com/MontOpsInc/nabled/commit/68634b199a6127bb52a60c38cdc6fc106699dd85))
- Introduces DenseKernelPolicies and centralizes convergence behaviors ([69246bc](https://github.com/MontOpsInc/nabled/commit/69246bc3d0378b26f718bc4ebf7ba7dc34d7fff5))
- Pushes through P0 and P1 items in capability matrix, sets foundation for P2 ([b86020d](https://github.com/MontOpsInc/nabled/commit/b86020d40eac248235f57ab6e101a8d0b613ebe3))
- Introduces more tensor/cube primitives, work on sparse, introduces accelerator-rayon ([879a6d5](https://github.com/MontOpsInc/nabled/commit/879a6d522763e416393a7c643fe840ec9190c8a2))
- Introduces sparse ILU(0) support, additional benches, and more ([074d823](https://github.com/MontOpsInc/nabled/commit/074d823f89cebe420cbf3eb432f2db2f41d5bdf1))
- Expands ILUT config/policy layer, additional gmres ilust solvers, tests, and benches ([1a1383c](https://github.com/MontOpsInc/nabled/commit/1a1383c6c35028df01be5d6ae88c85440165b090))
- Expands complex matrix, cs apis, cs/gmres primitives and some tensor/cube kernels ([18fcb18](https://github.com/MontOpsInc/nabled/commit/18fcb184b7eee2d8840eb927cdb7dd34ca752725))
- Sparse + tensor depth ([668465d](https://github.com/MontOpsInc/nabled/commit/668465d45e1965ad4a8e2f860b73427a715c153c))
- Adds distributed accelerator ([09b8c00](https://github.com/MontOpsInc/nabled/commit/09b8c0059105e48364c7fd1a997289c5c3146122))
- Introduces new sparse ILUK workflows and solvers ([8cc550b](https://github.com/MontOpsInc/nabled/commit/8cc550b439b5816fd75e53bcd4077ec3879a0973))
- Introduces batched decomposition surface, updates across key domains, coverage ci updated ([c4974c7](https://github.com/MontOpsInc/nabled/commit/c4974c708bf9416eba2c1259f7632e96fc64306f))
- Introduces additional tensor/cube capabilities ([0fd5028](https://github.com/MontOpsInc/nabled/commit/0fd502874a65e5af8e3fcd471bda06c1037e7092))

### Miscellaneous Tasks

- Checkpoint local changes after nabled repo migration ([8088cbb](https://github.com/MontOpsInc/nabled/commit/8088cbb5398c20138cffeae4f786befcf3d929fa))
- Addresses lints, upgrades deps, configures repo ([b2b9490](https://github.com/MontOpsInc/nabled/commit/b2b9490623b1fb114259ff8be15658215ece12cf))
- Adds test-utils feature as placeholder ([ddbed14](https://github.com/MontOpsInc/nabled/commit/ddbed140d4cbc14a8cf007884540752e5d6059b3))
- Removes docker login from ci, adds codecov token ([bfd5b48](https://github.com/MontOpsInc/nabled/commit/bfd5b48a1ee86c522f82831dfc3706fc57c49ca3))
- Introduces proper benchmarks and benchmark reporting ([48c9dfa](https://github.com/MontOpsInc/nabled/commit/48c9dfa52f4ba53991c47480b32cd1be3f47c5ba))
- Adds regression checks for performance ([ed44c1d](https://github.com/MontOpsInc/nabled/commit/ed44c1dc73bed6d4df462f3aef78300ef8883cae))
- Addresses ci failures ([1e1a6a5](https://github.com/MontOpsInc/nabled/commit/1e1a6a53c66833b27c1190510619e8c334a77a46))
- Fine-tuning the benchmark gates ([7369577](https://github.com/MontOpsInc/nabled/commit/736957752ee28146e63adb69f6e709c8b1218158))
- Adds benchmark publish ([2f10300](https://github.com/MontOpsInc/nabled/commit/2f10300308fe169bc3740e9159bb0ef95812589d))
- Patches path for lcov in ci ([41dfd86](https://github.com/MontOpsInc/nabled/commit/41dfd86d64a6a8b0b5ea4c6045ae446d07deeb62))

### Refactor

- Standardizes on ndarray only, moves to workspace ([8fd329b](https://github.com/MontOpsInc/nabled/commit/8fd329be15c085ca0966301fc3462f87a7519615))
- Fixes feature flags, organizes structure for backend support, housekeeping ([d9b7424](https://github.com/MontOpsInc/nabled/commit/d9b7424d10ff8ee95fe4cde35ca143aef7f8e08e))
- Refactors all methods to use new feature format, addresses ci ([35ae961](https://github.com/MontOpsInc/nabled/commit/35ae9613d71a2de1e2e43bbed4a740b8dc9a4d34))
- Flattens module structure, simplifies ([4f1f717](https://github.com/MontOpsInc/nabled/commit/4f1f717f379dcaebd04ff995c8ffe67d42113945))
- Continued view support, additional sparse expansion, new benches ([a59c7b5](https://github.com/MontOpsInc/nabled/commit/a59c7b5dd1b6a91c3108171eeec4147ce8f581a3))
- Builds out kernel catalog ([0d85cd1](https://github.com/MontOpsInc/nabled/commit/0d85cd10f575e1bde404cc8bcf5e05c5e67f0e9d))
- Finalizes first pass of kernel traits ([66e6099](https://github.com/MontOpsInc/nabled/commit/66e60992c4e34d5a11d6b7327b449bc58cfa763e))
- Removes distributed backend for now, cleans up cuda and related features ([e13563a](https://github.com/MontOpsInc/nabled/commit/e13563a3768d28fac1ef11fbfc303226935c0769))

### Testing

- Udpates ci to proper integration test ([70dfb52](https://github.com/MontOpsInc/nabled/commit/70dfb523ccf1bf05c6411b043549baa0c8305430))
- 85% line coverage ([b76ceb1](https://github.com/MontOpsInc/nabled/commit/b76ceb162614d8386746beee8b0ce03106d2fc3e))
- Reorganizes tests into unit tests ([95ae77e](https://github.com/MontOpsInc/nabled/commit/95ae77e73480d60d6b86e88a456bab545e63a03b))
- Raises coverage > 90% ([4b91866](https://github.com/MontOpsInc/nabled/commit/4b91866b5f2525bfa9262f49bf71cea25a4be541))

### Ai

- Introduces agents.md ([9d0474d](https://github.com/MontOpsInc/nabled/commit/9d0474d40c40983157170c0f626ac00d8f51d774))
- Checkpointing progress ([60691b7](https://github.com/MontOpsInc/nabled/commit/60691b7d322d7a1493b2a7fff5ec770ab40a9a97))

### Bench

- Adds LAPACK ([ccf5801](https://github.com/MontOpsInc/nabled/commit/ccf5801e0177001f05970b10f8dea2eebb389d7f))

### Build

- Prepares release workflow ([e040175](https://github.com/MontOpsInc/nabled/commit/e04017583c357a699007fa14e26dcfc933f97a18))

### Plan

- Introduces production readiness document ([13ba0da](https://github.com/MontOpsInc/nabled/commit/13ba0da5ebd0f0f770b7f9f329756861f14b0a16))


