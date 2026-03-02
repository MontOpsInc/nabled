//! Compile-time backend contracts and accelerator kernels.

pub mod backends;
pub mod cpu;
pub mod dispatch;
pub mod distributed;
pub mod gpu;
pub mod kernels;

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};

    use crate::accelerator::backends::{
        AcceleratorError, BackendKind, CpuBackend, CudaBackend, DistributedBackend, execute,
    };
    use crate::accelerator::cpu::{for_each_row_chunk, matmat_accelerated, matmat_serial};
    #[cfg(feature = "accelerator-wgpu")]
    use crate::accelerator::dispatch::matmat_with_backend_f32;
    use crate::accelerator::dispatch::{
        batched_matmat_with_backend, batched_row_matvec_with_backend, dot_with_backend,
        matmat_with_backend, matvec_with_backend, pairwise_cosine_with_backend,
        pairwise_l2_with_backend, sparse_matmat_dense_with_backend,
        sparse_matmat_sparse_with_backend, sparse_matvec_with_backend,
        tensor_batched_matmul_last_two_with_backend, tensor_contract_axes_with_backend,
        tensor_sum_last_axis_with_backend, triangular_solve_mat_with_backend,
        triangular_solve_vec_with_backend,
    };
    use crate::accelerator::distributed::{
        DistributedConfig, DistributedSchedule, matmat_distributed, matmat_distributed_tiled,
    };
    #[cfg(not(feature = "accelerator-wgpu"))]
    use crate::accelerator::gpu::matmat_gpu_f32;
    use crate::sparse::CsrMatrix;

    #[test]
    fn cpu_backend_executes_operation() {
        let value = execute::<CpuBackend, _, _>(|| 2 + 3).unwrap();
        assert_eq!(value, 5);
    }

    #[test]
    fn distributed_backend_executes_operation() {
        let value = execute::<DistributedBackend, _, _>(|| 7 * 6).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn cuda_backend_returns_error() {
        let cuda = execute::<CudaBackend, _, _>(|| 1);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn chunking_visits_all_rows() {
        let matrix =
            Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                .unwrap();

        let mut seen_rows = 0_usize;
        for_each_row_chunk(&matrix, 2, |chunk| {
            seen_rows += chunk.nrows();
        })
        .unwrap();
        assert_eq!(seen_rows, matrix.nrows());
    }

    #[test]
    fn chunking_rejects_invalid_chunk_size() {
        let matrix = Array2::<f64>::zeros((2, 2));
        let result = for_each_row_chunk(&matrix, 0, |_chunk| {});
        assert!(matches!(result, Err(AcceleratorError::InvalidChunkSize)));
    }

    #[test]
    fn serial_matmat_matches_expected() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let output = matmat_serial(&left, &right).unwrap();
        assert_eq!(output.dim(), (2, 2));
        assert!((output[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((output[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((output[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn distributed_matmat_matches_serial() {
        let left = Array2::from_shape_vec((5, 4), vec![
            1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 2.0, 0.0, 1.0, -1.0, 3.0, 1.0, 0.0, 2.0, 2.0,
            -1.0, 1.0, 0.0,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((4, 3), vec![
            1.0, 0.0, 2.0, 2.0, 1.0, -1.0, 1.0, 3.0, 0.0, -1.0, 2.0, 1.0,
        ])
        .unwrap();

        let serial = matmat_serial(&left, &right).unwrap();
        let distributed = matmat_distributed(&left, &right, DistributedConfig {
            workers:    3,
            chunk_rows: 2,
            schedule:   DistributedSchedule::Static,
        })
        .unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - distributed[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distributed_matmat_rejects_invalid_config() {
        let left = Array2::<f64>::eye(2);
        let right = Array2::<f64>::eye(2);
        let invalid_workers = matmat_distributed(&left, &right, DistributedConfig {
            workers:    0,
            chunk_rows: 1,
            schedule:   DistributedSchedule::Static,
        });
        assert!(matches!(invalid_workers, Err(AcceleratorError::InvalidWorkerCount)));

        let invalid_chunks = matmat_distributed(&left, &right, DistributedConfig {
            workers:    1,
            chunk_rows: 0,
            schedule:   DistributedSchedule::Static,
        });
        assert!(matches!(invalid_chunks, Err(AcceleratorError::InvalidChunkSize)));
    }

    #[test]
    fn distributed_dynamic_matches_static() {
        let left =
            Array2::from_shape_vec((8, 5), (0..40).map(|value| f64::from(value) * 0.125).collect())
                .unwrap();
        let right =
            Array2::from_shape_vec((5, 7), (0..35).map(|value| f64::from(value) * -0.25).collect())
                .unwrap();

        let static_result = matmat_distributed(&left, &right, DistributedConfig {
            workers:    3,
            chunk_rows: 2,
            schedule:   DistributedSchedule::Static,
        })
        .unwrap();
        let dynamic_result = matmat_distributed(&left, &right, DistributedConfig {
            workers:    3,
            chunk_rows: 2,
            schedule:   DistributedSchedule::Dynamic,
        })
        .unwrap();
        for row in 0..static_result.nrows() {
            for col in 0..static_result.ncols() {
                assert!((static_result[[row, col]] - dynamic_result[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distributed_tiled_matmat_matches_serial() {
        let left =
            Array2::from_shape_vec((7, 5), (0..35).map(|v| f64::from(v) * 0.25).collect()).unwrap();
        let right =
            Array2::from_shape_vec((5, 6), (0..30).map(|v| f64::from(v) * -0.5).collect()).unwrap();

        let serial = matmat_serial(&left, &right).unwrap();
        let tiled = matmat_distributed_tiled(&left, &right, 3, 2, 3).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - tiled[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distributed_tiled_matmat_rejects_invalid_config() {
        let left = Array2::<f64>::eye(2);
        let right = Array2::<f64>::eye(2);

        let invalid_workers = matmat_distributed_tiled(&left, &right, 0, 1, 1);
        assert!(matches!(invalid_workers, Err(AcceleratorError::InvalidWorkerCount)));

        let invalid_rows = matmat_distributed_tiled(&left, &right, 1, 0, 1);
        assert!(matches!(invalid_rows, Err(AcceleratorError::InvalidTileSize)));

        let invalid_cols = matmat_distributed_tiled(&left, &right, 1, 1, 0);
        assert!(matches!(invalid_cols, Err(AcceleratorError::InvalidTileSize)));
    }

    #[test]
    fn backend_dispatch_selects_expected_kernel() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();

        let serial = matmat_with_backend::<CpuBackend>(&left, &right).unwrap();
        let distributed = matmat_with_backend::<DistributedBackend>(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - distributed[[row, col]]).abs() < 1e-12);
            }
        }

        let cuda = matmat_with_backend::<CudaBackend>(&left, &right);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn backend_dispatch_selects_expected_matvec_kernel() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let cpu = matvec_with_backend::<CpuBackend>(&matrix, &vector).unwrap();
        let distributed = matvec_with_backend::<DistributedBackend>(&matrix, &vector).unwrap();
        assert_eq!(cpu, distributed);
        assert!((cpu[0] - 5.0).abs() < 1e-12);
        assert!((cpu[1] - 5.0).abs() < 1e-12);

        let cuda = matvec_with_backend::<CudaBackend>(&matrix, &vector);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn backend_dispatch_selects_expected_batched_matmat_kernel() {
        let left = Array3::from_shape_vec((2, 2, 2), vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ])
        .unwrap();
        let right = Array3::from_shape_vec((2, 2, 2), vec![
            1.0, 0.0, 0.0, 1.0, // identity
            2.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();

        let cpu = batched_matmat_with_backend::<CpuBackend>(&left, &right).unwrap();
        let distributed = batched_matmat_with_backend::<DistributedBackend>(&left, &right).unwrap();
        assert_eq!(cpu, distributed);

        let cuda = batched_matmat_with_backend::<CudaBackend>(&left, &right);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn backend_dispatch_selects_expected_sparse_matvec_kernel() {
        let matrix =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, -1.0]).unwrap();
        let vector = Array1::from_vec(vec![2.0, 3.0, 4.0]);

        let cpu = sparse_matvec_with_backend::<CpuBackend>(&matrix, &vector).unwrap();
        let distributed =
            sparse_matvec_with_backend::<DistributedBackend>(&matrix, &vector).unwrap();
        assert_eq!(cpu, distributed);
        assert!((cpu[0] - 10.0).abs() < 1e-12);
        assert!((cpu[1] + 3.0).abs() < 1e-12);

        let cuda = sparse_matvec_with_backend::<CudaBackend>(&matrix, &vector);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn backend_dispatch_selects_expected_batched_row_matvec_kernel() {
        let vectors = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();

        let cpu = batched_row_matvec_with_backend::<CpuBackend>(&vectors, &matrix).unwrap();
        let distributed =
            batched_row_matvec_with_backend::<DistributedBackend>(&vectors, &matrix).unwrap();
        assert_eq!(cpu, distributed);

        let cuda = batched_row_matvec_with_backend::<CudaBackend>(&vectors, &matrix);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn backend_dispatch_selects_expected_sparse_matmat_kernels() {
        let sparse_left =
            CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let dense_right = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 2.0, 1.0]).unwrap();
        let sparse_right = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 1.0]).unwrap();

        let dense_cpu =
            sparse_matmat_dense_with_backend::<CpuBackend>(&sparse_left, &dense_right).unwrap();
        let dense_distributed =
            sparse_matmat_dense_with_backend::<DistributedBackend>(&sparse_left, &dense_right)
                .unwrap();
        assert_eq!(dense_cpu, dense_distributed);

        let sparse_cpu =
            sparse_matmat_sparse_with_backend::<CpuBackend>(&sparse_left, &sparse_right).unwrap();
        let sparse_distributed =
            sparse_matmat_sparse_with_backend::<DistributedBackend>(&sparse_left, &sparse_right)
                .unwrap();
        assert_eq!(sparse_cpu, sparse_distributed);
    }

    #[test]
    fn backend_dispatch_selects_expected_vector_kernels() {
        let left = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let right = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let dot = dot_with_backend::<CpuBackend>(&left, &right).unwrap();
        assert!((dot - 32.0).abs() < 1e-12);

        let left_rows = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let right_rows = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let l2 = pairwise_l2_with_backend::<CpuBackend>(&left_rows, &right_rows).unwrap();
        let cosine = pairwise_cosine_with_backend::<CpuBackend>(&left_rows, &right_rows).unwrap();
        assert!((l2[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((cosine[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn backend_dispatch_selects_expected_triangular_kernels() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap();
        let rhs_vec = Array1::from_vec(vec![4.0_f64, 8.0]);
        let rhs_mat = Array2::from_shape_vec((2, 2), vec![4.0_f64, 2.0, 8.0, 7.0]).unwrap();

        let vec_solution =
            triangular_solve_vec_with_backend::<CpuBackend, f64>(&matrix, &rhs_vec, true, false)
                .unwrap();
        let mat_solution =
            triangular_solve_mat_with_backend::<CpuBackend, f64>(&matrix, &rhs_mat, true, false)
                .unwrap();
        let vec_reconstructed = matrix.dot(&vec_solution);
        let mat_reconstructed = matrix.dot(&mat_solution);

        for i in 0..rhs_vec.len() {
            assert!((vec_reconstructed[i] - rhs_vec[i]).abs() < 1e-12);
        }
        for row in 0..rhs_mat.nrows() {
            for col in 0..rhs_mat.ncols() {
                assert!((mat_reconstructed[[row, col]] - rhs_mat[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn backend_dispatch_selects_expected_tensor_kernels() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let summed = tensor_sum_last_axis_with_backend::<CpuBackend, f64>(&tensor).unwrap();
        assert_eq!(summed.shape(), &[2]);
        assert!((summed[[0]] - 3.0).abs() < 1e-12);
        assert!((summed[[1]] - 7.0).abs() < 1e-12);

        let left =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0_f64, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let contracted =
            tensor_contract_axes_with_backend::<CpuBackend, f64>(&left, &right, 1, 0).unwrap();
        assert_eq!(contracted.shape(), &[2, 2]);

        let batched_left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ])
        .unwrap();
        let batched_right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f64, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let batched = tensor_batched_matmul_last_two_with_backend::<CpuBackend, f64>(
            &batched_left,
            &batched_right,
        )
        .unwrap();
        assert_eq!(batched.shape(), &[2, 2, 2]);
    }

    #[test]
    fn matmat_rejects_dimension_mismatch() {
        let left = Array2::<f64>::zeros((2, 3));
        let right = Array2::<f64>::zeros((2, 2));
        assert!(matches!(matmat_serial(&left, &right), Err(AcceleratorError::DimensionMismatch)));
        assert!(matches!(
            matmat_accelerated(&left, &right),
            Err(AcceleratorError::DimensionMismatch)
        ));
    }

    #[cfg(not(feature = "accelerator-rayon"))]
    #[test]
    fn accelerated_matmat_requires_feature() {
        let left = Array2::<f64>::eye(2);
        let right = Array2::<f64>::eye(2);
        let result = matmat_accelerated(&left, &right);
        assert!(matches!(result, Err(AcceleratorError::FeatureNotEnabled)));
    }

    #[cfg(not(feature = "accelerator-wgpu"))]
    #[test]
    fn gpu_matmat_requires_feature() {
        let left = Array2::<f32>::eye(2);
        let right = Array2::<f32>::eye(2);
        let result = matmat_gpu_f32(&left, &right);
        assert!(matches!(result, Err(AcceleratorError::FeatureNotEnabled)));
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_matmat_matches_cpu_or_reports_unavailable_device() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let cpu = matmat_with_backend_f32::<CpuBackend>(&left, &right).unwrap();
        match matmat_with_backend_f32::<CudaBackend>(&left, &right) {
            Ok(gpu) => {
                for row in 0..cpu.nrows() {
                    for col in 0..cpu.ncols() {
                        assert!((cpu[[row, col]] - gpu[[row, col]]).abs() < 1e-4);
                    }
                }
            }
            Err(error) => assert!(matches!(error, AcceleratorError::DeviceUnavailable)),
        }
    }

    #[cfg(feature = "accelerator-rayon")]
    #[test]
    fn accelerated_matmat_matches_serial() {
        let left = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 2.0, 0.0, 1.0, -1.0,
        ])
        .unwrap();
        let right =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0, -1.0, 2.0]).unwrap();
        let serial = matmat_serial(&left, &right).unwrap();
        let accelerated = matmat_accelerated(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - accelerated[[row, col]]).abs() < 1e-12);
            }
        }
    }
}
