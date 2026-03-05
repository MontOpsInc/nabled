//! Compile-time backend contracts and accelerator kernels.

pub mod backends;
pub mod cpu;
pub mod dispatch;
pub mod gpu;
pub mod kernels;

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};

    use crate::accelerator::backends::{
        AcceleratorError, BackendKind, CpuBackend, GpuBackend, execute,
    };
    use crate::accelerator::cpu::{for_each_row_chunk, matmat_accelerated, matmat_serial};
    use crate::accelerator::dispatch::{
        batched_matmat_with_backend, batched_row_matvec_with_backend, dot_with_backend,
        matmat_with_backend, matvec_with_backend, pairwise_cosine_with_backend,
        pairwise_l2_with_backend, sparse_matmat_dense_with_backend,
        sparse_matmat_sparse_with_backend, sparse_matvec_with_backend,
        tensor_batched_matmul_last_two_with_backend, tensor_contract_axes_with_backend,
        tensor_sum_last_axis_with_backend, triangular_solve_mat_with_backend,
        triangular_solve_vec_with_backend,
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
    fn gpu_backend_returns_error() {
        let gpu = execute::<GpuBackend, _, _>(|| 1);
        assert!(matches!(gpu, Err(AcceleratorError::UnsupportedBackend(BackendKind::Gpu))));
    }

    #[test]
    fn chunking_visits_all_rows() {
        let matrix = Array2::from_shape_vec((5, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64, 9.0_f64,
            10.0_f64,
        ])
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
        let left = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64,
        ])
        .unwrap();
        let output = matmat_serial(&left, &right).unwrap();
        assert_eq!(output.dim(), (2, 2));
        assert!((output[[0, 0]] - 5.0_f64).abs() < 1e-12_f64);
        assert!((output[[0, 1]] - 2.0_f64).abs() < 1e-12_f64);
        assert!((output[[1, 0]] - 3.0_f64).abs() < 1e-12_f64);
        assert!((output[[1, 1]] - 4.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn backend_dispatch_selects_expected_kernel() {
        let left = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64,
        ])
        .unwrap();

        let serial = matmat_with_backend::<CpuBackend, f64>(&left, &right).unwrap();
        let gpu = matmat_with_backend::<GpuBackend, f64>(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - gpu[[row, col]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn backend_dispatch_selects_expected_matvec_kernel() {
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64,
        ])
        .unwrap();
        let vector = Array1::from_vec(vec![1.0_f64, 2.0_f64, 3.0_f64]);

        let cpu = matvec_with_backend::<CpuBackend, f64>(&matrix, &vector).unwrap();
        assert!((cpu[0] - 5.0_f64).abs() < 1e-12_f64);
        assert!((cpu[1] - 5.0_f64).abs() < 1e-12_f64);

        let gpu = matvec_with_backend::<GpuBackend, f64>(&matrix, &vector).unwrap();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_batched_matmat_kernel() {
        let left = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, // batch 0
            5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64, // batch 1
        ])
        .unwrap();
        let right = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, // identity
            2.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();

        let cpu = batched_matmat_with_backend::<CpuBackend, f64>(&left, &right).unwrap();
        let gpu = batched_matmat_with_backend::<GpuBackend, f64>(&left, &right).unwrap();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_sparse_matvec_kernel() {
        let matrix =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0_f64, -1.0_f64])
                .unwrap();
        let vector = Array1::from_vec(vec![2.0_f64, 3.0_f64, 4.0_f64]);

        let cpu = sparse_matvec_with_backend::<CpuBackend, f64>(&matrix, &vector).unwrap();
        assert!((cpu[0] - 10.0_f64).abs() < 1e-12_f64);
        assert!((cpu[1] + 3.0_f64).abs() < 1e-12_f64);

        let gpu = sparse_matvec_with_backend::<GpuBackend, f64>(&matrix, &vector).unwrap();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_batched_row_matvec_kernel() {
        let vectors = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 0.5_f64, -1.0_f64, 1.0_f64,
        ])
        .unwrap();
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 0.0_f64, 1.0_f64, 1.0_f64,
        ])
        .unwrap();

        let cpu = batched_row_matvec_with_backend::<CpuBackend, f64>(&vectors, &matrix).unwrap();
        let gpu = batched_row_matvec_with_backend::<GpuBackend, f64>(&vectors, &matrix).unwrap();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_sparse_matmat_kernels() {
        let sparse_left =
            CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 2.0_f64, 3.0_f64])
                .unwrap();
        let dense_right =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64]).unwrap();
        let sparse_right =
            CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 1.0_f64]).unwrap();

        let dense_serial =
            sparse_matmat_dense_with_backend::<CpuBackend, f64>(&sparse_left, &dense_right)
                .unwrap();
        let sparse_serial =
            sparse_matmat_sparse_with_backend::<CpuBackend, f64>(&sparse_left, &sparse_right)
                .unwrap();

        let dense_accel =
            sparse_matmat_dense_with_backend::<GpuBackend, f64>(&sparse_left, &dense_right)
                .unwrap();
        let sparse_accel =
            sparse_matmat_sparse_with_backend::<GpuBackend, f64>(&sparse_left, &sparse_right)
                .unwrap();
        assert_eq!(dense_accel, dense_serial);
        assert_eq!(sparse_accel, sparse_serial);
    }

    #[test]
    fn backend_dispatch_selects_expected_vector_kernels() {
        let left = Array1::from_vec(vec![1.0_f64, 2.0_f64, 3.0_f64]);
        let right = Array1::from_vec(vec![4.0_f64, 5.0_f64, 6.0_f64]);
        let dot = dot_with_backend::<CpuBackend, f64>(&left, &right).unwrap();
        assert!((dot - 32.0_f64).abs() < 1e-12_f64);

        let left_rows =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        let right_rows =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        let l2 = pairwise_l2_with_backend::<CpuBackend, f64>(&left_rows, &right_rows).unwrap();
        let cosine =
            pairwise_cosine_with_backend::<CpuBackend, f64>(&left_rows, &right_rows).unwrap();
        assert!((l2[[0, 0]] - 0.0_f64).abs() < 1e-12_f64);
        assert!((cosine[[0, 0]] - 1.0_f64).abs() < 1e-12_f64);

        let dot_gpu = dot_with_backend::<GpuBackend, f64>(&left, &right);
        let l2_gpu = pairwise_l2_with_backend::<GpuBackend, f64>(&left_rows, &right_rows);
        let cosine_gpu = pairwise_cosine_with_backend::<GpuBackend, f64>(&left_rows, &right_rows);
        assert!((dot_gpu.unwrap() - dot).abs() < 1e-12_f64);
        assert_eq!(l2_gpu.unwrap(), l2);
        assert_eq!(cosine_gpu.unwrap(), cosine);
    }

    #[test]
    fn backend_dispatch_selects_expected_triangular_kernels() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let rhs_vec = Array1::from_vec(vec![4.0_f64, 8.0_f64]);
        let rhs_mat =
            Array2::from_shape_vec((2, 2), vec![4.0_f64, 2.0_f64, 8.0_f64, 7.0_f64]).unwrap();

        let vec_solution =
            triangular_solve_vec_with_backend::<CpuBackend, f64>(&matrix, &rhs_vec, true, false)
                .unwrap();
        let mat_solution =
            triangular_solve_mat_with_backend::<CpuBackend, f64>(&matrix, &rhs_mat, true, false)
                .unwrap();
        let vec_reconstructed = matrix.dot(&vec_solution);
        let mat_reconstructed = matrix.dot(&mat_solution);

        for i in 0..rhs_vec.len() {
            assert!((vec_reconstructed[i] - rhs_vec[i]).abs() < 1e-12_f64);
        }
        for row in 0..rhs_mat.nrows() {
            for col in 0..rhs_mat.ncols() {
                assert!((mat_reconstructed[[row, col]] - rhs_mat[[row, col]]).abs() < 1e-12_f64);
            }
        }

        let vec_gpu =
            triangular_solve_vec_with_backend::<GpuBackend, f64>(&matrix, &rhs_vec, true, false);
        let mat_gpu =
            triangular_solve_mat_with_backend::<GpuBackend, f64>(&matrix, &rhs_mat, true, false);
        assert_eq!(vec_gpu.unwrap(), vec_solution);
        assert_eq!(mat_gpu.unwrap(), mat_solution);
    }

    #[test]
    fn backend_dispatch_selects_expected_tensor_kernels() {
        let tensor =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])
                .unwrap();
        let summed = tensor_sum_last_axis_with_backend::<CpuBackend, f64>(&tensor).unwrap();
        assert_eq!(summed.shape(), &[2]);
        assert!((summed[[0]] - 3.0_f64).abs() < 1e-12_f64);
        assert!((summed[[1]] - 7.0_f64).abs() < 1e-12_f64);

        let left = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64,
        ])
        .unwrap();
        let contracted =
            tensor_contract_axes_with_backend::<CpuBackend, f64>(&left, &right, 1, 0).unwrap();
        assert_eq!(contracted.shape(), &[2, 2]);

        let batched_left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64,
        ])
        .unwrap();
        let batched_right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let batched = tensor_batched_matmul_last_two_with_backend::<CpuBackend, f64>(
            &batched_left,
            &batched_right,
        )
        .unwrap();
        assert_eq!(batched.shape(), &[2, 2, 2]);

        let summed_gpu = tensor_sum_last_axis_with_backend::<GpuBackend, f64>(&tensor);
        let contracted_gpu =
            tensor_contract_axes_with_backend::<GpuBackend, f64>(&left, &right, 1, 0);
        let batched_gpu = tensor_batched_matmul_last_two_with_backend::<GpuBackend, f64>(
            &batched_left,
            &batched_right,
        );
        assert_eq!(summed_gpu.unwrap(), summed);
        assert_eq!(contracted_gpu.unwrap(), contracted);
        assert_eq!(batched_gpu.unwrap(), batched);
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
    fn gpu_matmat_matches_cpu() {
        let left = Array2::from_shape_vec((2, 3), vec![
            1.0_f32, 2.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![
            1.0_f32, 0.0_f32, 2.0_f32, 1.0_f32, 1.0_f32, 3.0_f32,
        ])
        .unwrap();
        let cpu = matmat_with_backend::<CpuBackend, f32>(&left, &right).unwrap();
        let gpu = matmat_with_backend::<GpuBackend, f32>(&left, &right).unwrap();
        for row in 0..cpu.nrows() {
            for col in 0..cpu.ncols() {
                assert!((cpu[[row, col]] - gpu[[row, col]]).abs() < 1e-4_f32);
            }
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_matvec_matches_cpu() {
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f32, 2.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32,
        ])
        .unwrap();
        let vector = Array1::from_vec(vec![1.0_f32, 2.0_f32, 3.0_f32]);
        let cpu = matvec_with_backend::<CpuBackend, f32>(&matrix, &vector).unwrap();
        let gpu = matvec_with_backend::<GpuBackend, f32>(&matrix, &vector).unwrap();
        for i in 0..cpu.len() {
            assert!((cpu[i] - gpu[i]).abs() < 1e-4_f32);
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_batched_matmat_matches_cpu() {
        let left = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, //
            5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32,
        ])
        .unwrap();
        let right = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, //
            2.0_f32, 1.0_f32, 1.0_f32, 2.0_f32,
        ])
        .unwrap();
        let cpu = batched_matmat_with_backend::<CpuBackend, f32>(&left, &right).unwrap();
        let gpu = batched_matmat_with_backend::<GpuBackend, f32>(&left, &right).unwrap();
        for b in 0..cpu.dim().0 {
            for r in 0..cpu.dim().1 {
                for c in 0..cpu.dim().2 {
                    assert!((cpu[[b, r, c]] - gpu[[b, r, c]]).abs() < 1e-4_f32);
                }
            }
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_batched_row_matvec_matches_cpu() {
        let vectors = Array2::from_shape_vec((2, 3), vec![
            1.0_f32, 0.0_f32, 2.0_f32, 0.5_f32, -1.0_f32, 1.0_f32,
        ])
        .unwrap();
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f32, 2.0_f32, 3.0_f32, 0.0_f32, 1.0_f32, 1.0_f32,
        ])
        .unwrap();
        let cpu = batched_row_matvec_with_backend::<CpuBackend, f32>(&vectors, &matrix).unwrap();
        let gpu = batched_row_matvec_with_backend::<GpuBackend, f32>(&vectors, &matrix).unwrap();
        for row in 0..cpu.nrows() {
            for col in 0..cpu.ncols() {
                assert!((cpu[[row, col]] - gpu[[row, col]]).abs() < 1e-4_f32);
            }
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_vector_kernels_match_cpu() {
        let left = Array1::from_vec(vec![1.0_f32, 2.0_f32, 3.0_f32]);
        let right = Array1::from_vec(vec![4.0_f32, 5.0_f32, 6.0_f32]);
        let baseline_dot = dot_with_backend::<CpuBackend, f32>(&left, &right).unwrap();
        let backend_dot = dot_with_backend::<GpuBackend, f32>(&left, &right).unwrap();
        assert!((baseline_dot - backend_dot).abs() < 1e-4_f32);

        let left_rows =
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap();
        let right_rows =
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap();
        let baseline_l2 =
            pairwise_l2_with_backend::<CpuBackend, f32>(&left_rows, &right_rows).unwrap();
        let backend_l2 =
            pairwise_l2_with_backend::<GpuBackend, f32>(&left_rows, &right_rows).unwrap();
        let baseline_cosine =
            pairwise_cosine_with_backend::<CpuBackend, f32>(&left_rows, &right_rows).unwrap();
        let backend_cosine =
            pairwise_cosine_with_backend::<GpuBackend, f32>(&left_rows, &right_rows).unwrap();
        for row in 0..baseline_l2.nrows() {
            for col in 0..baseline_l2.ncols() {
                assert!((baseline_l2[[row, col]] - backend_l2[[row, col]]).abs() < 1e-4_f32);
                assert!(
                    (baseline_cosine[[row, col]] - backend_cosine[[row, col]]).abs() < 1e-4_f32
                );
            }
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_tensor_batched_matmul_matches_cpu() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, //
            5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, //
            2.0_f32, 1.0_f32, 1.0_f32, 2.0_f32,
        ])
        .unwrap();
        let cpu =
            tensor_batched_matmul_last_two_with_backend::<CpuBackend, f32>(&left, &right).unwrap();
        let gpu =
            tensor_batched_matmul_last_two_with_backend::<GpuBackend, f32>(&left, &right).unwrap();
        for (lhs, rhs) in cpu.iter().zip(gpu.iter()) {
            assert!((*lhs - *rhs).abs() < 1e-4_f32);
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_tensor_batched_matmul_f64_falls_back_to_cpu() {
        let left =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])
                .unwrap();
        let right =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64])
                .unwrap();

        let result = tensor_batched_matmul_last_two_with_backend::<GpuBackend, f64>(&left, &right);
        let cpu =
            tensor_batched_matmul_last_two_with_backend::<CpuBackend, f64>(&left, &right).unwrap();
        assert_eq!(result.unwrap(), cpu);
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_tensor_contract_and_reduction_match_cpu() {
        let tensor =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])
                .unwrap();
        let contract = tensor_contract_axes_with_backend::<GpuBackend, f32>(&tensor, &tensor, 1, 0);
        let reduced = tensor_sum_last_axis_with_backend::<GpuBackend, f32>(&tensor);
        let cpu_contract =
            tensor_contract_axes_with_backend::<CpuBackend, f32>(&tensor, &tensor, 1, 0).unwrap();
        let cpu_reduced = tensor_sum_last_axis_with_backend::<CpuBackend, f32>(&tensor).unwrap();
        assert_eq!(contract.unwrap(), cpu_contract);
        assert_eq!(reduced.unwrap(), cpu_reduced);
    }

    #[cfg(feature = "accelerator-rayon")]
    #[test]
    fn accelerated_matmat_matches_serial() {
        let left = Array2::from_shape_vec((3, 4), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 3.0_f64, 2.0_f64, 2.0_f64,
            0.0_f64, 1.0_f64, -1.0_f64,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, -1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let serial = matmat_serial(&left, &right).unwrap();
        let accelerated = matmat_accelerated(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - accelerated[[row, col]]).abs() < 1e-12_f64);
            }
        }
    }
}
