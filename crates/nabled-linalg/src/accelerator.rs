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
        AcceleratorError, BackendKind, CpuBackend, CudaBackend, execute,
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
    #[cfg(feature = "accelerator-wgpu")]
    use crate::accelerator::dispatch::{
        batched_matmat_with_backend_f32, matmat_with_backend_f32, matvec_with_backend_f32,
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
    fn backend_dispatch_selects_expected_kernel() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();

        let serial = matmat_with_backend::<CpuBackend>(&left, &right).unwrap();
        let cuda = matmat_with_backend::<CudaBackend>(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - cuda[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn backend_dispatch_selects_expected_matvec_kernel() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let cpu = matvec_with_backend::<CpuBackend>(&matrix, &vector).unwrap();
        assert!((cpu[0] - 5.0).abs() < 1e-12);
        assert!((cpu[1] - 5.0).abs() < 1e-12);

        let cuda = matvec_with_backend::<CudaBackend>(&matrix, &vector).unwrap();
        assert_eq!(cuda, cpu);
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
        let cuda = batched_matmat_with_backend::<CudaBackend>(&left, &right).unwrap();
        assert_eq!(cuda, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_sparse_matvec_kernel() {
        let matrix =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, -1.0]).unwrap();
        let vector = Array1::from_vec(vec![2.0, 3.0, 4.0]);

        let cpu = sparse_matvec_with_backend::<CpuBackend>(&matrix, &vector).unwrap();
        assert!((cpu[0] - 10.0).abs() < 1e-12);
        assert!((cpu[1] + 3.0).abs() < 1e-12);

        let cuda = sparse_matvec_with_backend::<CudaBackend>(&matrix, &vector).unwrap();
        assert_eq!(cuda, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_batched_row_matvec_kernel() {
        let vectors = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();

        let cpu = batched_row_matvec_with_backend::<CpuBackend>(&vectors, &matrix).unwrap();
        let cuda = batched_row_matvec_with_backend::<CudaBackend>(&vectors, &matrix).unwrap();
        assert_eq!(cuda, cpu);
    }

    #[test]
    fn backend_dispatch_selects_expected_sparse_matmat_kernels() {
        let sparse_left =
            CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let dense_right = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 2.0, 1.0]).unwrap();
        let sparse_right = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 1.0]).unwrap();

        let dense_cpu =
            sparse_matmat_dense_with_backend::<CpuBackend>(&sparse_left, &dense_right).unwrap();
        let sparse_cpu =
            sparse_matmat_sparse_with_backend::<CpuBackend>(&sparse_left, &sparse_right).unwrap();

        let dense_cuda =
            sparse_matmat_dense_with_backend::<CudaBackend>(&sparse_left, &dense_right).unwrap();
        let sparse_cuda =
            sparse_matmat_sparse_with_backend::<CudaBackend>(&sparse_left, &sparse_right).unwrap();
        assert_eq!(dense_cuda, dense_cpu);
        assert_eq!(sparse_cuda, sparse_cpu);
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

        let dot_cuda = dot_with_backend::<CudaBackend>(&left, &right);
        let l2_cuda = pairwise_l2_with_backend::<CudaBackend>(&left_rows, &right_rows);
        let cosine_cuda = pairwise_cosine_with_backend::<CudaBackend>(&left_rows, &right_rows);
        assert!((dot_cuda.unwrap() - dot).abs() < 1e-12);
        assert_eq!(l2_cuda.unwrap(), l2);
        assert_eq!(cosine_cuda.unwrap(), cosine);
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

        let vec_cuda =
            triangular_solve_vec_with_backend::<CudaBackend, f64>(&matrix, &rhs_vec, true, false);
        let mat_cuda =
            triangular_solve_mat_with_backend::<CudaBackend, f64>(&matrix, &rhs_mat, true, false);
        assert_eq!(vec_cuda.unwrap(), vec_solution);
        assert_eq!(mat_cuda.unwrap(), mat_solution);
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

        let summed_cuda = tensor_sum_last_axis_with_backend::<CudaBackend, f64>(&tensor);
        let contracted_cuda =
            tensor_contract_axes_with_backend::<CudaBackend, f64>(&left, &right, 1, 0);
        let batched_cuda = tensor_batched_matmul_last_two_with_backend::<CudaBackend, f64>(
            &batched_left,
            &batched_right,
        );
        assert_eq!(summed_cuda.unwrap(), summed);
        assert_eq!(contracted_cuda.unwrap(), contracted);
        assert_eq!(batched_cuda.unwrap(), batched);
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
        let left = Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let cpu = matmat_with_backend_f32::<CpuBackend>(&left, &right).unwrap();
        let gpu = matmat_with_backend_f32::<CudaBackend>(&left, &right).unwrap();
        for row in 0..cpu.nrows() {
            for col in 0..cpu.ncols() {
                assert!((cpu[[row, col]] - gpu[[row, col]]).abs() < 1e-4);
            }
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_matvec_matches_cpu() {
        let matrix =
            Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]);
        let cpu = matvec_with_backend_f32::<CpuBackend>(&matrix, &vector).unwrap();
        let gpu = matvec_with_backend_f32::<CudaBackend>(&matrix, &vector).unwrap();
        for i in 0..cpu.len() {
            assert!((cpu[i] - gpu[i]).abs() < 1e-4);
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_batched_matmat_matches_cpu() {
        let left = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ])
        .unwrap();
        let right = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f32, 0.0, 0.0, 1.0, //
            2.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let cpu = batched_matmat_with_backend_f32::<CpuBackend>(&left, &right).unwrap();
        let gpu = batched_matmat_with_backend_f32::<CudaBackend>(&left, &right).unwrap();
        for b in 0..cpu.dim().0 {
            for r in 0..cpu.dim().1 {
                for c in 0..cpu.dim().2 {
                    assert!((cpu[[b, r, c]] - gpu[[b, r, c]]).abs() < 1e-4);
                }
            }
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_tensor_batched_matmul_matches_cpu() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f32, 0.0, 0.0, 1.0, //
            2.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let cpu =
            tensor_batched_matmul_last_two_with_backend::<CpuBackend, f32>(&left, &right).unwrap();
        let gpu =
            tensor_batched_matmul_last_two_with_backend::<CudaBackend, f32>(&left, &right).unwrap();
        for (lhs, rhs) in cpu.iter().zip(gpu.iter()) {
            assert!((*lhs - *rhs).abs() < 1e-4);
        }
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_tensor_batched_matmul_f64_falls_back_to_cpu() {
        let left = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let right =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();

        let result = tensor_batched_matmul_last_two_with_backend::<CudaBackend, f64>(&left, &right);
        let cpu =
            tensor_batched_matmul_last_two_with_backend::<CpuBackend, f64>(&left, &right).unwrap();
        assert_eq!(result.unwrap(), cpu);
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_tensor_contract_and_reduction_fall_back_to_cpu() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let contract =
            tensor_contract_axes_with_backend::<CudaBackend, f32>(&tensor, &tensor, 1, 0);
        let reduced = tensor_sum_last_axis_with_backend::<CudaBackend, f32>(&tensor);
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
