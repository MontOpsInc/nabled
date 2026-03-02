use nabled::sparse::{self, CooMatrix};
use ndarray::Array1;

fn residual_norm(
    matrix: &sparse::CsrMatrix,
    x: &Array1<f64>,
    rhs: &Array1<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let ax = sparse::matvec(matrix, x)?;
    let mut sq_sum = 0.0_f64;
    for i in 0..rhs.len() {
        let delta = ax[i] - rhs[i];
        sq_sum += delta * delta;
    }
    Ok(sq_sum.sqrt())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coo = CooMatrix::new(
        5,
        5,
        vec![
            0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, // row
        ],
        vec![
            0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, // col
        ],
        vec![
            4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0, // value
        ],
    )?;
    let matrix = coo.to_csr()?;

    let rhs_a = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
    let rhs_b = Array1::from_vec(vec![0.5, -1.0, 0.0, 1.0, 0.5]);

    let factorization = sparse::sparse_lu_factor(&matrix)?;
    let x_a = sparse::sparse_lu_solve_with_factorization(&matrix, &rhs_a, &factorization)?;
    let x_b = sparse::sparse_lu_solve_with_factorization(&matrix, &rhs_b, &factorization)?;

    let direct_residual_a = residual_norm(&matrix, &x_a, &rhs_a)?;
    let direct_residual_b = residual_norm(&matrix, &x_b, &rhs_b)?;
    println!("Direct sparse LU residuals: A={direct_residual_a:.3e}, B={direct_residual_b:.3e}");

    let x_a_iter = sparse::pcg_solve(&matrix, &rhs_a, 1e-10, 200)?;
    let x_b_iter = sparse::pcg_solve(&matrix, &rhs_b, 1e-10, 200)?;
    let iter_residual_a = residual_norm(&matrix, &x_a_iter, &rhs_a)?;
    let iter_residual_b = residual_norm(&matrix, &x_b_iter, &rhs_b)?;
    println!("Iterative PCG residuals: A={iter_residual_a:.3e}, B={iter_residual_b:.3e}");

    Ok(())
}
