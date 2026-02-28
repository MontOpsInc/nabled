use nabled::qr::{QRConfig, QRError, ndarray_qr};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])?;
    let config = QRConfig::default();

    let qr = ndarray_qr::compute_qr(&matrix, &config)?;
    println!("rank: {}", qr.rank);

    let reconstructed = ndarray_qr::reconstruct_matrix(&qr);
    println!("reconstructed:\n{reconstructed:?}");

    let a = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0])?;
    let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
    let x = ndarray_qr::solve_least_squares(&a, &b, &config)?;
    println!("least squares solution: {x:?}");

    let empty = Array2::<f64>::zeros((0, 0));
    match ndarray_qr::compute_qr(&empty, &config) {
        Err(QRError::EmptyMatrix) => println!("caught empty matrix error"),
        _ => println!("unexpected result"),
    }

    Ok(())
}
