//! PCA: dimensionality reduction and explained variance

use nalgebra::DMatrix;
use rust_linalg::pca;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PCA Example ===\n");

    let data = DMatrix::from_row_slice(
        5,
        3,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    );

    let pca_result = pca::nalgebra_pca::compute_pca(&data, Some(2))?;

    println!("Original data: 5 samples x 3 features");
    println!("Reduced to 2 components\n");

    println!("Explained variance ratio: {:?}", {
        pca_result
            .explained_variance_ratio
            .iter()
            .collect::<Vec<_>>()
    });
    println!(
        "Sum: {:.4}",
        pca_result.explained_variance_ratio.iter().sum::<f64>()
    );

    let projected = pca::nalgebra_pca::transform(&data, &pca_result);
    println!(
        "\nProjected data (scores): {} x {}",
        projected.nrows(),
        projected.ncols()
    );

    Ok(())
}
