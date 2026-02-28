use nabled::svd::ndarray_svd;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;

    let svd = ndarray_svd::decompose(&matrix)?;
    println!("singular values: {:?}", svd.singular_values);
    println!("condition number: {}", ndarray_svd::condition_number(&svd));

    let reconstructed = ndarray_svd::reconstruct_matrix(&svd);
    println!("reconstructed:\n{reconstructed:?}");

    Ok(())
}
