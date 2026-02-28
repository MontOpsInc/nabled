use nabled::pca::ndarray_pca;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::from_shape_vec((5, 3), vec![
        2.5, 2.4, 0.1, 0.5, 0.7, 0.2, 2.2, 2.9, 0.3, 1.9, 2.2, 0.4, 3.1, 3.0, 0.5,
    ])?;

    let pca_result = ndarray_pca::compute_pca(&data, Some(2))?;
    println!("components:\n{:?}", pca_result.components);

    let projected = ndarray_pca::transform(&data, &pca_result);
    println!("projected:\n{projected:?}");

    Ok(())
}
