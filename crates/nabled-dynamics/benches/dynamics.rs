use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nabled_dynamics::config::{DynamicsConfig, ForwardDynamicsMethod};
use nabled_dynamics::crba::mass_matrix;
use nabled_dynamics::fd::forward_dynamics_with_config;
use nabled_dynamics::rnea::rnea_with_config;
use nabled_kinematics::chain::ChainSpec;
use nabled_model::fixture::load_planar2r_json;
use ndarray::arr1;

fn planar2r_system() -> (nabled_model::robot::RobotModel<f64>, ChainSpec<f64>, DynamicsConfig<f64>)
{
    let fixture = load_planar2r_json().expect("fixture");
    let model = fixture.to_robot_model::<f64>().expect("model");
    let chain = fixture.to_chain_spec::<f64>().expect("chain");
    let config = DynamicsConfig {
        gravity:          fixture.gravity.unwrap_or([0.0, -9.81, 0.0]),
        forward_dynamics: ForwardDynamicsMethod::Aba,
    };
    (model, chain, config)
}

fn benchmark_dynamics(c: &mut Criterion) {
    let (model, chain, config) = planar2r_system();
    let q = arr1(&[0.3_f64, 0.5]);
    let qd = arr1(&[0.1_f64, -0.2]);
    let qdd = arr1(&[0.5_f64, 0.25]);
    let tau = rnea_with_config(&model, &chain, &q.view(), &qd.view(), &qdd.view(), &config)
        .expect("rnea");
    let lu_config = DynamicsConfig { forward_dynamics: ForwardDynamicsMethod::CrbaLu, ..config };

    let mut group = c.benchmark_group("dynamics_planar2r");
    let _ = group.bench_function("rnea", |bench| {
        bench.iter(|| {
            rnea_with_config(
                black_box(&model),
                black_box(&chain),
                black_box(&q.view()),
                black_box(&qd.view()),
                black_box(&qdd.view()),
                black_box(&config),
            )
            .expect("rnea")
        });
    });
    let _ = group.bench_function("mass_matrix", |bench| {
        bench.iter(|| {
            mass_matrix(
                black_box(&model),
                black_box(&chain),
                black_box(&q.view()),
                black_box(&config),
            )
            .expect("mass matrix")
        });
    });
    let _ = group.bench_function("forward_dynamics_aba", |bench| {
        bench.iter(|| {
            forward_dynamics_with_config(
                black_box(&model),
                black_box(&chain),
                black_box(&q.view()),
                black_box(&qd.view()),
                black_box(&tau.view()),
                black_box(&config),
            )
            .expect("forward dynamics aba")
        });
    });
    let _ = group.bench_function("forward_dynamics_crba_lu", |bench| {
        bench.iter(|| {
            forward_dynamics_with_config(
                black_box(&model),
                black_box(&chain),
                black_box(&q.view()),
                black_box(&qd.view()),
                black_box(&tau.view()),
                black_box(&lu_config),
            )
            .expect("forward dynamics crba+lu")
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_dynamics);
criterion_main!(benches);
