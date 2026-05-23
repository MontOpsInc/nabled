use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled_kinematics::chain::{ChainSpec, DhConvention, JointType};
use nabled_kinematics::fk::fk_view;
use nabled_kinematics::ik::{IkConfig, IkWorkspace, inverse_kinematics_dls_into};
use nabled_kinematics::jacobian::jacobian_view;
use ndarray::arr1;

fn six_dof_chain() -> ChainSpec<f64> {
    ChainSpec::from_dh(
        DhConvention::Standard,
        vec![JointType::Revolute; 6],
        arr1(&[0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0]),
        arr1(&[
            std::f64::consts::FRAC_PI_2,
            0.0,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            0.0,
        ]),
        arr1(&[0.089159, 0.0, 0.0, 0.43307, 0.0, 0.0]),
        arr1(&[0.0; 6]),
    )
    .expect("valid 6-DOF chain")
}

fn benchmark_kinematics(c: &mut Criterion) {
    let chain = six_dof_chain();
    let q = arr1(&[0.2_f64, -0.3, 0.5, 0.1, -0.2, 0.4]);
    let target = fk_view(&chain, &q.view()).expect("fk");

    let mut group = c.benchmark_group("kinematics_6dof");
    group.bench_function("fk", |bench| {
        bench.iter(|| fk_view(black_box(&chain), black_box(&q.view())).expect("fk"));
    });
    group.bench_function("jacobian", |bench| {
        bench.iter(|| jacobian_view(black_box(&chain), black_box(&q.view())).expect("jacobian"));
    });
    group.bench_function("ik_dls_cold_start", |bench| {
        let q_init = arr1(&[0.0; 6]);
        let mut workspace = IkWorkspace::new(6);
        let mut output = arr1(&[0.0; 6]);
        let config = IkConfig::default();
        bench.iter(|| {
            inverse_kinematics_dls_into(
                black_box(&chain),
                black_box(&q_init),
                black_box(&target),
                black_box(&config),
                None,
                black_box(&mut workspace),
                black_box(&mut output),
            )
            .expect("ik");
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_kinematics);
criterion_main!(benches);
