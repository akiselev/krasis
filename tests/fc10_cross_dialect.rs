use finitum::{
    DiscreteOperator, FiniteVolumeFace, FiniteVolumeRealization, MethodRealization,
    NetworkDaeRealization,
};
use krasis::CrossDialectOperator;
use quantitas::UnitRegistry;
use scientia::{
    AffineMethodKernelSpec, compile_conservation_law_method, compile_network_dae_method,
    compile_semantics,
};
use solverang::{
    BdfConfig, BdfOrder, BdfState, DaeOperator, EvaluationContext, StepOutcome, bdf_step,
    verify_dae_jvp,
};

const SOURCE: &str = r#"
module fixtures.cross_dialect;
model Conservation {
  domain Cells { dimension = 1; coordinates = cartesian; }
  field q: state scalar DG(order=0) on Cells { time_role = differential; };
  property speed = transport_speed(0);
  equation balance on Cells { dt(q) + div(speed * q) = 0; }
}
model Network {
  domain Graph { dimension = 0; coordinates = lumped; }
  field x: state scalar L2(order=0) on Graph { time_role = differential; };
  equation balance on Graph { dt(x) + x = 0; }
}
"#;

fn operator() -> CrossDialectOperator {
    let (finite_volume, network) = blocks();
    CrossDialectOperator::new(
        finite_volume,
        network,
        vec![vec![0.5], vec![-0.5]],
        vec![vec![0.25, -0.25]],
    )
    .unwrap()
}

fn blocks() -> (DiscreteOperator, DiscreteOperator) {
    let module = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic;
    let finite_volume = compile_conservation_law_method(
        &module,
        "Conservation",
        "balance",
        "q",
        AffineMethodKernelSpec {
            name: "upwind".into(),
            inputs: vec!["minus".into(), "plus".into()],
            coefficients: vec![1.0, 0.0],
            constant: 0.0,
        },
    )
    .unwrap();
    let network = compile_network_dae_method(&module, "Network", &["balance"], &["x"]).unwrap();
    let finite_volume = MethodRealization::FiniteVolume(
        FiniteVolumeRealization::new(
            finite_volume,
            vec![1.0, 1.0],
            vec![
                FiniteVolumeFace { minus: 0, plus: 1 },
                FiniteVolumeFace { minus: 1, plus: 0 },
            ],
        )
        .unwrap(),
    );
    let network = MethodRealization::NetworkDae(
        NetworkDaeRealization::new(network, vec![vec![1.0]], vec![vec![2.0]], vec![0.0]).unwrap(),
    );
    (
        DiscreteOperator::sibling(finite_volume),
        DiscreteOperator::sibling(network),
    )
}

#[test]
fn explicit_off_diagonal_blocks_form_a_real_bidirectional_dae() {
    let operator = operator();
    let context = EvaluationContext::reproducible();
    let mut residual = vec![0.0; 3];
    operator
        .residual(
            &context,
            0.0,
            &[1.0, 0.0, 2.0],
            &[0.1, 0.2, 0.3],
            &mut residual,
        )
        .unwrap();
    assert_eq!(residual, vec![2.1, -1.8, 4.55]);
    assert!(operator.identity().starts_with("blake3:"));
    let discrepancy = verify_dae_jvp(
        &operator,
        &context,
        0.3,
        &[1.0, 0.0, 2.0],
        &[0.1, 0.2, 0.3],
        &[-0.4, 0.7, 0.2],
        &[0.5, -0.1, 0.6],
        1.0e-6,
    )
    .unwrap();
    assert!(discrepancy < 1.0e-9);
}

#[test]
fn identity_is_canonical_and_covers_coupling_matrices() {
    let baseline = operator();
    assert_eq!(baseline.identity(), operator().identity());

    let (finite_volume, network) = blocks();
    let changed = CrossDialectOperator::new(
        finite_volume,
        network,
        vec![vec![0.75], vec![-0.5]],
        vec![vec![0.25, -0.25]],
    )
    .unwrap();
    assert_ne!(baseline.identity(), changed.identity());
}

#[test]
fn solverang_advances_the_cross_dialect_system_without_method_specific_policy() {
    let operator = operator();
    let context = EvaluationContext::reproducible();
    let state = BdfState::initialize(&operator, &context, 0.0, vec![1.0, 0.0, 0.5]).unwrap();
    let config = BdfConfig {
        order: BdfOrder::One,
        minimum_step: 0.1,
        maximum_step: 0.1,
        ..BdfConfig::default()
    };
    let StepOutcome::Accepted(accepted) =
        bdf_step(&operator, &context, &state, 0.1, &config).unwrap()
    else {
        panic!("the first-order fixed step must be accepted")
    };
    assert_eq!(accepted.state.accepted_steps, 1);
    assert_eq!(accepted.state.time, 0.1);
    assert_ne!(accepted.state.values, state.values);
}

#[test]
fn same_family_and_one_way_placeholders_are_refused() {
    let (finite_volume, network) = blocks();
    assert!(
        CrossDialectOperator::new(
            finite_volume.clone(),
            finite_volume,
            vec![vec![0.5, 0.0], vec![0.0, 0.5]],
            vec![vec![0.5, 0.0], vec![0.0, 0.5]],
        )
        .is_err()
    );
    let (finite_volume, _) = blocks();
    assert!(
        CrossDialectOperator::new(
            finite_volume,
            network,
            vec![vec![0.5], vec![-0.5]],
            vec![vec![0.0, 0.0]],
        )
        .is_err()
    );
}
