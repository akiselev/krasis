use krasis::{Block, BlockId, BlockLayout, FieldId, SimulationState, TransactionPhase};

#[test]
fn rejected_trial_is_bit_identical_and_checkpoint_restores() {
    let layout = BlockLayout::new(vec![Block {
        id: BlockId::new("temperature"),
        range: 0..2,
    }])
    .unwrap();
    let field = FieldId::new("temperature");
    let mut state = SimulationState::new(2);
    state
        .insert_field(field.clone(), vec![300.0, 301.0])
        .unwrap();

    let before = serde_json::to_vec(&state).unwrap();
    state.begin_trial().unwrap();
    state.set_trial(&field, &[500.0, 600.0]).unwrap();
    state.rollback().unwrap();
    assert_eq!(serde_json::to_vec(&state).unwrap(), before);

    state.begin_trial().unwrap();
    state.set_trial(&field, &[302.0, 303.0]).unwrap();
    state.commit(0.25).unwrap();
    assert_eq!(state.phase(), TransactionPhase::Committed);
    let checkpoint = state.checkpoint(&layout).unwrap();

    state.begin_trial().unwrap();
    state.set_trial(&field, &[900.0, 900.0]).unwrap();
    state.commit(0.5).unwrap();
    state.restore(&layout, &checkpoint).unwrap();
    assert_eq!(state.committed(&field).unwrap(), &[302.0, 303.0]);
    assert_eq!(state.time(), 0.25);
}
