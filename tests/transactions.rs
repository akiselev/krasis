use krasis::{
    BlockId, FieldId, KrasisError, SimulationState, StateBlock, StateLayout, TransactionPhase,
};

#[test]
fn rejected_trial_is_bit_identical_and_checkpoint_restores() {
    let layout =
        StateLayout::new(vec![StateBlock::new(BlockId::new("temperature"), 0..2)]).unwrap();
    let field = FieldId::new("temperature");
    let mut state = SimulationState::new(layout, 2);
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
    let checkpoint = state.checkpoint().unwrap();

    state.begin_trial().unwrap();
    state.set_trial(&field, &[900.0, 900.0]).unwrap();
    state.commit(0.5).unwrap();
    state.restore(&checkpoint).unwrap();
    assert_eq!(state.committed(&field).unwrap(), &[302.0, 303.0]);
    assert_eq!(state.time(), 0.25);
    assert_eq!(state.history(&field).unwrap().len(), 1);
}

#[test]
fn state_structure_is_locked_during_a_trial() {
    let layout = StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..1)]).unwrap();
    let field = FieldId::new("u");
    let mut state = SimulationState::new(layout, 1);
    state.insert_field(field, vec![0.0]).unwrap();
    state.begin_trial().unwrap();

    assert_eq!(
        state.insert_constitutive("material", vec![1.0]),
        Err(KrasisError::StructureDuringTrial)
    );
    state.rollback().unwrap();
}

#[test]
fn malformed_restore_does_not_partially_mutate_state() {
    let layout = StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..2)]).unwrap();
    let field = FieldId::new("u");
    let mut state = SimulationState::new(layout, 2);
    state.insert_field(field, vec![1.0, 2.0]).unwrap();
    state.insert_constitutive("material", vec![3.0]).unwrap();
    let mut checkpoint = state.checkpoint().unwrap();
    checkpoint.fields.get_mut("u").unwrap().pop();
    let before = serde_json::to_vec(&state).unwrap();

    assert!(state.restore(&checkpoint).is_err());
    assert_eq!(serde_json::to_vec(&state).unwrap(), before);
}

#[test]
fn constitutive_state_uses_the_same_commit_and_rollback_boundary() {
    let layout = StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..1)]).unwrap();
    let mut state = SimulationState::new(layout, 1);
    state.insert_field(FieldId::new("u"), vec![1.0]).unwrap();
    state.insert_constitutive("material", vec![3.0]).unwrap();

    state.begin_trial().unwrap();
    state.set_constitutive_trial("material", &[9.0]).unwrap();
    state.rollback().unwrap();
    assert_eq!(state.constitutive("material").unwrap().committed(), &[3.0]);

    state.begin_trial().unwrap();
    state.set_constitutive_trial("material", &[5.0]).unwrap();
    state.commit(0.25).unwrap();
    assert_eq!(state.constitutive("material").unwrap().committed(), &[5.0]);
}

#[test]
fn step_overflow_leaves_the_trial_uncommitted() {
    let layout = StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..1)]).unwrap();
    let field = FieldId::new("u");
    let mut state = SimulationState::new(layout, 0);
    state.insert_field(field.clone(), vec![1.0]).unwrap();
    let mut checkpoint = state.checkpoint().unwrap();
    checkpoint.step = u64::MAX;
    state.restore(&checkpoint).unwrap();
    state.begin_trial().unwrap();
    state.set_trial(&field, &[2.0]).unwrap();

    assert!(state.commit(1.0).is_err());
    assert_eq!(state.phase(), TransactionPhase::Trial);
    assert_eq!(state.committed(&field).unwrap(), &[1.0]);
    assert_eq!(state.trial(&field).unwrap(), &[2.0]);
}
