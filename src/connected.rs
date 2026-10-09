//! SC-W2: compose realization groups through Finitum-owned trace elimination.
use crate::{
    CoupledSystemOperator, FinitumRealization, KrasisError, StateBinding, StateLayout,
    TransactionalOperator,
};
use finitum::{ConnectionRealizationPlan, ConstraintSet, ReducedSystemOperator};
use methodus::{
    BlockLayout, BlockNonlinearOperator, DaeOperator, EvaluationContext, NonlinearOperator,
    NumericError,
};

/// The single-relation, two-operator shape Finitum's
/// [`ConnectionRealizationPlan::system_linearized_diagonal`] covers (`matching` or
/// `nested_p1`): the plan and the inner leaf index of each connection endpoint, retained at
/// construction so the connected Jacobian diagonal is Finitum's exact `diag(Rᵀ J R)` (W10
/// package 1) rather than a Krasis-side sum of leaf diagonals.
#[derive(Clone, Debug)]
struct ConnectedPair {
    plan: ConnectionRealizationPlan,
    leaves: [usize; 2],
}

#[derive(Clone, Debug)]
pub struct ConnectedSystemOperator {
    inner: CoupledSystemOperator,
    constraints: ConstraintSet,
    pair: Option<ConnectedPair>,
    identity: String,
    consistent_initialization: Option<crate::coupled::ConsistentInitialization>,
}
fn numerical(error: finitum::FinitumError) -> NumericError {
    NumericError::Evaluation {
        code: "CONNECTION_ELIMINATION".into(),
        origin: "finitum matching trace constraints".into(),
        message: error.to_string(),
    }
}
impl ConnectedSystemOperator {
    /// The leaf order must be the connection endpoint order. Finitum verifies the concrete
    /// geometry, trace DOFs and essential-boundary non-overlap before returning constraints.
    pub fn new(
        inner: CoupledSystemOperator,
        connection: &ConnectionRealizationPlan,
        operators: [&ReducedSystemOperator; 2],
    ) -> Result<Self, KrasisError> {
        let constraints = connection
            .constraints(operators)
            .map_err(|e| KrasisError::InvalidCoupling(e.to_string()))?;
        let identities = inner
            .realizations()
            .into_iter()
            .map(|r| match r {
                FinitumRealization::ReducedSystem(s) => Some(s.realization_digest()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if identities
            != operators
                .iter()
                .map(|o| Some(o.realization_digest()))
                .collect::<Vec<_>>()
            || DaeOperator::dimension(&inner) != constraints.dof_count()
        {
            return Err(KrasisError::InvalidCoupling(
                "connection operators differ from coupled leaves".into(),
            ));
        }
        let identity = format!(
            "krasis-connected-system/1:{}:{}",
            inner.identity(),
            connection.identity()
        );
        Ok(Self {
            inner,
            constraints,
            pair: Some(ConnectedPair {
                plan: connection.clone(),
                leaves: [0, 1],
            }),
            identity,
            consistent_initialization: None,
        })
    }

    /// Compose matching interfaces over leaves in scientific instance order.
    ///
    /// A system of exactly one relation over exactly two operators whose composed constraints
    /// are the relation's own (same targets, dependencies and orientation) is the shape
    /// Finitum's connected diagonal covers and takes that path; any other shape (several
    /// relations, more leaves, or a relation the union-find re-oriented) keeps the Krasis-side
    /// sum-of-leaf-diagonals convention Finitum does not cover yet.
    pub fn new_system(
        inner: CoupledSystemOperator,
        connections: &[ConnectionRealizationPlan],
        operators: &[&ReducedSystemOperator],
    ) -> Result<Self, KrasisError> {
        let constraints = ConnectionRealizationPlan::system_constraints(connections, operators)
            .map_err(|e| KrasisError::InvalidCoupling(e.to_string()))?;
        if inner.leaves().len() != operators.len()
            || inner
                .realizations()
                .iter()
                .zip(operators)
                .any(|(realization, expected)| {
                    !matches!(realization, FinitumRealization::ReducedSystem(actual)
                    if actual.realization_digest() == expected.realization_digest())
                })
            || inner.realizations().len() != operators.len()
            || DaeOperator::dimension(&inner) != constraints.dof_count()
        {
            return Err(KrasisError::InvalidCoupling(
                "connection operators differ from coupled leaves".into(),
            ));
        }
        let pair = match (connections, operators) {
            ([plan], [first, second]) => {
                let leaves = plan
                    .relation()
                    .ports
                    .each_ref()
                    .map(|port| port.instance.index());
                let endpoints = match leaves {
                    [0, 1] => Some([*first, *second]),
                    [1, 0] => Some([*second, *first]),
                    _ => None,
                };
                endpoints.and_then(|endpoints| {
                    // `system_constraints` has already formed `constraints(endpoints)` on this
                    // same pair, so a refusal here is unreachable in practice; were it reached,
                    // the operator falls back to the sum-of-leaf convention rather than refuse.
                    let own = plan.constraints(endpoints).ok()?;
                    let split = endpoints[0].operator().dimension();
                    let offset = inner.leaf_range(leaves[0]).expect("leaf index").start;
                    let other = inner.leaf_range(leaves[1]).expect("leaf index").start;
                    let global = |local: usize| {
                        if local < split {
                            offset + local
                        } else {
                            other + local - split
                        }
                    };
                    let same_orientation = own.dof_count() == constraints.dof_count()
                        && own.constraints().count() == constraints.constraints().count()
                        && own.constraints().all(|constraint| {
                            constraints.constraints().any(|composed| {
                                composed.target.0 == global(constraint.target.0)
                                    && composed.offset == constraint.offset
                                    && composed.dependencies.len() == constraint.dependencies.len()
                                    && composed
                                        .dependencies
                                        .iter()
                                        .zip(&constraint.dependencies)
                                        .all(|(composed, own)| {
                                            composed.dof.0 == global(own.dof.0)
                                                && composed.weight == own.weight
                                        })
                            })
                        });
                    same_orientation.then(|| ConnectedPair {
                        plan: plan.clone(),
                        leaves,
                    })
                })
            }
            _ => None,
        };
        let mut identities = connections.iter().map(|c| c.identity()).collect::<Vec<_>>();
        identities.sort_unstable();
        let identity = format!(
            "krasis-connected-system/2:{}:{}",
            inner.identity(),
            identities.join(":")
        );
        Ok(Self {
            inner,
            constraints,
            pair,
            identity,
            consistent_initialization: None,
        })
    }

    /// A merged trace row is differential if any participating row stores state.
    /// Eliminated coordinates carry algebraic equality equations.
    pub fn row_kinds(&self) -> Result<Vec<crate::RowKind>, KrasisError> {
        let mut mask = Vec::new();
        for leaf in self.inner.leaves() {
            mask.extend_from_slice(leaf.row_kinds().ok_or_else(|| {
                KrasisError::InvalidCoupling("connected leaf has no row classification".into())
            })?);
        }
        for constraint in self.constraints.constraints() {
            if mask[constraint.target.0] == crate::RowKind::Differential {
                for dependency in &constraint.dependencies {
                    mask[dependency.dof.0] = crate::RowKind::Differential;
                }
            }
            mask[constraint.target.0] = crate::RowKind::Algebraic;
        }
        Ok(mask)
    }

    pub fn with_consistent_initialization(
        mut self,
        newton: methodus::NewtonConfig,
    ) -> Result<Self, KrasisError> {
        let mask = self.row_kinds()?;
        let policy = crate::coupled::RateSolvePolicy::Always;
        self.identity = format!(
            "{}:consistent-init={}",
            self.identity,
            crate::coupled::consistent_initialization_identity(&mask, &newton, policy)
        );
        self.consistent_initialization = Some(crate::coupled::ConsistentInitialization {
            mask,
            newton,
            policy,
        });
        Ok(self)
    }
    pub fn layout(&self) -> &StateLayout {
        self.inner.layout()
    }
    pub fn binding(&self) -> &StateBinding {
        self.inner.binding()
    }
    pub fn physical_state(&self, state: &[f64]) -> Result<Vec<f64>, NumericError> {
        self.constraints.expand(state).map_err(numerical)
    }

    /// Whether the Jacobian diagonal is Finitum's connected `diag(Rᵀ J R)` (the
    /// single-relation, two-operator shape) rather than the Krasis-side sum-of-leaf
    /// convention over a multi-relation union.
    pub fn has_connected_diagonal(&self) -> bool {
        self.pair.is_some()
    }

    /// Finitum's exact connected diagonal for the retained pair: the per-leaf slices of the
    /// connection-reduced point (leaf ranges are preserved under elimination) go to
    /// [`ConnectionRealizationPlan::system_linearized_diagonal`] in endpoint order, and its
    /// `[a; b]` result is scattered back into the leaves' ranges. Target and essential rows
    /// are the identity rows Finitum already sets.
    fn pair_diagonal(
        &self,
        pair: &ConnectedPair,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<Option<Vec<f64>>, NumericError> {
        let realizations = self.inner.realizations();
        let ranges = pair
            .leaves
            .map(|leaf| self.inner.leaf_range(leaf).expect("pair leaf index"));
        let [
            FinitumRealization::ReducedSystem(a),
            FinitumRealization::ReducedSystem(b),
        ] = [realizations[pair.leaves[0]], realizations[pair.leaves[1]]]
        else {
            return Ok(None);
        };
        let Some(connected) = pair
            .plan
            .system_linearized_diagonal(
                [a, b],
                time,
                [&state[ranges[0].clone()], &state[ranges[1].clone()]],
                [
                    &state_rate[ranges[0].clone()],
                    &state_rate[ranges[1].clone()],
                ],
                rate_shift,
            )
            .map_err(numerical)?
        else {
            return Ok(None);
        };
        let split = ranges[0].len();
        let mut diagonal = vec![0.0; DaeOperator::dimension(self)];
        diagonal[ranges[0].clone()].copy_from_slice(&connected[..split]);
        diagonal[ranges[1].clone()].copy_from_slice(&connected[split..]);
        Ok(Some(diagonal))
    }

    /// The multi-relation union shape Finitum's connected diagonal does not cover: weight-1
    /// equivalence classes identify coordinates across leaves and each physical node occurs at
    /// most once per leaf, so there are no within-leaf cross entries in a class and
    /// `diag(Pᵀ J P)` is the transpose restriction of the concatenated leaf diagonals, with
    /// target rows the identity. Anything else (a weighted dependency, a class with two nodes
    /// of one leaf) returns `None`.
    fn union_diagonal(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<Option<Vec<f64>>, NumericError> {
        if self
            .constraints
            .constraints()
            .any(|c| c.dependencies.len() != 1 || c.dependencies[0].weight != 1.0)
        {
            return Ok(None);
        }
        let mut classes = std::collections::BTreeSet::new();
        let leaf_for = |dof: usize| {
            self.inner
                .leaves()
                .iter()
                .enumerate()
                .find(|(index, _)| self.inner.leaf_range(*index).unwrap().contains(&dof))
                .map(|(i, _)| i)
                .unwrap()
        };
        for constraint in self.constraints.constraints() {
            let master = constraint.dependencies[0].dof.0;
            classes.insert((master, leaf_for(master)));
        }
        for constraint in self.constraints.constraints() {
            let master = constraint.dependencies[0].dof.0;
            if !classes.insert((master, leaf_for(constraint.target.0))) {
                return Ok(None);
            }
        }
        let physical = self.constraints.expand(state).map_err(numerical)?;
        let rate = self
            .constraints
            .expand_homogeneous(state_rate)
            .map_err(numerical)?;
        let mut diagonal = Vec::new();
        let mut offset = 0;
        for realization in self.inner.realizations() {
            let FinitumRealization::ReducedSystem(operator) = realization else {
                return Ok(None);
            };
            let n = operator.operator().dimension();
            let Some(local) = DaeOperator::jacobian_diagonal(
                operator,
                context,
                time,
                &physical[offset..offset + n],
                &rate[offset..offset + n],
                rate_shift,
            )?
            else {
                return Ok(None);
            };
            diagonal.extend(local);
            offset += n;
        }
        let mut diagonal = self
            .constraints
            .restrict_transpose(&diagonal)
            .map_err(numerical)?;
        for constraint in self.constraints.constraints() {
            diagonal[constraint.target.0] = 1.0;
        }
        Ok(Some(diagonal))
    }
}
impl DaeOperator for ConnectedSystemOperator {
    fn make_initial_state_consistent(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &mut [f64],
    ) -> Result<(), NumericError> {
        crate::coupled::make_initial_state_consistent_for(
            self,
            self.consistent_initialization.as_ref(),
            context,
            time,
            state,
        )
    }
    /// `diag(Rᵀ J R)` of the connection-reduced system. The single-relation, two-operator shape
    /// is Finitum's exact connected diagonal (weighted `nested_p1` interpolation included: the
    /// within-leaf cross term `Σ w_β J_B[β,γ] w_γ` an interface-adjacent master row gains); the
    /// multi-relation union shape is the Krasis-side sum-of-leaf-diagonals convention. Either
    /// is `None` when the inner composition carries coupling edges (their entries fall into the
    /// eliminated classes) or a leaf supplies no diagonal.
    fn jacobian_diagonal(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<Option<Vec<f64>>, NumericError> {
        if !self.inner.graph().dependencies().is_empty() {
            return Ok(None);
        }
        match &self.pair {
            Some(pair) => self.pair_diagonal(pair, time, state, state_rate, rate_shift),
            None => self.union_diagonal(context, time, state, state_rate, rate_shift),
        }
    }

    fn dimension(&self) -> usize {
        self.constraints.dof_count()
    }
    fn residual(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let physical = self.constraints.expand(state).map_err(numerical)?;
        let rate = self
            .constraints
            .expand_homogeneous(rate)
            .map_err(numerical)?;
        let mut raw = vec![0.0; DaeOperator::dimension(self)];
        DaeOperator::residual(&self.inner, context, time, &physical, &rate, &mut raw)?;
        let mut reduced = self
            .constraints
            .restrict_transpose(&raw)
            .map_err(numerical)?;
        for constraint in self.constraints.constraints() {
            reduced[constraint.target.0] =
                state[constraint.target.0] - physical[constraint.target.0];
        }
        if output.len() != reduced.len() {
            return Err(numerical(finitum::FinitumError::InvalidRealization(
                "connection residual extent mismatch".into(),
            )));
        }
        output.copy_from_slice(&reduced);
        Ok(())
    }
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        rate: &[f64],
        direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let physical = self.constraints.expand(state).map_err(numerical)?;
        let rate = self
            .constraints
            .expand_homogeneous(rate)
            .map_err(numerical)?;
        let delta = self
            .constraints
            .expand_homogeneous(direction)
            .map_err(numerical)?;
        let rate_delta = self
            .constraints
            .expand_homogeneous(rate_direction)
            .map_err(numerical)?;
        let mut raw = vec![0.0; DaeOperator::dimension(self)];
        DaeOperator::jacobian_vector_product(
            &self.inner,
            context,
            time,
            &physical,
            &rate,
            &delta,
            &rate_delta,
            &mut raw,
        )?;
        let mut reduced = self
            .constraints
            .restrict_transpose(&raw)
            .map_err(numerical)?;
        for constraint in self.constraints.constraints() {
            reduced[constraint.target.0] =
                direction[constraint.target.0] - delta[constraint.target.0];
        }
        if output.len() != reduced.len() {
            return Err(numerical(finitum::FinitumError::InvalidRealization(
                "connection product extent mismatch".into(),
            )));
        }
        output.copy_from_slice(&reduced);
        Ok(())
    }
}
impl NonlinearOperator for ConnectedSystemOperator {
    fn dimension(&self) -> usize {
        DaeOperator::dimension(self)
    }
    fn jacobian_diagonal(
        &self,
        context: &EvaluationContext,
        state: &[f64],
    ) -> Result<Option<Vec<f64>>, NumericError> {
        DaeOperator::jacobian_diagonal(self, context, 0.0, state, &vec![0.0; state.len()], 0.0)
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        DaeOperator::residual(
            self,
            context,
            0.0,
            state,
            &vec![0.0; DaeOperator::dimension(self)],
            output,
        )
    }
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; DaeOperator::dimension(self)];
        DaeOperator::jacobian_vector_product(
            self, context, 0.0, state, &zero, direction, &zero, output,
        )
    }
}
impl TransactionalOperator for ConnectedSystemOperator {
    fn identity(&self) -> &str {
        &self.identity
    }
    fn state_layout_identity(&self) -> &str {
        self.inner.state_layout_identity()
    }
    fn realizations(&self) -> Vec<FinitumRealization<'_>> {
        self.inner.realizations()
    }
}
impl BlockNonlinearOperator for ConnectedSystemOperator {
    /// The eliminated (algebraic) rows keep their original index in `self.inner`'s state space
    /// (`new`/`new_system` require `constraints.dof_count() == inner.dimension()`; elimination
    /// here reassigns a target row's equation, it never removes or reindexes a row), so `inner`'s
    /// leaf-range block layout partitions this operator's state exactly as it does the
    /// unconnected composition: a `partitioned` decomposition (SC-W3 package 1) schedules over
    /// the same leaves whether or not their interfaces are matched and eliminated.
    fn block_layout(&self) -> &BlockLayout {
        BlockNonlinearOperator::block_layout(&self.inner)
    }
}
