//! SC-W2: compose realization groups through Finitum-owned trace elimination.
use crate::{
    CoupledSystemOperator, FinitumRealization, KrasisError, StateBinding, StateLayout,
    TransactionalOperator,
};
use finitum::{ConnectionRealizationPlan, ConstraintSet, ReducedSystemOperator};
use methodus::{DaeOperator, EvaluationContext, NonlinearOperator, NumericError};

#[derive(Clone, Debug)]
pub struct ConnectedSystemOperator {
    inner: CoupledSystemOperator,
    constraints: ConstraintSet,
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
            identity,
            consistent_initialization: None,
        })
    }

    /// Compose matching interfaces over leaves in scientific instance order.
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
        // Weighted interpolation needs within-leaf cross entries in diag(P^T J P).
        // A sum of leaf diagonals is not exact for a nonmatching trace.
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
        // Each physical node occurs at most once per leaf; matching identifies
        // coordinates across leaves. There are no within-leaf cross entries in
        // an equivalence class. Thus diag(P^T J P) is the sum of these diagonals.
        let mut diagonal = self
            .constraints
            .restrict_transpose(&diagonal)
            .map_err(numerical)?;
        for constraint in self.constraints.constraints() {
            diagonal[constraint.target.0] = 1.0;
        }
        Ok(Some(diagonal))
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
