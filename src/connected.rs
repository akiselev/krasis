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
        })
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
