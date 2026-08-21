//! Krasis-owned composition across distinct Finitum discretization dialects.

use crate::KrasisError;
use finitum::DiscreteOperator;
use solverang::{
    BlockLayout, BlockNonlinearOperator, BlockSpec, DaeOperator, EvaluationContext,
    NonlinearOperator, NumericError,
};

/// Bidirectionally coupled pair of concrete operators from distinct method families.
///
/// Off-diagonal matrices are explicit derivatives: `left_from_right` contributes to the left
/// residual from the right state, and `right_from_left` contributes in the opposite direction.
#[derive(Clone, Debug)]
pub struct CrossDialectOperator {
    left: DiscreteOperator,
    right: DiscreteOperator,
    left_from_right: Vec<Vec<f64>>,
    right_from_left: Vec<Vec<f64>>,
    layout: BlockLayout,
    identity: String,
}

impl CrossDialectOperator {
    pub fn new(
        left: DiscreteOperator,
        right: DiscreteOperator,
        left_from_right: Vec<Vec<f64>>,
        right_from_left: Vec<Vec<f64>>,
    ) -> Result<Self, KrasisError> {
        if left.family_identity() == right.family_identity() {
            return Err(KrasisError::InvalidCoupling(format!(
                "cross-dialect composition requires distinct families, got `{}` twice",
                left.family_identity()
            )));
        }
        let left_dimension = left.dimension();
        let right_dimension = right.dimension();
        validate_matrix(
            &left_from_right,
            left_dimension,
            right_dimension,
            "left-from-right coupling",
        )?;
        validate_matrix(
            &right_from_left,
            right_dimension,
            left_dimension,
            "right-from-left coupling",
        )?;
        if !has_nonzero(&left_from_right) || !has_nonzero(&right_from_left) {
            return Err(KrasisError::InvalidCoupling(
                "a real cross-dialect system requires nonzero coupling in both directions".into(),
            ));
        }
        let layout = BlockLayout::new(vec![
            BlockSpec {
                name: left.family_identity().into(),
                length: left_dimension,
                residual_scale: 1.0,
            },
            BlockSpec {
                name: right.family_identity().into(),
                length: right_dimension,
                residual_scale: 1.0,
            },
        ])
        .map_err(|error| KrasisError::InvalidCoupling(error.to_string()))?;
        let identity = format!(
            "krasis-cross-dialect/1:left-family={}:left={}:right-family={}:right={}:lr={left_from_right:?}:rl={right_from_left:?}",
            left.family_identity(),
            left.identity(),
            right.family_identity(),
            right.identity()
        );
        Ok(Self {
            left,
            right,
            left_from_right,
            right_from_left,
            layout,
            identity,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn left(&self) -> &DiscreteOperator {
        &self.left
    }

    pub fn right(&self) -> &DiscreteOperator {
        &self.right
    }

    pub fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }

    fn split<'a>(
        &self,
        values: &'a [f64],
        label: &str,
    ) -> Result<(&'a [f64], &'a [f64]), NumericError> {
        require_len(label, values.len(), DaeOperator::dimension(self))?;
        require_finite(label, values)?;
        Ok(values.split_at(self.left.dimension()))
    }

    fn split_mut<'a>(
        &self,
        values: &'a mut [f64],
        label: &str,
    ) -> Result<(&'a mut [f64], &'a mut [f64]), NumericError> {
        require_len(label, values.len(), DaeOperator::dimension(self))?;
        Ok(values.split_at_mut(self.left.dimension()))
    }
}

impl DaeOperator for CrossDialectOperator {
    fn dimension(&self) -> usize {
        self.left.dimension() + self.right.dimension()
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let (left_state, right_state) = self.split(state, "cross-dialect state")?;
        let (left_rate, right_rate) = self.split(state_rate, "cross-dialect state rate")?;
        let (left_output, right_output) = self.split_mut(output, "cross-dialect residual")?;
        self.left
            .residual(context, time, left_state, left_rate, left_output)?;
        self.right
            .residual(context, time, right_state, right_rate, right_output)?;
        add_matrix_action(&self.left_from_right, right_state, left_output);
        add_matrix_action(&self.right_from_left, left_state, right_output);
        require_finite("cross-dialect residual", output)
    }

    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let (left_state, right_state) = self.split(state, "cross-dialect state")?;
        let (left_rate, right_rate) = self.split(state_rate, "cross-dialect state rate")?;
        let (left_direction, right_direction) =
            self.split(state_direction, "cross-dialect state direction")?;
        let (left_rate_direction, right_rate_direction) =
            self.split(rate_direction, "cross-dialect rate direction")?;
        let (left_output, right_output) = self.split_mut(output, "cross-dialect JVP")?;
        self.left.jacobian_vector_product(
            context,
            time,
            left_state,
            left_rate,
            left_direction,
            left_rate_direction,
            left_output,
        )?;
        self.right.jacobian_vector_product(
            context,
            time,
            right_state,
            right_rate,
            right_direction,
            right_rate_direction,
            right_output,
        )?;
        add_matrix_action(&self.left_from_right, right_direction, left_output);
        add_matrix_action(&self.right_from_left, left_direction, right_output);
        require_finite("cross-dialect JVP", output)
    }
}

impl NonlinearOperator for CrossDialectOperator {
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

impl BlockNonlinearOperator for CrossDialectOperator {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

fn validate_matrix(
    matrix: &[Vec<f64>],
    rows: usize,
    columns: usize,
    label: &str,
) -> Result<(), KrasisError> {
    if matrix.len() != rows
        || matrix
            .iter()
            .any(|row| row.len() != columns || row.iter().any(|value| !value.is_finite()))
    {
        return Err(KrasisError::InvalidCoupling(format!(
            "{label} must be a finite {rows} by {columns} matrix"
        )));
    }
    Ok(())
}

fn has_nonzero(matrix: &[Vec<f64>]) -> bool {
    matrix.iter().flatten().any(|value| *value != 0.0)
}

fn add_matrix_action(matrix: &[Vec<f64>], input: &[f64], output: &mut [f64]) {
    for (row, result) in matrix.iter().zip(output) {
        *result += row
            .iter()
            .zip(input)
            .map(|(coefficient, value)| coefficient * value)
            .sum::<f64>();
    }
}

fn require_len(label: &str, actual: usize, expected: usize) -> Result<(), NumericError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NumericError::DimensionMismatch {
            operation: label.into(),
            expected,
            actual,
        })
    }
}

fn require_finite(label: &str, values: &[f64]) -> Result<(), NumericError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(NumericError::NonFinite {
            operation: label.into(),
            index,
        })
    } else {
        Ok(())
    }
}
