//! W10 package 4: Krasis's N-block composition structure handed to Methodus's block
//! Gauss-Seidel preconditioner ([`BlockGaussSeidelPreconditioner`]) behind the same
//! [`PreconditionerFactory`] shape Sinbad already consumes for [`methodus::JacobiFactory`].
//!
//! Krasis owns the composition (which rows belong to which leaf, and that the leaf blocks of
//! [`crate::CoupledSystemOperator`] / [`crate::ConnectedSystemOperator`] tile the operator); it
//! supplies exactly what [`BlockGaussSeidelPreconditioner::new`] consumes and no algorithm:
//!
//! - the [`BlockLayout`] (one block per leaf, [`BlockNonlinearOperator::block_layout`]);
//! - one per-block diagonal [`Preconditioner`]: Jacobi over the block's slice of the operator's
//!   own [`methodus::NonlinearOperator::jacobian_diagonal`] at the linearization point (the exact
//!   owner-supplied diagonal, never probed; an operator without one refuses typed, exactly as
//!   `JacobiFactory` does);
//! - every off-diagonal coupling action `A_ij · v` at the linearization point
//!   ([`BlockCoupling`]), computed as the **masked full Jacobian-vector product**: `v` is
//!   zero-padded into block `j`, pushed through the operator's `jacobian_vector_product` at the
//!   frozen point and restricted to block `i`. For this bounded shape (two to a few leaves) that
//!   is exact by definition and costs one JVP per application; assembling the action from the
//!   edge / connection exchange structure instead would be an optimisation, not a change of
//!   value, and is not done here.
//!
//! No coupling declares [`BlockCouplingAction::transpose_of`]: Krasis knows no owner-declared
//! transposed structure between leaves today (no edge is declared the transpose of another, and
//! an eliminated interface mixes one leaf's rows into another's), so the symmetric sweep
//! declares [`OperatorSymmetry::Unknown`] (which CG/MINRES refuse by design, W9 decision 3) and
//! the forward/backward sweeps `Nonsymmetric`. Symmetry is never tested numerically and then
//! declared.

use std::sync::Arc;

use methodus::{
    BlockCouplingAction, BlockDiagonalPreconditioner, BlockGaussSeidelPreconditioner, BlockLayout,
    BlockNonlinearOperator, BlockSpec, EvaluationContext, GaussSeidelSweep, LinearOperator,
    NumericError, OperatorSymmetry, Preconditioner, PreconditionerFactory,
};

fn require_len(operation: &str, actual: usize, expected: usize) -> Result<(), NumericError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NumericError::DimensionMismatch {
            operation: operation.to_owned(),
            expected,
            actual,
        })
    }
}

fn require_finite(operation: &str, values: &[f64]) -> Result<(), NumericError> {
    match values.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(NumericError::NonFinite {
            operation: operation.to_owned(),
            index,
        }),
        None => Ok(()),
    }
}

/// `A_ij · v` of a block-structured operator at a frozen linearization point, as the masked
/// full Jacobian-vector product (see the module docs). Declares no transpose.
pub struct BlockCoupling<'op, Op: BlockNonlinearOperator + ?Sized> {
    operator: &'op Op,
    point: Arc<Vec<f64>>,
    row: usize,
    column: usize,
}

impl<Op: BlockNonlinearOperator + ?Sized> std::fmt::Debug for BlockCoupling<'_, Op> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlockCoupling")
            .field("row", &self.row)
            .field("column", &self.column)
            .field("dimension", &self.point.len())
            .finish()
    }
}

impl<'op, Op: BlockNonlinearOperator + ?Sized> BlockCoupling<'op, Op> {
    /// Refuses a point whose length is not the operator's dimension or holds a non-finite
    /// value, an out-of-range block, and `row == column` (a diagonal block is a diagonal solve,
    /// never a coupling -- the same rule `BlockGaussSeidelPreconditioner::new` enforces).
    pub fn new(
        operator: &'op Op,
        point: Arc<Vec<f64>>,
        row: usize,
        column: usize,
    ) -> Result<Self, NumericError> {
        require_len("block coupling point", point.len(), operator.dimension())?;
        require_finite("block coupling point", &point)?;
        let blocks = operator.block_layout().blocks().len();
        if row >= blocks || column >= blocks {
            return Err(NumericError::InvalidInput {
                message: format!(
                    "block coupling ({row}, {column}) names an out-of-range block ({blocks} \
                     blocks in the operator's layout)"
                ),
            });
        }
        if row == column {
            return Err(NumericError::InvalidInput {
                message: format!(
                    "block coupling ({row}, {column}) couples a block to itself; the diagonal \
                     block is a diagonal solve"
                ),
            });
        }
        Ok(Self {
            operator,
            point,
            row,
            column,
        })
    }

    /// The linearization point this action is frozen at.
    pub fn point(&self) -> &[f64] {
        &self.point
    }
}

impl<Op: BlockNonlinearOperator + ?Sized> BlockCouplingAction for BlockCoupling<'_, Op> {
    fn row_block(&self) -> usize {
        self.row
    }

    fn column_block(&self) -> usize {
        self.column
    }

    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let layout = self.operator.block_layout();
        let row = layout.blocks()[self.row].range();
        let column = layout.blocks()[self.column].range();
        require_len("block coupling input", input.len(), column.len())?;
        require_len("block coupling output", output.len(), row.len())?;
        require_finite("block coupling input", input)?;
        let dimension = self.operator.dimension();
        let mut padded = vec![0.0; dimension];
        padded[column].copy_from_slice(input);
        let mut full = vec![0.0; dimension];
        self.operator
            .jacobian_vector_product(context, &self.point, &padded, &mut full)?;
        output.copy_from_slice(&full[row]);
        require_finite("block coupling action", output)
    }
}

/// [`PreconditionerFactory`] producing Methodus's block Gauss-Seidel preconditioner over a
/// Krasis block-structured operator (one block per leaf): per-block Jacobi diagonal solves from
/// the operator's own diagonal and masked-JVP coupling actions, both frozen at the Newton
/// iterate the solver hands [`PreconditionerFactory::build`].
///
/// The factory must be built over the operator the Newton-Krylov solve runs on (the `jacobian`
/// handed to `build` is checked for its extent only; its action and diagonal are taken from
/// this operator at `state`), exactly as a caller passing `&preconditioner` as a factory
/// vouches for the preconditioner fitting the operator.
pub struct BlockGaussSeidelFactory<'op, Op: BlockNonlinearOperator + ?Sized> {
    operator: &'op Op,
    sweep: GaussSeidelSweep,
}

impl<Op: BlockNonlinearOperator + ?Sized> std::fmt::Debug for BlockGaussSeidelFactory<'_, Op> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlockGaussSeidelFactory")
            .field("blocks", &self.operator.block_layout().blocks().len())
            .field("sweep", &self.sweep)
            .finish()
    }
}

impl<'op, Op: BlockNonlinearOperator + ?Sized> BlockGaussSeidelFactory<'op, Op> {
    #[must_use]
    pub const fn new(operator: &'op Op, sweep: GaussSeidelSweep) -> Self {
        Self { operator, sweep }
    }

    #[must_use]
    pub const fn sweep(&self) -> GaussSeidelSweep {
        self.sweep
    }

    /// One coupling action per ordered pair of distinct blocks, frozen at `point`.
    pub fn couplings(&self, point: &[f64]) -> Result<Vec<BlockCoupling<'op, Op>>, NumericError> {
        let point = Arc::new(point.to_vec());
        let blocks = self.operator.block_layout().blocks().len();
        let mut couplings = Vec::with_capacity(blocks * blocks.saturating_sub(1));
        for row in 0..blocks {
            for column in 0..blocks {
                if row != column {
                    couplings.push(BlockCoupling::new(
                        self.operator,
                        Arc::clone(&point),
                        row,
                        column,
                    )?);
                }
            }
        }
        Ok(couplings)
    }

    /// The per-block Jacobi solves: the inverse of the operator's own Jacobian diagonal at
    /// `point`, sliced per block. Refuses typed when the operator supplies no diagonal (the
    /// same refusal `JacobiFactory` makes), a diagonal of the wrong extent, or a non-finite
    /// diagonal or inverse.
    pub fn diagonal_solves(
        &self,
        context: &EvaluationContext,
        point: &[f64],
    ) -> Result<Vec<BlockDiagonalPreconditioner>, NumericError> {
        require_len(
            "block Gauss-Seidel point",
            point.len(),
            self.operator.dimension(),
        )?;
        let diagonal = self
            .operator
            .jacobian_diagonal(context, point)?
            .ok_or_else(|| NumericError::InvalidInput {
                message: "block Gauss-Seidel requires an owner-supplied Jacobian diagonal for \
                          its per-block Jacobi solves"
                    .into(),
            })?;
        require_len(
            "block Gauss-Seidel diagonal",
            diagonal.len(),
            self.operator.dimension(),
        )?;
        require_finite("block Gauss-Seidel diagonal", &diagonal)?;
        let inverse = diagonal.iter().map(|value| 1.0 / value).collect::<Vec<_>>();
        require_finite("block Gauss-Seidel inverse diagonal", &inverse)?;
        self.operator
            .block_layout()
            .blocks()
            .iter()
            .map(|block| {
                let layout = BlockLayout::new(vec![BlockSpec {
                    name: block.name().to_owned(),
                    length: block.range().len(),
                    residual_scale: 1.0,
                }])
                .map_err(|error| NumericError::InvalidInput {
                    message: error.to_string(),
                })?;
                BlockDiagonalPreconditioner::new(layout, inverse[block.range()].to_vec())
            })
            .collect()
    }
}

impl<Op: BlockNonlinearOperator + ?Sized> PreconditionerFactory
    for BlockGaussSeidelFactory<'_, Op>
{
    fn build<'a>(
        &'a self,
        context: &EvaluationContext,
        jacobian: &dyn LinearOperator,
        state: &[f64],
    ) -> Result<Option<Box<dyn Preconditioner + 'a>>, NumericError> {
        require_len(
            "block Gauss-Seidel Jacobian rows",
            jacobian.rows(),
            self.operator.dimension(),
        )?;
        require_len(
            "block Gauss-Seidel Jacobian columns",
            jacobian.columns(),
            self.operator.dimension(),
        )?;
        let preconditioner = BlockGaussSeidel {
            layout: self.operator.block_layout().clone(),
            solves: self.diagonal_solves(context, state)?,
            couplings: self.couplings(state)?,
            sweep: self.sweep,
        };
        // Methodus validates the block structure once here; every later application rebuilds
        // the same validated view over the owned solves and actions.
        preconditioner.methodus()?;
        Ok(Some(Box::new(preconditioner)))
    }
}

/// The owned solves and coupling actions a [`BlockGaussSeidelFactory`] built, applied through
/// Methodus's [`BlockGaussSeidelPreconditioner`] (which borrows its parts, so the borrowing view
/// is rebuilt per application over this owner; the construction is a bounded validation).
struct BlockGaussSeidel<'op, Op: BlockNonlinearOperator + ?Sized> {
    layout: BlockLayout,
    solves: Vec<BlockDiagonalPreconditioner>,
    couplings: Vec<BlockCoupling<'op, Op>>,
    sweep: GaussSeidelSweep,
}

impl<Op: BlockNonlinearOperator + ?Sized> BlockGaussSeidel<'_, Op> {
    fn methodus(&self) -> Result<BlockGaussSeidelPreconditioner<'_>, NumericError> {
        let solves = self
            .solves
            .iter()
            .map(|solve| solve as &dyn Preconditioner)
            .collect();
        let couplings = self
            .couplings
            .iter()
            .map(|coupling| coupling as &dyn BlockCouplingAction)
            .collect();
        BlockGaussSeidelPreconditioner::new(self.layout.clone(), solves, couplings, self.sweep)
    }
}

impl<Op: BlockNonlinearOperator + ?Sized> Preconditioner for BlockGaussSeidel<'_, Op> {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    /// Methodus's own declaration over these parts (`Nonsymmetric` for a one-directional
    /// sweep, `Unknown` for the symmetric sweep since no coupling declares its transpose).
    fn symmetry(&self) -> OperatorSymmetry {
        self.methodus()
            .map_or(OperatorSymmetry::Unknown, |preconditioner| {
                preconditioner.symmetry()
            })
    }

    fn apply_inverse(
        &self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.methodus()?
            .apply_inverse(context, right_hand_side, output)
    }
}
