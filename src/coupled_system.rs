//! Composition across realization plans: [`CoupledSystemOperator`] (SC-W1 skeleton; SV1-F1,
//! SV7-F2, SV4-H1 keep their IDs).
//!
//! Finitum composes every block that shares one realization plan (one mesh, one method family,
//! any number of model instances). Krasis composes **across** plans: each [`CoupledLeaf`] is one
//! realization group exposed as a `methodus::DaeOperator` with its own [`StateLayout`] and
//! `SemanticId` binding, and each [`CouplingEdge`] is a linear action from one leaf's state (or
//! state rate) into another leaf's residual -- the shape a Finitum transfer/connection
//! realization or an FC10-style explicit cross-derivative block has. The composed operator
//! implements Methodus's `DaeOperator`, `NonlinearOperator` and `BlockNonlinearOperator`
//! contracts over the concatenated state, so `bdf_step` (Newton inside BDF), `solve_newton`
//! (monolithic) and `solve_blocks` (partitioned Gauss-Seidel/Jacobi over the leaves) all drive it
//! unchanged, and [`crate::CoupledExecution`] encloses it in the same trial/commit/rollback
//! transaction with checkpoints binding Krasis state to Methodus BDF history.
//!
//! The [`CouplingGraph`] is SV7-F2's explicit coupling graph: leaves are nodes, edges are
//! typed dependencies, and its strongly connected components (dependencies first) are the
//! sequential schedule a DAG admits and the fixed-point blocks a cycle requires.
//!
//! Nothing here names a physics, a mesh, or a kernel: leaves are opaque Methodus operators and
//! edges are opaque Methodus linear actions. Each carries a caller-supplied content identity
//! (a Finitum digest, a Scientia relation digest) that the composed `krasis-coupled-system/1`
//! identity is built from.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use methodus::{
    BlockLayout as SolverBlockLayout, BlockNonlinearOperator, BlockSpec, DaeOperator,
    EvaluationContext, LinearOperator, NewtonConfig, NonlinearOperator, NumericError,
};
use serde::{Deserialize, Serialize};

use crate::coupled::{
    ConsistentInitialization, TransactionalOperator, consistent_initialization_identity,
    solve_consistent_state_rate_for,
};
use crate::{
    BlockId, CoupledOperator, KrasisError, RowKind, StateBinding, StateBlock, StateLayout,
};

/// One realization group: a Methodus DAE operator over its own state layout, with every block
/// bound to a system-level semantic id.
#[derive(Clone)]
pub struct CoupledLeaf {
    name: String,
    operator: Arc<dyn DaeOperator>,
    layout: StateLayout,
    binding: StateBinding,
    row_kinds: Option<Vec<RowKind>>,
    identity: String,
}

impl fmt::Debug for CoupledLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoupledLeaf")
            .field("name", &self.name)
            .field("dimension", &self.operator.dimension())
            .field("layout", &self.layout)
            .field("binding", &self.binding)
            .field("row_kinds", &self.row_kinds)
            .field("identity", &self.identity)
            .finish()
    }
}

impl CoupledLeaf {
    /// A leaf over any Methodus DAE operator. `identity` is the operator's content identity
    /// (for a Finitum-backed leaf its realization digest); `binding` must name exactly the
    /// blocks of `layout`, and `layout.width()` must equal the operator's dimension.
    pub fn new(
        name: impl Into<String>,
        operator: Arc<dyn DaeOperator>,
        layout: StateLayout,
        binding: StateBinding,
        identity: impl Into<String>,
    ) -> Result<Self, KrasisError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(KrasisError::InvalidCoupling(
                "a coupled leaf needs a non-empty name".into(),
            ));
        }
        if operator.dimension() != layout.width() {
            return Err(KrasisError::InvalidCoupling(format!(
                "leaf `{name}` operator dimension {} differs from its state width {}",
                operator.dimension(),
                layout.width()
            )));
        }
        let layout_blocks: BTreeSet<&BlockId> =
            layout.blocks().iter().map(StateBlock::id).collect();
        let binding_blocks: BTreeSet<&BlockId> = binding.blocks().collect();
        if layout_blocks != binding_blocks {
            return Err(KrasisError::StateBindingLayoutMismatch);
        }
        Ok(Self {
            name,
            operator,
            layout,
            binding,
            row_kinds: None,
            identity: identity.into(),
        })
    }

    /// A leaf over a Finitum realization already composed by [`CoupledOperator`] (which must
    /// carry a `StateBinding`); its identity, layout and any recorded differential/algebraic
    /// mask are inherited.
    pub fn realization(
        name: impl Into<String>,
        operator: CoupledOperator,
        layout: StateLayout,
    ) -> Result<Self, KrasisError> {
        let name = name.into();
        let Some(binding) = operator.state_binding().cloned() else {
            return Err(KrasisError::InvalidCoupling(format!(
                "leaf `{name}` needs a coupled operator built with a state binding"
            )));
        };
        if layout.identity() != operator.state_layout_identity() {
            return Err(KrasisError::InvalidCoupling(format!(
                "leaf `{name}` state layout does not match its coupled operator's layout"
            )));
        }
        let row_kinds = operator.row_kinds().map(<[RowKind]>::to_vec);
        let identity = operator.identity().to_owned();
        let mut leaf = Self::new(name, Arc::new(operator), layout, binding, identity)?;
        leaf.row_kinds = row_kinds;
        Ok(leaf)
    }

    /// Records the per-row differential/algebraic mask this leaf contributes to the composed
    /// consistent initialization ([`CoupledSystemOperator::with_consistent_initialization`]).
    pub fn with_row_kinds(mut self, row_kinds: Vec<RowKind>) -> Result<Self, KrasisError> {
        if row_kinds.len() != self.dimension() {
            return Err(KrasisError::ConsistentInitializationMaskLength {
                actual: row_kinds.len(),
                expected: self.dimension(),
            });
        }
        self.row_kinds = Some(row_kinds);
        Ok(self)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    pub fn binding(&self) -> &StateBinding {
        &self.binding
    }

    pub fn row_kinds(&self) -> Option<&[RowKind]> {
        self.row_kinds.as_deref()
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}

/// Which argument of the source leaf a [`CouplingEdge`] acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CouplingArgument {
    State,
    Rate,
}

/// A linear contribution `action * (state | rate of column leaf)` added to the row leaf's
/// residual. `action` is `rows = row leaf dimension` by `columns = column leaf dimension`;
/// `identity` is the opaque content identity of the relation it realizes (a Scientia relation
/// digest, a Finitum transfer digest, or a fixture label).
#[derive(Clone)]
pub struct CouplingEdge {
    row: String,
    column: String,
    argument: CouplingArgument,
    action: Arc<dyn LinearOperator>,
    identity: String,
}

impl fmt::Debug for CouplingEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CouplingEdge")
            .field("row", &self.row)
            .field("column", &self.column)
            .field("argument", &self.argument)
            .field("rows", &self.action.rows())
            .field("columns", &self.action.columns())
            .field("identity", &self.identity)
            .finish()
    }
}

impl CouplingEdge {
    pub fn new(
        row: impl Into<String>,
        column: impl Into<String>,
        argument: CouplingArgument,
        action: Arc<dyn LinearOperator>,
        identity: impl Into<String>,
    ) -> Self {
        Self {
            row: row.into(),
            column: column.into(),
            argument,
            action,
            identity: identity.into(),
        }
    }

    /// An explicit sparse coupling block (FC10's explicit cross-derivative shape); the edge
    /// identity is `krasis-coupling-matrix/1`, blake3 over the serialized matrix, so changing
    /// any entry changes the composed identity.
    pub fn matrix(
        row: impl Into<String>,
        column: impl Into<String>,
        argument: CouplingArgument,
        matrix: methodus::CsrMatrix,
    ) -> Self {
        #[derive(Serialize)]
        struct Payload<'a> {
            schema: &'static str,
            matrix: &'a methodus::CsrMatrix,
        }
        let bytes = serde_json::to_vec(&Payload {
            schema: "krasis-coupling-matrix/1",
            matrix: &matrix,
        })
        .expect("coupling matrix is serializable");
        let identity = format!("blake3:{}", blake3::hash(&bytes).to_hex());
        Self::new(row, column, argument, Arc::new(matrix), identity)
    }
}

#[derive(Clone)]
struct ResolvedEdge {
    row: usize,
    column: usize,
    argument: CouplingArgument,
    action: Arc<dyn LinearOperator>,
    identity: String,
}

/// One typed dependency of the coupling graph: the residual of leaf `row` depends on the
/// `argument` of leaf `column`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouplingDependency {
    pub row: usize,
    pub column: usize,
    pub argument: CouplingArgument,
}

/// SV7-F2's explicit coupling graph over the leaves of a [`CoupledSystemOperator`].
///
/// `stages` lists the strongly connected components in dependency-first order: every leaf in a
/// stage depends only on leaves in the same or an earlier stage. A graph whose stages are all
/// singletons is a DAG and admits a sequential schedule; a stage with several leaves is a
/// fixed-point block that a partitioned scheme must iterate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CouplingGraph {
    nodes: Vec<String>,
    dependencies: Vec<CouplingDependency>,
    stages: Vec<Vec<usize>>,
}

impl CouplingGraph {
    fn new(nodes: Vec<String>, edges: &[ResolvedEdge]) -> Self {
        let mut dependencies: Vec<CouplingDependency> = edges
            .iter()
            .map(|edge| CouplingDependency {
                row: edge.row,
                column: edge.column,
                argument: edge.argument,
            })
            .collect();
        dependencies
            .sort_by_key(|dependency| (dependency.row, dependency.column, dependency.argument));
        dependencies.dedup();
        let stages = strongly_connected_components(nodes.len(), &dependencies);
        Self {
            nodes,
            dependencies,
            stages,
        }
    }

    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    pub fn dependencies(&self) -> &[CouplingDependency] {
        &self.dependencies
    }

    pub fn stages(&self) -> &[Vec<usize>] {
        &self.stages
    }

    pub fn is_acyclic(&self) -> bool {
        self.stages.iter().all(|stage| stage.len() == 1)
    }
}

/// Tarjan's algorithm over arcs `row -> column` ("row depends on column"); components are
/// emitted after everything reachable from them, i.e. dependencies first, and each component
/// is listed in ascending leaf order so the result is canonical.
fn strongly_connected_components(
    node_count: usize,
    dependencies: &[CouplingDependency],
) -> Vec<Vec<usize>> {
    struct Tarjan {
        successors: Vec<Vec<usize>>,
        index: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        next_index: usize,
        components: Vec<Vec<usize>>,
    }

    impl Tarjan {
        fn visit(&mut self, node: usize) {
            self.index[node] = Some(self.next_index);
            self.lowlink[node] = self.next_index;
            self.next_index += 1;
            self.stack.push(node);
            self.on_stack[node] = true;
            for successor_index in 0..self.successors[node].len() {
                let successor = self.successors[node][successor_index];
                match self.index[successor] {
                    None => {
                        self.visit(successor);
                        self.lowlink[node] = self.lowlink[node].min(self.lowlink[successor]);
                    }
                    Some(index) if self.on_stack[successor] => {
                        self.lowlink[node] = self.lowlink[node].min(index);
                    }
                    Some(_) => {}
                }
            }
            if Some(self.lowlink[node]) == self.index[node] {
                let mut component = Vec::new();
                loop {
                    let member = self.stack.pop().expect("tarjan stack holds the root");
                    self.on_stack[member] = false;
                    component.push(member);
                    if member == node {
                        break;
                    }
                }
                component.sort_unstable();
                self.components.push(component);
            }
        }
    }

    let mut successors = vec![Vec::new(); node_count];
    for dependency in dependencies {
        successors[dependency.row].push(dependency.column);
    }
    let mut tarjan = Tarjan {
        successors,
        index: vec![None; node_count],
        lowlink: vec![0; node_count],
        on_stack: vec![false; node_count],
        stack: Vec::new(),
        next_index: 0,
        components: Vec::new(),
    };
    for node in 0..node_count {
        if tarjan.index[node].is_none() {
            tarjan.visit(node);
        }
    }
    tarjan.components
}

/// N leaves composed over one concatenated state, with typed cross-leaf coupling edges.
#[derive(Clone)]
pub struct CoupledSystemOperator {
    leaves: Vec<CoupledLeaf>,
    offsets: Vec<usize>,
    edges: Vec<ResolvedEdge>,
    layout: StateLayout,
    binding: StateBinding,
    block_layout: SolverBlockLayout,
    graph: CouplingGraph,
    consistent_initialization: Option<ConsistentInitialization>,
    identity: String,
}

impl fmt::Debug for CoupledSystemOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoupledSystemOperator")
            .field("leaves", &self.leaves)
            .field("graph", &self.graph)
            .field("identity", &self.identity)
            .finish()
    }
}

impl CoupledSystemOperator {
    /// Composes `leaves` in declaration order (the composed state is their concatenation) with
    /// `edges` resolved by leaf name. Refuses an empty or duplicate leaf name, a block id or
    /// semantic id shared by two leaves (system-level ids are dense across the whole system,
    /// which is why `SemanticId` mirrors `SysVarId` and not a per-model `SymbolId`), an edge
    /// naming an unknown leaf or the same leaf twice (within-plan coupling belongs to Finitum),
    /// and an edge whose action shape does not match its endpoints.
    pub fn new(leaves: Vec<CoupledLeaf>, edges: Vec<CouplingEdge>) -> Result<Self, KrasisError> {
        if leaves.is_empty() {
            return Err(KrasisError::EmptyLayout);
        }
        let mut names = BTreeSet::new();
        for leaf in &leaves {
            if !names.insert(leaf.name.as_str()) {
                return Err(KrasisError::InvalidCoupling(format!(
                    "leaf `{}` is declared more than once",
                    leaf.name
                )));
            }
        }

        let mut offsets = Vec::with_capacity(leaves.len());
        let mut blocks = Vec::new();
        let mut bindings = Vec::new();
        let mut offset = 0usize;
        for leaf in &leaves {
            offsets.push(offset);
            for block in leaf.layout.blocks() {
                let range = block.range();
                blocks.push(StateBlock::new(
                    block.id().clone(),
                    offset + range.start..offset + range.end,
                ));
                let semantic = leaf
                    .binding
                    .semantic_for(block.id())
                    .expect("leaf binding covers every leaf block");
                bindings.push((semantic, block.id().clone()));
            }
            offset += leaf.dimension();
        }
        let layout = StateLayout::new(blocks)?;
        let binding = StateBinding::new(&layout, bindings)?;

        let index_of = |name: &str| {
            leaves
                .iter()
                .position(|leaf| leaf.name == name)
                .ok_or_else(|| {
                    KrasisError::InvalidCoupling(format!(
                        "coupling edge names unknown leaf `{name}`"
                    ))
                })
        };
        let mut resolved = Vec::with_capacity(edges.len());
        for edge in edges {
            let row = index_of(&edge.row)?;
            let column = index_of(&edge.column)?;
            if row == column {
                return Err(KrasisError::InvalidCoupling(format!(
                    "coupling edge `{}` couples leaf `{}` to itself; within-plan coupling is \
                     realized by Finitum, not composed here",
                    edge.identity, edge.row
                )));
            }
            let (rows, columns) = (leaves[row].dimension(), leaves[column].dimension());
            if edge.action.rows() != rows || edge.action.columns() != columns {
                return Err(KrasisError::InvalidCoupling(format!(
                    "coupling edge `{}` from `{}` into `{}` is {}x{}, expected {rows}x{columns}",
                    edge.identity,
                    edge.column,
                    edge.row,
                    edge.action.rows(),
                    edge.action.columns()
                )));
            }
            resolved.push(ResolvedEdge {
                row,
                column,
                argument: edge.argument,
                action: edge.action,
                identity: edge.identity,
            });
        }
        // Canonical evaluation order, so the composed action and identity are order-independent.
        resolved.sort_by(|left, right| {
            (left.row, left.column, left.argument, &left.identity).cmp(&(
                right.row,
                right.column,
                right.argument,
                &right.identity,
            ))
        });

        let block_layout = SolverBlockLayout::new(
            leaves
                .iter()
                .map(|leaf| BlockSpec {
                    name: leaf.name.clone(),
                    length: leaf.dimension(),
                    residual_scale: 1.0,
                })
                .collect(),
        )
        .map_err(|error| KrasisError::InvalidCoupling(error.to_string()))?;
        let graph = CouplingGraph::new(
            leaves.iter().map(|leaf| leaf.name.clone()).collect(),
            &resolved,
        );
        let identity = coupled_system_identity(&leaves, &resolved);
        Ok(Self {
            leaves,
            offsets,
            edges: resolved,
            layout,
            binding,
            block_layout,
            graph,
            consistent_initialization: None,
            identity,
        })
    }

    /// Records the composed differential/algebraic mask (every leaf must carry row kinds) and
    /// the Newton policy for index-1 consistent initialization, folded into the identity;
    /// semantics as [`CoupledOperator::with_consistent_initialization`], evaluated over the
    /// composed residual so cross-leaf edges take part.
    pub fn with_consistent_initialization(
        mut self,
        newton: NewtonConfig,
    ) -> Result<Self, KrasisError> {
        let mut mask = Vec::with_capacity(self.dimension());
        for leaf in &self.leaves {
            let Some(rows) = leaf.row_kinds() else {
                return Err(KrasisError::InvalidCoupling(format!(
                    "leaf `{}` carries no differential/algebraic row kinds",
                    leaf.name
                )));
            };
            mask.extend_from_slice(rows);
        }
        self.identity = format!(
            "{}:consistent-init={}",
            self.identity,
            consistent_initialization_identity(&mask, &newton)
        );
        self.consistent_initialization = Some(ConsistentInitialization { mask, newton });
        Ok(self)
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn dimension(&self) -> usize {
        self.layout.width()
    }

    pub fn leaves(&self) -> &[CoupledLeaf] {
        &self.leaves
    }

    /// Composed state layout: every leaf's blocks, shifted, in leaf order.
    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Composed binding from system-level semantic ids to composed blocks.
    pub fn binding(&self) -> &StateBinding {
        &self.binding
    }

    /// Range of leaf `index` inside the composed state.
    pub fn leaf_range(&self, index: usize) -> Option<std::ops::Range<usize>> {
        let leaf = self.leaves.get(index)?;
        let start = self.offsets[index];
        Some(start..start + leaf.dimension())
    }

    pub fn graph(&self) -> &CouplingGraph {
        &self.graph
    }

    /// Solves the composed `F(time, state, ydot) = 0` for the differential rows' rate; see
    /// [`CoupledOperator::solve_consistent_state_rate`].
    pub fn solve_consistent_state_rate(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Vec<f64>, KrasisError> {
        let config = self.consistent_initialization.as_ref().ok_or_else(|| {
            KrasisError::InvalidCoupling(
                "operator has no differential/algebraic mask for consistent initialization".into(),
            )
        })?;
        solve_consistent_state_rate_for(self, &config.mask, &config.newton, context, time, state)
    }

    fn require_len(&self, label: &str, actual: usize) -> Result<(), NumericError> {
        require_len(label, actual, self.dimension())
    }

    fn apply_edges(
        &self,
        context: &EvaluationContext,
        state_argument: &[f64],
        rate_argument: &[f64],
        output: &mut [f64],
        label: &str,
    ) -> Result<(), NumericError> {
        let mut scratch = Vec::new();
        for edge in &self.edges {
            let source = match edge.argument {
                CouplingArgument::State => state_argument,
                CouplingArgument::Rate => rate_argument,
            };
            let column = self.leaf_range(edge.column).expect("resolved edge column");
            let row = self.leaf_range(edge.row).expect("resolved edge row");
            scratch.clear();
            scratch.resize(row.len(), 0.0);
            edge.action.apply(context, &source[column], &mut scratch)?;
            for (value, contribution) in output[row].iter_mut().zip(&scratch) {
                *value += contribution;
            }
        }
        require_finite(label, output)
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

fn coupled_system_identity(leaves: &[CoupledLeaf], edges: &[ResolvedEdge]) -> String {
    #[derive(Serialize)]
    struct LeafIdentity<'a> {
        name: &'a str,
        identity: &'a str,
        layout: &'a str,
        binding: &'a str,
        row_kinds: Option<&'a [RowKind]>,
    }
    #[derive(Serialize)]
    struct EdgeIdentity<'a> {
        row: usize,
        column: usize,
        argument: CouplingArgument,
        rows: usize,
        columns: usize,
        identity: &'a str,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        leaves: Vec<LeafIdentity<'a>>,
        edges: Vec<EdgeIdentity<'a>>,
    }

    let payload = Payload {
        schema: "krasis-coupled-system/1",
        leaves: leaves
            .iter()
            .map(|leaf| LeafIdentity {
                name: &leaf.name,
                identity: &leaf.identity,
                layout: leaf.layout.identity(),
                binding: leaf.binding.identity(),
                row_kinds: leaf.row_kinds(),
            })
            .collect(),
        edges: edges
            .iter()
            .map(|edge| EdgeIdentity {
                row: edge.row,
                column: edge.column,
                argument: edge.argument,
                rows: edge.action.rows(),
                columns: edge.action.columns(),
                identity: &edge.identity,
            })
            .collect(),
    };
    let bytes = serde_json::to_vec(&payload).expect("coupled system identity is serializable");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}

impl DaeOperator for CoupledSystemOperator {
    fn dimension(&self) -> usize {
        self.layout.width()
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.require_len("coupled system state", state.len())?;
        self.require_len("coupled system state rate", state_rate.len())?;
        self.require_len("coupled system residual", output.len())?;
        for (index, leaf) in self.leaves.iter().enumerate() {
            let range = self.leaf_range(index).expect("leaf index");
            leaf.operator.residual(
                context,
                time,
                &state[range.clone()],
                &state_rate[range.clone()],
                &mut output[range],
            )?;
        }
        self.apply_edges(
            context,
            state,
            state_rate,
            output,
            "coupled system residual",
        )
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
        self.require_len("coupled system state", state.len())?;
        self.require_len("coupled system state rate", state_rate.len())?;
        self.require_len("coupled system state direction", state_direction.len())?;
        self.require_len("coupled system rate direction", rate_direction.len())?;
        self.require_len("coupled system JVP", output.len())?;
        for (index, leaf) in self.leaves.iter().enumerate() {
            let range = self.leaf_range(index).expect("leaf index");
            leaf.operator.jacobian_vector_product(
                context,
                time,
                &state[range.clone()],
                &state_rate[range.clone()],
                &state_direction[range.clone()],
                &rate_direction[range.clone()],
                &mut output[range],
            )?;
        }
        self.apply_edges(
            context,
            state_direction,
            rate_direction,
            output,
            "coupled system JVP",
        )
    }

    fn make_initial_state_consistent(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &mut [f64],
    ) -> Result<(), NumericError> {
        if self.consistent_initialization.is_none() {
            return Ok(());
        }
        // Validates that a consistent composed state rate exists; never adjusts `state`.
        self.solve_consistent_state_rate(context, time, state)
            .map(|_| ())
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }

    fn event_count(&self) -> usize {
        self.leaves
            .iter()
            .map(|leaf| leaf.operator.event_count())
            .sum()
    }

    fn event_values(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.require_len("coupled system state", state.len())?;
        require_len("coupled system events", output.len(), self.event_count())?;
        let mut cursor = 0;
        for (index, leaf) in self.leaves.iter().enumerate() {
            let range = self.leaf_range(index).expect("leaf index");
            let count = leaf.operator.event_count();
            leaf.operator.event_values(
                context,
                time,
                &state[range],
                &mut output[cursor..cursor + count],
            )?;
            cursor += count;
        }
        Ok(())
    }
}

impl NonlinearOperator for CoupledSystemOperator {
    fn dimension(&self) -> usize {
        self.layout.width()
    }

    /// The steady view at `t = 0`, `ydot = 0` (the convention [`CoupledOperator`] uses).
    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.layout.width()];
        DaeOperator::residual(self, context, 0.0, state, &zero, output)
    }

    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.layout.width()];
        DaeOperator::jacobian_vector_product(
            self, context, 0.0, state, &zero, direction, &zero, output,
        )
    }
}

impl BlockNonlinearOperator for CoupledSystemOperator {
    /// One Methodus block per leaf: the partition `solve_blocks` iterates over is the
    /// realization-group partition, since within a leaf Finitum already solves monolithically.
    fn block_layout(&self) -> &SolverBlockLayout {
        &self.block_layout
    }
}

impl TransactionalOperator for CoupledSystemOperator {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn state_layout_identity(&self) -> &str {
        self.layout.identity()
    }
}
