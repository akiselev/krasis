//! W10 shared scaffolding: real `ConnectionRealizationPlan`-based two-leaf heat-heat conduction
//! fixtures composed through `ConnectedSystemOperator::new` (the constraint-eliminated matching
//! or `nested_p1` interface, not an edge-composed exchange), modelled on Finitum's
//! `tests/nested_p1_diagonal.rs` / `tests/connection_matching.rs` construction over the same
//! `two-material-conduction.res` fixtures (copied here; Krasis tests do not reach into sibling
//! test directories).
//!
//! Each fixture is the scientific system `TwoMaterials` (two `Conduction` instances `a`, `b`
//! joined at `x = 0.5`), realized over two simplex boxes, bound with closed-form constitutive
//! closures, reduced by essential (Dirichlet) values over each side's whole far exterior face, and
//! composed as leaves `a`, `b` of a `CoupledSystemOperator` (the second instance's semantic ids
//! re-keyed to system level, as `tests/coupled_finitum_system.rs` does) under the connection.

use std::collections::BTreeMap;

use finitum::{
    AffineConstraint, BlockLayout as FinitumBlockLayout, ConnectionRealizationPlan, ConstraintSet,
    DofId, Mesh, MeshProfile, PointEvaluation, ReducedSystemOperator, RegionTagId,
    SystemConstitutiveInput, SystemOperator, SystemQuadrature, SystemRealizationPlan, TaggedMesh,
    realize,
};
use krasis::{
    ConnectedSystemOperator, CoupledLeaf, CoupledSystemOperator, FieldId, SemanticId,
    SimulationState, StateBinding,
};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, IntegralOperatorFactorization, NoImports,
    OperatorSystem, Registries, SymbolId, SystemCompilation, SystemOperatorCompilation,
    TensorInputId, TensorInputRole, compile_system, compile_system_operator,
    resolve_module_closure,
};

pub const STEADY: &str = include_str!("two-material-conduction.res");
pub const TRANSIENT: &str = include_str!("two-material-transient-conduction.res");

/// System-level id offset for the second instance of the model (`SysVarId`s are dense across
/// every instance; the fixture stands in for system elaboration).
const SECOND_INSTANCE_OFFSET: u32 = 1000;

/// Which bounded Finitum interface realization joins the two leaves.
#[derive(Clone, Copy, Debug)]
pub enum Interface {
    /// Geometrically coincident traces (`subdivisions` per axis on both boxes).
    Matching { subdivisions: usize },
    /// Endpoint `a` coarse (`coarse` per axis), endpoint `b` a nested refinement (`fine`).
    NestedP1 { coarse: usize, fine: usize },
}

/// Coefficients of the storage-bearing `Conduction`: `k(T) = k0 * (1 + k1 * T)`, `rho`, `cp`.
#[derive(Clone, Copy, Debug)]
pub struct Storage {
    pub k0: f64,
    pub k1: f64,
    pub rho: f64,
    pub cp: f64,
}

pub struct ConnectedFixture {
    pub operator: ConnectedSystemOperator,
    /// The uneliminated two-leaf composition the operator was built over (no edges).
    pub inner: CoupledSystemOperator,
    pub reduced: [ReducedSystemOperator; 2],
    pub plan: ConnectionRealizationPlan,
    /// `plan.constraints([&reduced[0], &reduced[1]])`.
    pub pair: ConstraintSet,
    pub meshes: [TaggedMesh; 2],
    /// The essential (Dirichlet) vertices of each side -- its whole far exterior face -- in its
    /// own mesh's numbering.
    pub essential: [Vec<usize>; 2],
    pub dirichlet: [f64; 2],
    pub dimensions: [usize; 2],
}

impl ConnectedFixture {
    pub fn dimension(&self) -> usize {
        self.dimensions[0] + self.dimensions[1]
    }

    /// Whether any connection dependency is weighted (a `nested_p1` interpolation), the shape
    /// whose Krasis diagonal was `None` before W10.
    pub fn is_weighted(&self) -> bool {
        self.pair
            .constraints()
            .any(|c| c.dependencies.len() != 1 || c.dependencies[0].weight != 1.0)
    }

    /// Target (eliminated) rows in the composed numbering.
    pub fn targets(&self) -> Vec<usize> {
        self.pair.constraints().map(|c| c.target.0).collect()
    }

    /// Master rows (dependencies) in the composed numbering.
    pub fn masters(&self) -> Vec<usize> {
        let mut masters: Vec<usize> = self
            .pair
            .constraints()
            .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
            .collect();
        masters.sort_unstable();
        masters.dedup();
        masters
    }

    /// A committed state over the operator's layout holding `values` (composed numbering).
    pub fn committed_state(&self, values: &[f64]) -> SimulationState {
        assert_eq!(values.len(), self.dimension());
        let mut state = SimulationState::new(self.operator.layout().clone(), 4);
        for block in self.operator.layout().blocks() {
            state
                .insert_field(
                    FieldId::new(block.id().as_str()),
                    values[block.range()].to_vec(),
                )
                .unwrap();
        }
        state
    }

    pub fn zero_state(&self) -> SimulationState {
        self.committed_state(&vec![0.0; self.dimension()])
    }

    /// `field` sampled at every vertex of both meshes, in the composed numbering, with the
    /// essential rows holding their prescribed values (the connection-reduced coordinates of
    /// the target rows are whatever the sample gives; a spatially affine field makes them
    /// consistent with the interface for both `matching` and `nested_p1`).
    pub fn sampled(&self, field: impl Fn(&[f64]) -> f64) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.dimension());
        for side in 0..2 {
            let mut local: Vec<f64> = self.meshes[side]
                .mesh
                .vertices()
                .iter()
                .map(|p| field(p))
                .collect();
            for &vertex in &self.essential[side] {
                local[vertex] = self.dirichlet[side];
            }
            values.extend(local);
        }
        values
    }
}

/// A spatially affine field, so P1 interpolation reproduces it exactly at every vertex.
pub fn affine(p: &[f64], base: f64, gx: f64, gy: f64, gz: f64) -> f64 {
    base + gx * p[0] + gy * p[1] + gz * p[2]
}

pub fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f64::max)
}

fn compile(source: &str) -> (SystemCompilation, SystemOperatorCompilation) {
    let closure = resolve_module_closure(source, &NoImports).unwrap();
    let compiled = compile_system(
        &closure,
        Registries::new(
            &UnitRegistry::si_bootstrap(),
            &QuantityKindRegistry::si_bootstrap(),
        ),
        "TwoMaterials",
    )
    .unwrap();
    let operators = compile_system_operator(&compiled).unwrap();
    (compiled, operators)
}

/// The identifier of `symbol`'s active basis input (of the given derivative) in `integral`.
fn active_input(
    integral: &IntegralOperatorFactorization,
    symbol: SymbolId,
    derivative: DerivativeEvaluation,
) -> TensorInputId {
    integral
        .primal
        .inputs
        .iter()
        .find(|input| {
            input.source == InputSourceRequirement::Basis
                && input.role == TensorInputRole::Active
                && input.binding.symbol == symbol
                && input.binding.evaluation.derivative == derivative
        })
        .map(|input| input.id)
        .unwrap_or_else(|| panic!("no {derivative:?} active input for {symbol}"))
}

/// Side `side`'s box (`[0, 0.5]` or `[0.5, 1]` in `x`, unit in `y`, `z`), `n` cells per axis.
fn box_mesh(side: usize, n: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 3,
        extent: vec![
            [side as f64 * 0.5, (side + 1) as f64 * 0.5],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        subdivisions: vec![n; 3],
    })
    .unwrap()
}

/// The single non-basis input of the steady `Conduction`'s one equation is the whole flux
/// `q = -k * grad(T)` (one opaque vector constitutive): a constant `k` makes the interior
/// Jacobian exact and state-independent, linear in `grad(T)` alone.
fn constant_flux(model: &OperatorSystem, conductivity: f64) -> Vec<SystemConstitutiveInput> {
    let t_symbol = model.field_order[0];
    let mut built = Vec::new();
    for block in &model.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let grad_t = active_input(integral, t_symbol, DerivativeEvaluation::Gradient);
                built.push(
                    SystemConstitutiveInput::try_new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        3,
                        "krasis-w10/q",
                        move |point: &PointEvaluation| {
                            Ok(point
                                .input_values(grad_t)
                                .unwrap()
                                .iter()
                                .map(|g| -conductivity * g)
                                .collect())
                        },
                        move |_point: &PointEvaluation, direction: &PointEvaluation| {
                            Ok(direction
                                .input_values(grad_t)
                                .unwrap()
                                .iter()
                                .map(|dg| -conductivity * dg)
                                .collect())
                        },
                    )
                    .unwrap(),
                );
            }
        }
    }
    built
}

/// Every non-basis input of the storage-bearing `Conduction`, bound by symbol name: the whole
/// flux `q = -k(T) * grad(T)` with `k(T) = k0 * (1 + k1 * T)` and the constant storage
/// coefficients `rho`, `cp` of `rho * cp * dt(T)`.
fn storage_closures(
    model: &OperatorSystem,
    names: &BTreeMap<SymbolId, String>,
    storage: Storage,
) -> Vec<SystemConstitutiveInput> {
    let t_symbol = model.field_order[0];
    let mut built = Vec::new();
    for block in &model.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let equation = block.equation.clone();
                let index = integral.integral_index;
                let name = names[&input.binding.symbol].as_str();
                let bound = match name {
                    "q" => {
                        let grad_t =
                            active_input(integral, t_symbol, DerivativeEvaluation::Gradient);
                        let t = active_input(integral, t_symbol, DerivativeEvaluation::Value);
                        let Storage { k0, k1, .. } = storage;
                        SystemConstitutiveInput::try_new(
                            equation,
                            index,
                            input.id,
                            3,
                            "krasis-w10/q(T)",
                            move |point: &PointEvaluation| {
                                let k = k0 * (1.0 + k1 * point.input_values(t).unwrap()[0]);
                                Ok(point
                                    .input_values(grad_t)
                                    .unwrap()
                                    .iter()
                                    .map(|g| -k * g)
                                    .collect())
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                let k = k0 * (1.0 + k1 * point.input_values(t).unwrap()[0]);
                                let dk = k0 * k1 * direction.input_values(t).unwrap()[0];
                                Ok(point
                                    .input_values(grad_t)
                                    .unwrap()
                                    .iter()
                                    .zip(direction.input_values(grad_t).unwrap())
                                    .map(|(g, dg)| -dk * g - k * dg)
                                    .collect())
                            },
                        )
                    }
                    "rho" | "cp" => {
                        let value = if name == "rho" {
                            storage.rho
                        } else {
                            storage.cp
                        };
                        SystemConstitutiveInput::try_new(
                            equation,
                            index,
                            input.id,
                            1,
                            format!("krasis-w10/{name}"),
                            move |_point: &PointEvaluation| Ok(vec![value]),
                            move |_point: &PointEvaluation, _direction: &PointEvaluation| {
                                Ok(vec![0.0])
                            },
                        )
                    }
                    other => panic!("unexpected non-basis input {other}"),
                };
                built.push(bound.unwrap());
            }
        }
    }
    built
}

/// Binds `model` over `mesh` under `plan` with the given closures (scalar nodal P1 layout).
fn bind(
    model: OperatorSystem,
    mesh: &Mesh,
    plan: &ConnectionRealizationPlan,
    built: Vec<SystemConstitutiveInput>,
) -> SystemOperator {
    let field = model.field_order[0];
    let layout = FinitumBlockLayout::new(vec![(field, mesh.vertices().len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::with_connections(
        model,
        mesh.clone(),
        layout,
        SystemQuadrature::Richest,
        std::slice::from_ref(plan),
    )
    .unwrap();
    plan.bind_kernels(built, BTreeMap::new()).unwrap()
}

/// Every vertex of `mesh` on the plane `x == plane` (the exterior face away from the interface
/// at `x == 0.5`, so never a trace vertex).
fn vertices_on(mesh: &Mesh, plane: f64) -> Vec<usize> {
    let vertices: Vec<usize> = mesh
        .vertices()
        .iter()
        .enumerate()
        .filter(|(_, p)| (p[0] - plane).abs() <= 1e-12)
        .map(|(index, _)| index)
        .collect();
    assert!(!vertices.is_empty());
    vertices
}

/// Essential (Dirichlet) constraints holding `value` at every `vertex`, nothing else.
fn essential(dimension: usize, vertices: &[usize], value: f64) -> ConstraintSet {
    ConstraintSet::new(
        dimension,
        vertices.iter().map(|&vertex| AffineConstraint {
            target: DofId(vertex),
            dependencies: Vec::new(),
            offset: value,
        }),
    )
    .unwrap()
}

fn build(
    source: &str,
    interface: Interface,
    dirichlet: [f64; 2],
    closures: impl Fn(
        usize,
        &OperatorSystem,
        &BTreeMap<SymbolId, String>,
    ) -> Vec<SystemConstitutiveInput>,
) -> ConnectedFixture {
    let (compiled, operators) = compile(source);
    let system = &compiled.system;
    let relation = &system.connections[0];
    let meshes = match interface {
        Interface::Matching { subdivisions } => {
            [box_mesh(0, subdivisions), box_mesh(1, subdivisions)]
        }
        Interface::NestedP1 { coarse, fine } => [box_mesh(0, coarse), box_mesh(1, fine)],
    };
    let facets_a = &meshes[0].tags.facet_regions[&RegionTagId::new("x_max")];
    let facets_b = &meshes[1].tags.facet_regions[&RegionTagId::new("x_min")];
    let plan = match interface {
        Interface::Matching { .. } => ConnectionRealizationPlan::matching(
            system,
            relation,
            [&meshes[0].mesh, &meshes[1].mesh],
            [facets_a, facets_b],
            1e-12,
        ),
        Interface::NestedP1 { .. } => ConnectionRealizationPlan::nested_p1(
            system,
            relation,
            [&meshes[0].mesh, &meshes[1].mesh],
            [facets_a, facets_b],
            1e-12,
        ),
    }
    .unwrap();
    let names = |side: usize| -> BTreeMap<SymbolId, String> {
        compiled
            .model(operators.model_systems[side].0)
            .symbols
            .iter()
            .map(|symbol| (symbol.id, symbol.name.clone()))
            .collect()
    };
    let models = [
        operators.model_systems[0].1.clone(),
        operators.model_systems[1].1.clone(),
    ];
    let system_operators = [0, 1].map(|side| {
        bind(
            models[side].clone(),
            &meshes[side].mesh,
            &plan,
            closures(side, &models[side], &names(side)),
        )
    });
    let dimensions = [
        system_operators[0].dimension(),
        system_operators[1].dimension(),
    ];
    let essential_vertices = [
        vertices_on(&meshes[0].mesh, 0.0),
        vertices_on(&meshes[1].mesh, 1.0),
    ];
    let reduced = [0, 1].map(|side| {
        system_operators[side]
            .reduced(essential(
                dimensions[side],
                &essential_vertices[side],
                dirichlet[side],
            ))
            .unwrap()
    });
    let pair = plan.constraints([&reduced[0], &reduced[1]]).unwrap();

    let leaf_a = CoupledLeaf::reduced_system("a", reduced[0].clone()).unwrap();
    let leaf_b = CoupledLeaf::reduced_system("b", reduced[1].clone()).unwrap();
    // The second instance of the model repeats `T`'s per-model id: rebind it to its
    // system-level id.
    let rekeyed = StateBinding::new(
        leaf_b.layout(),
        leaf_b
            .layout()
            .blocks()
            .iter()
            .map(|block| {
                let semantic = leaf_b.binding().semantic_for(block.id()).unwrap();
                (
                    SemanticId::new(semantic.as_u32() + SECOND_INSTANCE_OFFSET),
                    block.id().clone(),
                )
            })
            .collect(),
    )
    .unwrap();
    let leaf_b = leaf_b.with_binding(rekeyed).unwrap();
    let inner = CoupledSystemOperator::new(vec![leaf_a, leaf_b], Vec::new()).unwrap();
    let operator =
        ConnectedSystemOperator::new(inner.clone(), &plan, [&reduced[0], &reduced[1]]).unwrap();
    ConnectedFixture {
        operator,
        inner,
        reduced,
        plan,
        pair,
        meshes,
        essential: essential_vertices,
        dirichlet,
        dimensions,
    }
}

/// The steady `Conduction` pair with constant conductivities `[k_a, k_b]` and the Dirichlet
/// values `[T_a, T_b]` on the far faces.
pub fn steady_fixture(
    interface: Interface,
    conductivity: [f64; 2],
    dirichlet: [f64; 2],
) -> ConnectedFixture {
    build(STEADY, interface, dirichlet, |side, model, _names| {
        constant_flux(model, conductivity[side])
    })
}

/// The storage-bearing, state-dependent-conductivity `Conduction` pair.
pub fn transient_fixture(
    interface: Interface,
    storage: [Storage; 2],
    dirichlet: [f64; 2],
) -> ConnectedFixture {
    build(TRANSIENT, interface, dirichlet, |side, model, names| {
        storage_closures(model, names, storage[side])
    })
}
