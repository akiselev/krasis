//! Initial-condition projection from Finitum field sources onto a P1 nodal DOF map.

use finitum::FieldSource;

use crate::{BlockId, FieldId, KrasisError, SimulationState, StateLayout};

/// Vertex coordinates for pointwise evaluation of a [`FieldSource`] onto a P1 nodal DOF map.
///
/// Coordinates are vertex-major and share one spatial dimension; the vertex count drives the
/// component width expected from each bound [`FieldSource`], not the raw DOF count (a
/// vector-valued block has several DOF-map entries per vertex).
#[derive(Clone, Debug)]
pub struct NodalContext<'a> {
    coordinates: &'a [Vec<f64>],
}

impl<'a> NodalContext<'a> {
    pub fn new(coordinates: &'a [Vec<f64>]) -> Result<Self, KrasisError> {
        let Some(first) = coordinates.first() else {
            return Err(KrasisError::EmptyNodalContext);
        };
        let dimension = first.len();
        for (index, point) in coordinates.iter().enumerate() {
            if point.len() != dimension {
                return Err(KrasisError::InconsistentNodalCoordinates {
                    index,
                    expected: dimension,
                    actual: point.len(),
                });
            }
            if let Some(component) = point.iter().position(|value| !value.is_finite()) {
                return Err(KrasisError::NonFiniteValue {
                    label: format!("nodal coordinate {index}"),
                    index: component,
                });
            }
        }
        Ok(Self { coordinates })
    }

    pub fn vertex_count(&self) -> usize {
        self.coordinates.len()
    }

    pub fn coordinates(&self) -> &'a [Vec<f64>] {
        self.coordinates
    }
}

/// Project each bound [`FieldSource`] onto its block's P1 nodal DOF range and assemble a
/// freshly committed [`SimulationState`].
///
/// `bindings` must name every block in `layout` exactly once. A block's component count is
/// derived from its width over `nodal`'s vertex count (`components = width / vertex_count`,
/// vertex-major); [`FieldSource::Constant`] and [`FieldSource::Sampled`] must each produce
/// `components` values, and [`FieldSource::Nodal`] must already carry `components` values per
/// vertex. Any `FieldSource` variant this function does not evaluate pointwise (for example a
/// future table- or kernel-backed source) is refused rather than silently mismatched.
pub fn initial_state_from(
    layout: &StateLayout,
    nodal: &NodalContext,
    history_limit: usize,
    bindings: &[(BlockId, FieldSource)],
) -> Result<SimulationState, KrasisError> {
    let mut resolved = std::collections::BTreeMap::new();
    for (block, source) in bindings {
        if layout.block(block).is_none() {
            return Err(KrasisError::InitialBlockUnknown(block.to_string()));
        }
        if resolved.insert(block.clone(), source).is_some() {
            return Err(KrasisError::InitialBlockDuplicate(block.to_string()));
        }
    }
    for block in layout.blocks() {
        if !resolved.contains_key(block.id()) {
            return Err(KrasisError::InitialBlockMissing(block.id().to_string()));
        }
    }

    let vertex_count = nodal.vertex_count();
    let mut state = SimulationState::new(layout.clone(), history_limit);
    for block in layout.blocks() {
        let width = block.range().len();
        if width % vertex_count != 0 {
            return Err(KrasisError::InitialDimensionMismatch {
                block: block.id().to_string(),
                width,
                vertex_count,
            });
        }
        let components = width / vertex_count;
        let source = resolved[block.id()];
        let values = evaluate_field_source(block.id(), source, nodal, components)?;
        state.insert_field(FieldId::new(block.id().as_str()), values)?;
    }
    Ok(state)
}

fn evaluate_field_source(
    block: &BlockId,
    source: &FieldSource,
    nodal: &NodalContext,
    components: usize,
) -> Result<Vec<f64>, KrasisError> {
    // `FieldSource` is not `#[non_exhaustive]`, so every current variant is matched by name;
    // the wildcard arm is unreachable today but keeps this function compiling (refusing rather
    // than mismatching) once Finitum adds a non-pointwise variant such as `Table` or `Kernel`.
    #[allow(unreachable_patterns)]
    match source {
        FieldSource::Constant(values) => {
            if values.len() != components {
                return Err(KrasisError::FieldLength {
                    field: block.to_string(),
                    actual: values.len(),
                    expected: components,
                });
            }
            let mut assembled = Vec::with_capacity(components * nodal.vertex_count());
            for _ in 0..nodal.vertex_count() {
                assembled.extend_from_slice(values);
            }
            Ok(assembled)
        }
        FieldSource::Nodal(values) => {
            let expected = components * nodal.vertex_count();
            if values.len() != expected {
                return Err(KrasisError::FieldLength {
                    field: block.to_string(),
                    actual: values.len(),
                    expected,
                });
            }
            Ok(values.clone())
        }
        FieldSource::Sampled(sampler) => {
            let mut assembled = Vec::with_capacity(components * nodal.vertex_count());
            for coordinates in nodal.coordinates() {
                let value = sampler(coordinates);
                if value.len() != components {
                    return Err(KrasisError::FieldLength {
                        field: block.to_string(),
                        actual: value.len(),
                        expected: components,
                    });
                }
                assembled.extend(value);
            }
            Ok(assembled)
        }
        // Non-exhaustive by design: a future `FieldSource` variant (for example table- or
        // kernel-backed) is refused here rather than silently mismatched.
        _ => Err(KrasisError::InitialSourceNotPointwise(block.to_string())),
    }
}
