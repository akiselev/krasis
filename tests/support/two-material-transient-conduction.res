module showcase.two_material_transient_conduction;
connector ThermalBoundary {
    equal temperature: ThermodynamicTemperature;
    balance heat_flux: HeatFlux conserves Energy;
}
model Conduction {
    domain body { dimension = 3; coordinates = cartesian; }
    region contact: boundary of body;
    field T: state scalar H1(order=1) on body {
        quantity = ThermodynamicTemperature; unit = K; time_role = differential;
    };
    provider density(T: ThermodynamicTemperature) -> Density { differentiability = symbolic; }
    provider specific_heat(T: ThermodynamicTemperature) -> SpecificHeat { differentiability = symbolic; }
    provider conductivity(T: ThermodynamicTemperature) -> ThermalConductivity { differentiability = symbolic; }
    provider fixed_temperature() -> ThermodynamicTemperature { differentiability = symbolic; }
    property rho = density(T);
    property cp = specific_heat(T);
    property k = conductivity(T);
    constitutive q = -k * grad(T);
    equation energy on body oriented by q { rho * cp * dt(T) + div(q) = 0; }
    boundary exterior on boundary("exterior") { dirichlet T = fixed_temperature(); }
    port surface: ThermalBoundary on contact from equation energy {
        temperature = trace(T);
        heat_flux = boundary_flux(energy);
    }
}
system TwoMaterials {
    domain left { dimension = 3; coordinates = cartesian; }
    domain right { dimension = 3; coordinates = cartesian; }
    interface joint between boundary(left), boundary(right);
    instance a: Conduction(body = left, contact = joint);
    instance b: Conduction(body = right, contact = joint);
    connect(a.surface, b.surface);
}
