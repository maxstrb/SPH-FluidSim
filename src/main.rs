use raylib::prelude::*;

/// Enum representing the type of a grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CellType {
    Fluid,
    Air,
    Solid,
}

/// Struct for storing information about a single grid cell.
#[derive(Debug)]
struct GridCell {
    velocity: Vector2, // Velocity vector (x, y)
    previous_velocity: Vector2,
    d_velocity: Vector2,
    pressure: f32,       // Pressure at this cell
    cell_type: CellType, // Type of this cell (fluid, air, or solid)
    particle_density: f32,
}

impl GridCell {
    fn new(cell_type: CellType) -> Self {
        Self {
            velocity: Vector2::new(0.0, 0.0),
            previous_velocity: Vector2::new(0.0, 0.0),
            d_velocity: Vector2::new(0.0, 0.0),
            pressure: 0.0,
            cell_type,
            particle_density: 0.,
        }
    }
}

/// Struct representing the grid used in the simulation.
#[derive(Debug)]
struct FluidGrid {
    overall_density: f32, // Fluid density
    cell_size: f32,       // Size of each cell
    inverse_cell_size: f32,
    half_cell_size: f32,
    num_cells_x: usize,   // Number of cells along the x-axis
    num_cells_y: usize,   // Number of cells along the y-axis
    cells: Vec<GridCell>, // Collection of grid cells
}

impl FluidGrid {
    fn new(density: f32, width: f32, height: f32, cell_size: f32) -> Self {
        let num_cells_x = (width / cell_size).ceil() as usize + 1;
        let num_cells_y = (height / cell_size).ceil() as usize + 1;

        // Initialize all cells as air cells
        let cells = (0..num_cells_x * num_cells_y)
            .map(|_| GridCell::new(CellType::Air))
            .collect();

        Self {
            overall_density: density,
            cell_size,
            half_cell_size: cell_size * 0.5,
            inverse_cell_size: 1. / cell_size,
            num_cells_x,
            num_cells_y,
            cells,
        }
    }

    fn store_previous_velocities(&mut self) {
        for cell in &mut self.cells {
            cell.previous_velocity = cell.velocity.clone();
        }
    }

    /// Get a mutable reference to a grid cell by its (x, y) index.
    fn cell_at(&mut self, x: usize, y: usize) -> Option<&mut GridCell> {
        if x < self.num_cells_x && y < self.num_cells_y {
            Some(&mut self.cells[x * self.num_cells_y + y])
        } else {
            None
        }
    }
}

/// Struct for storing information about a single particle.
#[derive(Debug)]
struct Particle {
    position: Vector2, // Position of the particle
    velocity: Vector2, // Velocity of the particle
    color: Color,      // Color for visualization
}

impl Particle {
    fn new(position: Vector2, velocity: Vector2, color: Color) -> Self {
        Self {
            position,
            velocity,
            color,
        }
    }

    fn apply_forces(&mut self, dt: f32, gravity: f32) {
        self.velocity.y = dt * gravity;
        self.position += self.velocity * dt;
    }
}

/// Struct representing the particle grid used in the simulation.
#[derive(Debug)]
struct ParticleGrid {
    max_particles: usize,     // Maximum number of particles
    radius: f32,              // Radius of each particle
    particles: Vec<Particle>, // Collection of particles
    particle_rest_density: f32,
}

impl ParticleGrid {
    fn new(radius: f32, max_particles: usize) -> Self {
        Self {
            max_particles,
            radius,
            particles: Vec::with_capacity(max_particles),
            particle_rest_density: 0.,
        }
    }

    /// Add a new particle to the grid.
    fn add_particle(&mut self, position: Vector2, velocity: Vector2, color: Color) {
        if self.particles.len() < self.max_particles {
            self.particles
                .push(Particle::new(position, velocity, color));
        }
    }

    fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

/// Main struct integrating the grid and particle system.
#[derive(Debug)]
struct FlipFluid {
    fluid_grid: FluidGrid,
    particle_grid: ParticleGrid,
    gravity: f32,
}

impl FlipFluid {
    fn new(
        density: f32,
        width: f32,
        height: f32,
        cell_size: f32,
        particle_radius: f32,
        max_particles: usize,
        gravity: f32,
    ) -> Self {
        let fluid_grid = FluidGrid::new(density, width, height, cell_size);
        let particle_grid = ParticleGrid::new(particle_radius, max_particles);
        let mut flip_fluid = Self {
            fluid_grid,
            particle_grid,
            gravity,
        };

        // Initialize particles in a dam break configuration
        // Fill a rectangular region on the left side of the domain
        let dam_width = width * 0.8; // Use 40% of width for initial fluid column
        let dam_height = height * 0.8; // Use 80% of height for fluid column
        let spacing = particle_radius * 4.0; // Space particles 2 radii apart

        let start_x = spacing;
        let start_y = spacing;

        // Calculate number of particles in each dimension
        let particles_x = (dam_width / spacing) as i32;
        let particles_y = (dam_height / spacing) as i32;

        // Initial velocity of particles (starting from rest)
        let initial_velocity = Vector2::new(0.0, 0.0);
        // Default particle color
        let particle_color = Color::new(255, 255, 255, 255); // Blue color

        // Create a uniform grid of particles
        'creating: for i in 0..particles_x {
            for j in 0..particles_y {
                let x = start_x + (i as f32 * spacing);
                let y = start_y + (j as f32 * spacing);
                let position = Vector2::new(x, y);

                // Only add particle if we haven't exceeded max_particles
                if flip_fluid.particle_grid.particle_count() < max_particles {
                    flip_fluid.add_particle(position, initial_velocity, particle_color);
                } else {
                    break 'creating;
                }
            }
        }

        flip_fluid
    }

    /// Add a particle to the simulation.
    fn add_particle(&mut self, position: Vector2, velocity: Vector2, color: Color) {
        self.particle_grid.add_particle(position, velocity, color);
    }

    fn update_particle_density(&mut self) {
        // Reset particle density for all non-solid cells
        for cell in &mut self.fluid_grid.cells {
            if cell.cell_type != CellType::Solid {
                cell.particle_density = 0.;
            }
        }

        for particle in &mut self.particle_grid.particles {
            // Clamp particle positions to grid boundaries
            particle.position.x = particle.position.x.clamp(
                self.fluid_grid.cell_size,
                self.fluid_grid.cell_size * (self.fluid_grid.num_cells_x as f32 - 1.),
            );
            particle.position.y = particle.position.y.clamp(
                self.fluid_grid.cell_size,
                self.fluid_grid.cell_size * (self.fluid_grid.num_cells_y as f32 - 1.),
            );

            // Calculate grid cell indices and interpolation weights
            let left_cell_x = ((particle.position.x - self.fluid_grid.half_cell_size)
                * self.fluid_grid.inverse_cell_size)
                .floor() as usize;
            let interpolated_weight_x = ((particle.position.x - self.fluid_grid.half_cell_size)
                - left_cell_x as f32 * self.fluid_grid.cell_size)
                * self.fluid_grid.inverse_cell_size;
            let right_cell_x = (left_cell_x + 1).min(self.fluid_grid.num_cells_x - 2);

            let bottom_cell_y = ((particle.position.y - self.fluid_grid.half_cell_size)
                * self.fluid_grid.inverse_cell_size)
                .floor() as usize;
            let interpolated_weight_y = ((particle.position.y - self.fluid_grid.half_cell_size)
                - bottom_cell_y as f32 * self.fluid_grid.cell_size)
                * self.fluid_grid.inverse_cell_size;
            let top_cell_y = (bottom_cell_y + 1).min(self.fluid_grid.num_cells_y - 2);

            let inverse_weight_x = 1. - interpolated_weight_x;
            let inverse_weight_y = 1. - interpolated_weight_y;

            // Update particle density for each surrounding cell if it's not solid
            if left_cell_x < self.fluid_grid.num_cells_x
                && bottom_cell_y < self.fluid_grid.num_cells_y
            {
                let cell = self.fluid_grid.cell_at(left_cell_x, bottom_cell_y).unwrap();
                if cell.cell_type != CellType::Solid {
                    cell.particle_density += inverse_weight_x * inverse_weight_y;
                }
            }

            if right_cell_x < self.fluid_grid.num_cells_x
                && bottom_cell_y < self.fluid_grid.num_cells_y
            {
                let cell = self
                    .fluid_grid
                    .cell_at(right_cell_x, bottom_cell_y)
                    .unwrap();
                if cell.cell_type != CellType::Solid {
                    cell.particle_density += interpolated_weight_x * inverse_weight_y;
                }
            }

            if right_cell_x < self.fluid_grid.num_cells_x
                && top_cell_y < self.fluid_grid.num_cells_y
            {
                let cell = self.fluid_grid.cell_at(right_cell_x, top_cell_y).unwrap();
                if cell.cell_type != CellType::Solid {
                    cell.particle_density += interpolated_weight_x * interpolated_weight_y;
                }
            }

            if left_cell_x < self.fluid_grid.num_cells_x && top_cell_y < self.fluid_grid.num_cells_y
            {
                let cell = self.fluid_grid.cell_at(left_cell_x, top_cell_y).unwrap();
                if cell.cell_type != CellType::Solid {
                    cell.particle_density += inverse_weight_x * interpolated_weight_y;
                }
            }
        }

        // Calculate particle rest density only considering fluid cells
        if self.particle_grid.particle_rest_density == 0. {
            let mut total_fluid_density: f32 = 0.;
            let mut fluid_cell_count: usize = 0;

            for cell in &mut self.fluid_grid.cells {
                if cell.cell_type == CellType::Fluid {
                    total_fluid_density += cell.particle_density;
                    fluid_cell_count += 1;
                }
            }

            if fluid_cell_count > 0 {
                self.particle_grid.particle_rest_density =
                    total_fluid_density / fluid_cell_count as f32;
            }
        }
    }

    fn handle_particle_collisions(&mut self) {
        let grid_cell_size = self.fluid_grid.cell_size;
        let particle_radius = self.particle_grid.radius;

        // Define boundaries
        let min_x = grid_cell_size + particle_radius;
        let max_x = (self.fluid_grid.num_cells_x - 1) as f32 * grid_cell_size - particle_radius;
        let min_y = grid_cell_size + particle_radius;
        let max_y = (self.fluid_grid.num_cells_y - 1) as f32 * grid_cell_size - particle_radius;

        // Iterate through all particles
        for particle in &mut self.particle_grid.particles {
            let mut pos = particle.position;

            // Handle wall collisions (boundary conditions)
            if pos.x < min_x {
                pos.x = min_x;
                particle.velocity.x = 0.0;
            }
            if pos.x > max_x {
                pos.x = max_x;
                particle.velocity.x = 0.0;
            }
            if pos.y < min_y {
                pos.y = min_y;
                particle.velocity.y = 0.0;
            }
            if pos.y > max_y {
                pos.y = max_y;
                particle.velocity.y = 0.0;
            }

            // Calculate grid cell indices
            let grid_x = (pos.x * self.fluid_grid.inverse_cell_size) as usize;
            let grid_y = (pos.y * self.fluid_grid.inverse_cell_size) as usize;

            // Handle collisions with solid cells
            if let Some(cell) = self.fluid_grid.cell_at(grid_x, grid_y) {
                if cell.cell_type == CellType::Solid {
                    // Calculate solid cell center position
                    let solid_cell_x = grid_x as f32 * grid_cell_size;
                    let solid_cell_y = grid_y as f32 * grid_cell_size;

                    // Calculate vector from solid cell to particle
                    let dx = pos.x - solid_cell_x;
                    let dy = pos.y - solid_cell_y;
                    let distance = (dx * dx + dy * dy).sqrt();

                    // If particle is inside or too close to solid cell
                    if distance < particle_radius {
                        let overlap = particle_radius - distance;

                        // Calculate normal vector
                        let nx = dx / distance;
                        let ny = dy / distance;

                        // Move particle outside the solid cell
                        pos.x += nx * overlap;
                        pos.y += ny * overlap;

                        // Reflect particle's velocity off the solid cell boundary
                        let dot_product = particle.velocity.x * nx + particle.velocity.y * ny;

                        particle.velocity.x -= 2.0 * dot_product * nx;
                        particle.velocity.y -= 2.0 * dot_product * ny;
                    }
                }
            }

            // Update particle position
            particle.position = pos;
        }
    }

    fn apply_forces_to_particles(&mut self, dt: f32) {
        for particle in &mut self.particle_grid.particles {
            particle.apply_forces(dt, self.gravity);
        }
    }

    fn push_particles_apart(&mut self, num_iterations: usize) {
        let inv_spacing = self.fluid_grid.inverse_cell_size;

        // Create grid for spatial partitioning
        let grid_size_x = self.fluid_grid.num_cells_x;
        let grid_size_y = self.fluid_grid.num_cells_y;
        let total_cells = grid_size_x * grid_size_y;

        // Temporary vectors for spatial partitioning
        let mut particles_per_cell = vec![0; total_cells];
        let mut first_particle_in_cell = vec![0; total_cells + 1]; // +1 for guard
        let mut cell_particle_ids = vec![0; self.particle_grid.particles.len()];

        // Count particles per cell
        for particle in &mut self.particle_grid.particles.iter() {
            let xi = (particle.position.x * inv_spacing).floor() as usize;
            let yi = (particle.position.y * inv_spacing).floor() as usize;
            let xi = xi.clamp(0, grid_size_x - 1);
            let yi = yi.clamp(0, grid_size_y - 1);
            let cell_nr = xi * grid_size_y + yi;
            particles_per_cell[cell_nr] += 1;
        }

        // Calculate partial sums for cell starts
        let mut sum = 0;
        for i in 0..total_cells {
            sum += particles_per_cell[i];
            first_particle_in_cell[i] = sum;
        }
        first_particle_in_cell[total_cells] = sum; // guard

        // Fill particle IDs into cells
        let mut temp_first_particle = first_particle_in_cell.clone();
        for (i, particle) in self.particle_grid.particles.iter().enumerate() {
            let xi = (particle.position.x * inv_spacing).floor() as usize;
            let yi = (particle.position.y * inv_spacing).floor() as usize;
            let xi = xi.clamp(0, grid_size_x - 1);
            let yi = yi.clamp(0, grid_size_y - 1);
            let cell_nr = xi * grid_size_y + yi;
            temp_first_particle[cell_nr] -= 1;
            cell_particle_ids[temp_first_particle[cell_nr]] = i;
        }

        // Push particles apart
        let min_dist = 2.0 * self.particle_grid.radius;
        let min_dist2 = min_dist * min_dist;

        for _ in 0..num_iterations {
            for i in 0..self.particle_grid.particles.len() {
                let pos = self.particle_grid.particles[i].position;
                let pxi = (pos.x * inv_spacing).floor() as isize;
                let pyi = (pos.y * inv_spacing).floor() as isize;

                // Check neighboring cells
                let x0 = (pxi - 1).max(0) as usize;
                let y0 = (pyi - 1).max(0) as usize;
                let x1 = (pxi + 1).min(grid_size_x as isize - 1) as usize;
                let y1 = (pyi + 1).min(grid_size_y as isize - 1) as usize;

                for xi in x0..=x1 {
                    for yi in y0..=y1 {
                        let cell_nr = xi * grid_size_y + yi;
                        let first = first_particle_in_cell[cell_nr];
                        let last = first_particle_in_cell[cell_nr + 1];

                        for j in first..last {
                            let id = cell_particle_ids[j];
                            if id == i {
                                continue;
                            }

                            let other_pos = self.particle_grid.particles[id].position;
                            let diff = other_pos - pos;
                            let d2 = diff.x * diff.x + diff.y * diff.y;

                            if d2 > min_dist2 || d2 == 0.0 {
                                continue;
                            }

                            let d = d2.sqrt();
                            let s = 0.5 * (min_dist - d) / d;
                            let push = diff * s;

                            // Update positions
                            self.particle_grid.particles[i].position -= push;
                            self.particle_grid.particles[id].position += push;
                        }
                    }
                }
            }
        }
    }

    fn transfer_velocities(&mut self, to_grid: bool, flip_ratio: f32) {
        if to_grid {
            // Store previous velocities and reset current ones
            self.fluid_grid.store_previous_velocities();

            // Reset all cells to air initially
            for cell in &mut self.fluid_grid.cells {
                cell.d_velocity = Vector2::zero();
                cell.velocity = Vector2::zero();
                cell.cell_type = if cell.pressure == 0.0 {
                    CellType::Solid
                } else {
                    CellType::Air
                };
            }

            // Mark fluid cells based on particle positions
            for particle in &self.particle_grid.particles {
                let xi = (particle.position.x * self.fluid_grid.inverse_cell_size).floor() as usize;
                let yi = (particle.position.y * self.fluid_grid.inverse_cell_size).floor() as usize;

                if let Some(cell) = self.fluid_grid.cell_at(xi, yi) {
                    if cell.cell_type == CellType::Air {
                        cell.cell_type = CellType::Fluid;
                    }
                }
            }
        }

        // Process x and y components separately
        for component in 0..2 {
            let (dx, dy) = if component == 0 {
                (0.0, self.fluid_grid.half_cell_size)
            } else {
                (self.fluid_grid.half_cell_size, 0.0)
            };

            for particle in self.particle_grid.particles.iter_mut() {
                let x = particle.position.x.clamp(
                    self.fluid_grid.cell_size,
                    (self.fluid_grid.num_cells_x - 1) as f32 * self.fluid_grid.cell_size,
                );
                let y = particle.position.y.clamp(
                    self.fluid_grid.cell_size,
                    (self.fluid_grid.num_cells_y - 1) as f32 * self.fluid_grid.cell_size,
                );

                let x0 = ((x - dx) * self.fluid_grid.inverse_cell_size)
                    .floor()
                    .min((self.fluid_grid.num_cells_x - 2) as f32)
                    as usize;
                let tx = ((x - dx) - x0 as f32 * self.fluid_grid.cell_size)
                    * self.fluid_grid.inverse_cell_size;
                let x1 = (x0 + 1).min(self.fluid_grid.num_cells_x - 2);

                let y0 = ((y - dy) * self.fluid_grid.inverse_cell_size)
                    .floor()
                    .min((self.fluid_grid.num_cells_y - 2) as f32)
                    as usize;
                let ty = ((y - dy) - y0 as f32 * self.fluid_grid.cell_size)
                    * self.fluid_grid.inverse_cell_size;
                let y1 = (y0 + 1).min(self.fluid_grid.num_cells_y - 2);

                let sx = 1.0 - tx;
                let sy = 1.0 - ty;

                let d0 = sx * sy;
                let d1 = tx * sy;
                let d2 = tx * ty;
                let d3 = sx * ty;

                if to_grid {
                    let pv = if component == 0 {
                        particle.velocity.x
                    } else {
                        particle.velocity.y
                    };

                    // Update velocities and weights for surrounding cells
                    for &(x, y, weight) in &[(x0, y0, d0), (x1, y0, d1), (x1, y1, d2), (x0, y1, d3)]
                    {
                        if let Some(cell) = self.fluid_grid.cell_at(x, y) {
                            if component == 0 {
                                cell.velocity.x += pv * weight;
                                cell.d_velocity.x += weight;
                            } else {
                                cell.velocity.y += pv * weight;
                                cell.d_velocity.y += weight;
                            }
                        }
                    }
                } else {
                    // Transfer velocities back to particles using FLIP/PIC blend
                    let mut total_weight = 0.0;
                    let mut pic_velocity = 0.0;
                    let mut velocity_correction = 0.0;

                    for &(x, y, weight) in &[(x0, y0, d0), (x1, y0, d1), (x1, y1, d2), (x0, y1, d3)]
                    {
                        if let Some(cell) = self.fluid_grid.cell_at(x, y) {
                            let is_valid = cell.cell_type != CellType::Air;
                            if is_valid {
                                let cell_vel = if component == 0 {
                                    cell.velocity.x
                                } else {
                                    cell.velocity.y
                                };
                                let prev_vel = if component == 0 {
                                    cell.previous_velocity.x
                                } else {
                                    cell.previous_velocity.y
                                };

                                total_weight += weight;
                                pic_velocity += weight * cell_vel;
                                velocity_correction += weight * (cell_vel - prev_vel);
                            }
                        }
                    }

                    if total_weight > 0.0 {
                        pic_velocity /= total_weight;
                        velocity_correction /= total_weight;

                        let current_vel = if component == 0 {
                            particle.velocity.x
                        } else {
                            particle.velocity.y
                        };

                        let flip_vel = current_vel + velocity_correction;
                        let new_vel = (1.0 - flip_ratio) * pic_velocity + flip_ratio * flip_vel;

                        if component == 0 {
                            particle.velocity.x = new_vel;
                        } else {
                            particle.velocity.y = new_vel;
                        }
                    }
                }
            }

            if to_grid {
                // Normalize velocities and restore solid cells
                for x in 0..self.fluid_grid.num_cells_x {
                    for y in 0..self.fluid_grid.num_cells_y {
                        if let Some(cell) = self.fluid_grid.cell_at(x, y) {
                            if cell.d_velocity.x > 0.0 {
                                cell.velocity.x /= cell.d_velocity.x;
                            }
                            if cell.d_velocity.y > 0.0 {
                                cell.velocity.y /= cell.d_velocity.y;
                            }

                            // Restore solid cell velocities
                            if cell.cell_type == CellType::Solid {
                                cell.velocity = cell.previous_velocity;
                            }
                        }
                    }
                }
            }
        }
    }

    fn solve_incompressibility(&mut self, num_iterations: usize, dt: f32, over_relaxation: f32) {
        let grid = &mut self.fluid_grid;

        // Reset pressures and store previous velocities
        for cell in &mut grid.cells {
            cell.pressure = 0.0;
            cell.previous_velocity = cell.velocity;
        }

        let pressure_coefficient = grid.overall_density * grid.cell_size / dt;
        let num_cells_x = grid.num_cells_x;
        let num_cells_y = grid.num_cells_y;

        // Main pressure solve iteration loop
        for _ in 0..num_iterations {
            // Iterate over internal cells (excluding boundaries)
            for i in 1..num_cells_x - 1 {
                for j in 1..num_cells_y - 1 {
                    let center_idx = i * num_cells_y + j;

                    // Skip non-fluid cells
                    if let Some(center_cell) = grid.cell_at(i, j) {
                        if center_cell.cell_type != CellType::Fluid {
                            continue;
                        }
                    }

                    // Get indices for neighboring cells
                    let left_idx = (i - 1) * num_cells_y + j;
                    let right_idx = (i + 1) * num_cells_y + j;
                    let bottom_idx = i * num_cells_y + (j - 1);
                    let top_idx = i * num_cells_y + (j + 1);

                    // Get solid boundaries weights for each direction
                    let solid_weight_left = grid.cells[left_idx].pressure;
                    let solid_weight_right = grid.cells[right_idx].pressure;
                    let solid_weight_bottom = grid.cells[bottom_idx].pressure;
                    let solid_weight_top = grid.cells[top_idx].pressure;

                    // Calculate total solid weight
                    let total_solid_weight = solid_weight_left
                        + solid_weight_right
                        + solid_weight_bottom
                        + solid_weight_top;

                    if total_solid_weight == 0.0 {
                        continue;
                    }

                    // Calculate divergence
                    let divergence = grid.cells[right_idx].velocity.x
                        - grid.cells[center_idx].velocity.x
                        + grid.cells[top_idx].velocity.y
                        - grid.cells[center_idx].velocity.y;

                    let mut adjusted_divergence = divergence;

                    // Apply drift compensation if enabled
                    if self.particle_grid.particle_rest_density > 0.0 {
                        let drift_coefficient = 1.0;
                        let density_compression = grid.cells[center_idx].particle_density
                            - self.particle_grid.particle_rest_density;

                        if density_compression > 0.0 {
                            adjusted_divergence -= drift_coefficient * density_compression;
                        }
                    }

                    // Calculate pressure adjustment
                    let pressure_change = -adjusted_divergence / total_solid_weight;
                    let relaxed_pressure = pressure_change * over_relaxation;
                    let scaled_pressure = pressure_coefficient * relaxed_pressure;

                    // Apply pressure adjustments to center cell
                    if let Some(center_cell) = grid.cell_at(i, j) {
                        center_cell.pressure += scaled_pressure;

                        // Update velocities based on pressure
                        center_cell.velocity.x -= solid_weight_left * relaxed_pressure;
                        center_cell.velocity.y -= solid_weight_bottom * relaxed_pressure;
                    }

                    // Apply pressure adjustments to neighboring cells
                    if let Some(right_cell) = grid.cell_at(i + 1, j) {
                        right_cell.velocity.x += solid_weight_right * relaxed_pressure;
                    }

                    if let Some(top_cell) = grid.cell_at(i, j + 1) {
                        top_cell.velocity.y += solid_weight_top * relaxed_pressure;
                    }
                }
            }
        }
    }

    fn simulate(&mut self, delta_time: f32) {
        let num_of_substeps: usize = 3;
        let num_of_pushes: usize = 4;
        let flip_ratio: f32 = 0.9;
        let pressure_incompresibility_iters = 100;
        let over_relaxation: f32 = 1.9;
        let corrected_delta_time = delta_time / num_of_substeps as f32;

        for _step in 0..num_of_substeps {
            self.apply_forces_to_particles(corrected_delta_time);
            self.push_particles_apart(num_of_pushes);
            self.handle_particle_collisions();
            self.transfer_velocities(true, flip_ratio);
            self.update_particle_density();
            self.solve_incompressibility(
                pressure_incompresibility_iters,
                corrected_delta_time,
                over_relaxation,
            );
            self.transfer_velocities(false, flip_ratio);
        }
    }
}

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(800, 600)
        .title("Shader Particles")
        .build();

    let fps = 120;
    rl.set_target_fps(fps);

    let mut fluid = FlipFluid::new(1000., 800., 600., 5., 2., 5000, 3000.);

    while !rl.window_should_close() {
        fluid.simulate(1. / fps as f32);

        // Begin drawing
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        // Begin shader mode and 2D mode
        {
            // Draw particles
            for particle in &fluid.particle_grid.particles {
                d.draw_circle_v(
                    particle.position,
                    fluid.particle_grid.radius,
                    particle.color,
                );
            }
        }

        println!("{}", d.get_frame_time());
    }
}
