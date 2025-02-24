use rand::rngs::ThreadRng;
use rand::Rng;
use raylib::prelude::*;
use std::{
    cell::Cell,
    f32::consts::PI,
    time::{Duration, Instant},
};

const OMEGA: f32 = 1e37;

#[derive(Clone)]
struct Particle {
    position: Cell<Vector2>,
    velocity: Cell<Vector2>,
    density: Cell<f32>,
}

struct Simulation {
    particles: Vec<Particle>,
    radius: f32,
    rest_density: f32,
    fluid_stiffness: f32,
    bounds: Vector2,
    gravity: f32,
    volume: f32,
}

impl Simulation {
    fn new_from_grid(
        number_of_particles: usize,
        starting_point: Vector2,
        number_of_columns: usize,
        spacing: f32,
        jitter: f32,
        radius: f32,
        rest_density: f32,
        fluid_stiffness: f32,
        bounds: Vector2,
        gravity: f32,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let particles = (0..number_of_particles)
            .map(|i| {
                let row = i / number_of_columns;
                let col = i % number_of_columns;
                let base_position =
                    starting_point + Vector2::new(col as f32 * spacing, row as f32 * spacing);
                let jittered_position = base_position
                    + Vector2::new(
                        rng.gen_range(-jitter..jitter),
                        rng.gen_range(-jitter..jitter),
                    );
                Particle {
                    position: Cell::new(jittered_position),
                    velocity: Cell::new(Vector2::zero()),
                    density: Cell::new(0.0),
                }
            })
            .collect();

        Self {
            particles,
            radius,
            rest_density,
            fluid_stiffness,
            bounds,
            gravity,
            volume: (PI * radius.powi(4)) * 0.1666666667,
        }
    }

    fn calculate_pressure_force(&self, particle_index: usize) -> Vector2 {
        let mut pressure_force = Vector2::zero();
        let sample_point = self.particles[particle_index].position.get();

        for (i, particle) in self.particles.iter().enumerate() {
            if particle_index == i {
                continue;
            }

            let position = particle.position.get();
            let distance = sample_point.distance_to(position);

            if distance >= self.radius {
                continue;
            }

            let direction_of_change = if distance != 0.0 {
                (position - sample_point) / distance
            } else {
                Vector2::new(1., 0.).rotated(ThreadRng::default().gen_range(0.0..(2.0 * PI)))
            };

            let slope = self.smoothing_kernel_derivative(distance);
            let shared_pressure = self.densities_to_shared_pressure(
                self.particles[particle_index].density.get(),
                particle.density.get(),
            );

            let density_i = particle.density.get().max(1.0 / OMEGA);
            pressure_force += direction_of_change * shared_pressure * slope / density_i;
        }

        pressure_force
    }

    fn update_densities(&self) {
        for (i, particle) in self.particles.iter().enumerate() {
            particle.density.set(self.calculate_density(i));
        }
    }

    fn calculate_density(&self, index: usize) -> f32 {
        let mut density = 0.0;
        let sample_position = self.particles[index].position.get();

        for (i, particle) in self.particles.iter().enumerate() {
            if i == index {
                continue;
            }

            let distance = particle.position.get().distance_to(sample_position);
            density += self.smoothing_kernel(distance);
        }

        density.max(1.0 / OMEGA)
    }

    fn density_to_pressure(&self, density: f32) -> f32 {
        self.fluid_stiffness * ((density / self.rest_density).powi(7) - 1.0)
    }

    fn densities_to_shared_pressure(&self, density_one: f32, density_two: f32) -> f32 {
        (self.density_to_pressure(density_one) + self.density_to_pressure(density_two)) * 0.5
    }

    fn smoothing_kernel(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.0;
        }
        (self.radius - distance).powi(2) / self.volume
    }

    fn smoothing_kernel_derivative(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.0;
        }
        (distance - self.radius) * 12.0 / (self.radius.powi(4) * PI)
    }

    fn apply_velocity(&self, damping_factor: f32, delta_time: f32) {
        for particle in &self.particles {
            let current_position = particle.position.get();
            let current_velocity = particle.velocity.get();

            let mut proposed_position = current_position + current_velocity * delta_time;
            let mut final_velocity = current_velocity;

            // X-axis boundary check
            if proposed_position.x < 0.0 || proposed_position.x > self.bounds.x {
                final_velocity.x *= -damping_factor;
                proposed_position.x = proposed_position.x.clamp(0.0, self.bounds.x);
            }

            // Y-axis boundary check
            if proposed_position.y < 0.0 || proposed_position.y > self.bounds.y {
                final_velocity.y *= -damping_factor;
                proposed_position.y = proposed_position.y.clamp(0.0, self.bounds.y);
            }

            particle.position.set(proposed_position);
            particle.velocity.set(final_velocity);
        }
    }

    fn simulate_step(&self, delta_time: f32) {
        self.update_densities();

        for (i, particle) in self.particles.iter().enumerate() {
            let pressure_force = self.calculate_pressure_force(i);
            let gravity_force = Vector2::new(0.0, self.gravity * delta_time);
            let total_force = pressure_force + gravity_force;
            let acceleration = total_force / particle.density.get().max(1.0 / OMEGA);

            let new_velocity = particle.velocity.get() + acceleration * delta_time;
            particle.velocity.set(new_velocity);
        }

        self.apply_velocity(0.8, delta_time);
    }
}

enum GameState {
    Configuring { selected_particles: usize },
    Running,
}

fn main() {
    const WIDTH: i32 = 1600;
    const HEIGHT: i32 = 800;

    let (mut rl, thread) = raylib::init()
        .size(WIDTH, HEIGHT)
        .title("Confined Fluid Sim")
        .build();

    let mut game_state = GameState::Configuring {
        selected_particles: 1225,
    };
    let mut sim: Option<Simulation> = None;

    rl.set_target_fps(60);
    let test_time = Duration::from_secs_f32(4000.0);
    let start_time = Instant::now();

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        match &mut game_state {
            GameState::Configuring { selected_particles } => {
                let mouse_pos = d.get_mouse_position();
                let mouse_pressed = d.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT);

                // Particle count slider
                let slider_bounds = Rectangle::new(100.0, 300.0, 400.0, 20.0);
                let slider_min = 100;
                let slider_max = 2500;

                if slider_bounds.check_collision_point_rec(mouse_pos) && mouse_pressed {
                    let t = ((mouse_pos.x - slider_bounds.x) / slider_bounds.width).clamp(0.0, 1.0);
                    *selected_particles =
                        (slider_min as f32 + t * (slider_max - slider_min) as f32) as usize;
                }

                // Draw slider
                d.draw_rectangle_rec(slider_bounds, Color::GRAY);
                let slider_value_width = ((*selected_particles - slider_min) as f32
                    / (slider_max - slider_min) as f32)
                    * slider_bounds.width;
                d.draw_rectangle(
                    slider_bounds.x as i32,
                    slider_bounds.y as i32,
                    slider_value_width as i32,
                    slider_bounds.height as i32,
                    Color::BLUE,
                );

                // Draw particle count
                d.draw_text(
                    &format!("Particles: {}", selected_particles),
                    100,
                    330,
                    20,
                    Color::WHITE,
                );

                // Start button
                let start_button = Rectangle::new(250.0, 400.0, 100.0, 50.0);
                d.draw_rectangle_rec(start_button, Color::GREEN);
                d.draw_text("Start", 260, 410, 30, Color::BLACK);

                if start_button.check_collision_point_rec(mouse_pos) && mouse_pressed {
                    let number_of_columns = (*selected_particles as f32).sqrt() as usize;
                    sim = Some(Simulation::new_from_grid(
                        *selected_particles,
                        Vector2::new(10.0, 10.0),
                        number_of_columns,
                        15.0,
                        0.5,
                        130.0,
                        0.004,
                        0.95,
                        Vector2::new(WIDTH as f32, HEIGHT as f32),
                        0.7,
                    ));
                    game_state = GameState::Running;
                }
            }
            GameState::Running => {
                if let Some(sim) = &sim {
                    for _ in 0..2 {
                        sim.simulate_step(1.0 / 60.0);
                    }

                    for particle in &sim.particles {
                        let pos = particle.position.get();
                        d.draw_circle(pos.x as i32, pos.y as i32, 6., Color::BLUE);
                    }
                }
            }
        }

        d.draw_fps(0, 0);

        if Instant::now().duration_since(start_time) > test_time {
            break;
        }
    }
}
