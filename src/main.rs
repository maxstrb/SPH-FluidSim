use rand::rngs::ThreadRng;
use raylib::prelude::*;

use rand::Rng;
use std::{
    cell::Cell,
    f32::consts::PI,
    time::{Duration, Instant},
};

const OMEGA: f32 = 1e30;

struct Simulation {
    number_of_particles: usize,

    positions: Vec<Cell<Vector2>>,
    velocities: Vec<Cell<Vector2>>,

    densities: Vec<Cell<f32>>,

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

        let positions: Vec<Cell<Vector2>> = (0..number_of_particles)
            .map(|i| {
                let row = i / number_of_columns;
                let col = i % number_of_columns;

                let base_position =
                    starting_point + Vector2::new(col as f32 * spacing, row as f32 * spacing);

                let jitter_x = rng.gen_range(-jitter..jitter);
                let jitter_y = rng.gen_range(-jitter..jitter);
                let jittered_position = base_position + Vector2::new(jitter_x, jitter_y);

                Cell::new(jittered_position)
            })
            .collect();

        let velocities = vec![Cell::new(Vector2::zero()); number_of_particles];

        let densities = vec![Cell::new(0.); number_of_particles];

        Self {
            number_of_particles,

            positions,
            velocities,

            densities,

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
        let sample_point = self.positions[particle_index].get();

        for i in 0..self.number_of_particles {
            let position = self.positions[i].get();
            let distance = sample_point.distance_to(position);

            if distance >= self.radius || particle_index == i {
                continue;
            }

            let direction_of_change = if distance != 0. {
                (position - sample_point) / distance.max(1. / OMEGA)
            } else {
                Vector2::new(1., 0.).rotated(ThreadRng::default().gen_range(0.0..(2. * PI)))
            };

            let slope = self.smoothing_kernel_derivative(distance);
            let shared_pressure = self.densities_to_shared_pressure(
                self.densities[particle_index].get(),
                self.densities[i].get(),
            );

            let density_i = self.densities[i].get().max(1. / OMEGA);
            pressure_force += direction_of_change * shared_pressure * slope / density_i;
        }

        pressure_force
    }

    fn update_densities(&self) {
        for i in 0..self.number_of_particles {
            self.densities[i].set(self.calculate_density(i));
        }
    }

    fn calculate_density(&self, index: usize) -> f32 {
        let mut density = 0.;

        for (current_index, position) in self.positions.iter().enumerate() {
            let distance = position.get().distance_to(self.positions[index].get());
            if current_index == index {
                continue;
            }

            let influence = self.smoothing_kernel(distance.max(1. / OMEGA));

            density += influence;
        }

        density.max(1. / OMEGA)
    }

    fn density_to_pressure(&self, density: f32) -> f32 {
        let pressure =
            self.fluid_stiffness * ((density.max(1. / OMEGA) / self.rest_density).powi(7) - 1.);

        pressure.clamp(-OMEGA, OMEGA)
    }

    fn densities_to_shared_pressure(&self, density_one: f32, density_two: f32) -> f32 {
        let pressure_one = self.density_to_pressure(density_one);
        let pressure_two = self.density_to_pressure(density_two);

        (pressure_one + pressure_two) * 0.5
    }

    fn smoothing_kernel(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.;
        }

        let kernel_value = ((self.radius - distance).powi(2) / self.volume).clamp(0., OMEGA);

        kernel_value
    }

    fn smoothing_kernel_derivative(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.;
        }

        let scale = 12. / (self.radius.powi(4) * PI);

        (distance - self.radius) * scale
    }

    fn apply_velocity(&self, damping_factor: f32, delta_time: f32) {
        for i in 0..self.number_of_particles {
            let current_position = self.positions[i].get();
            let mut current_velocity = self.velocities[i].get();

            if !current_velocity.x.is_finite() || !current_velocity.y.is_finite() {
                println!(
                    "Invalid velocity detected for particle {}: {:?}. Resetting to zero.",
                    i, current_velocity
                );
                current_velocity = Vector2::zero();
            }
            if current_velocity.x == 0.0 && current_velocity.y == 0.0 {
                println!(
                    "Zero velocity detected for particle {}. No movement applied.",
                    i
                );
                continue;
            }

            let proposed_position = current_position + current_velocity * delta_time;

            let mut final_position = proposed_position;
            let mut velocity = current_velocity;

            if proposed_position.x < 0. || proposed_position.x > self.bounds.x {
                if current_velocity.x != 0.0 {
                    let time_to_collision_x = if proposed_position.x < 0. {
                        -current_position.x / current_velocity.x
                    } else {
                        (self.bounds.x - current_position.x) / current_velocity.x
                    };

                    velocity.x = -velocity.x * damping_factor;

                    let remaining_time = delta_time - time_to_collision_x;

                    final_position.x = if proposed_position.x < 0. {
                        0. + velocity.x * remaining_time
                    } else {
                        self.bounds.x + velocity.x * remaining_time
                    };
                } else {
                    println!(
                        "Zero velocity.x for particle {} during X-axis collision correction.",
                        i
                    );
                    velocity.x = 0.0;
                }
            }

            if proposed_position.y < 0. || proposed_position.y > self.bounds.y {
                if current_velocity.y != 0.0 {
                    let time_to_collision_y = if proposed_position.y < 0. {
                        -current_position.y / current_velocity.y
                    } else {
                        (self.bounds.y - current_position.y) / current_velocity.y
                    };

                    velocity.y = -velocity.y * damping_factor;

                    let remaining_time = delta_time - time_to_collision_y;

                    final_position.y = if proposed_position.y < 0. {
                        0. + velocity.y * remaining_time
                    } else {
                        self.bounds.y + velocity.y * remaining_time
                    };
                } else {
                    println!(
                        "Zero velocity.y for particle {} during Y-axis collision correction.",
                        i
                    );
                    velocity.y = 0.0;
                }
            }

            if !velocity.x.is_finite() || !velocity.y.is_finite() {
                println!(
                    "Invalid velocity during bounds correction for particle {}: {:?}",
                    i, velocity
                );
                velocity = Vector2::zero();
            }

            self.positions[i].set(final_position);
            self.velocities[i].set(velocity);
        }
    }

    fn simulate_step(&self, delta_time: f32) {
        self.update_densities();

        for i in 0..self.number_of_particles {
            let pressure_force = self.calculate_pressure_force(i);
            let gravity = Vector2::new(0., self.gravity * delta_time);

            let total_force = pressure_force + gravity;
            let acceleration = total_force / self.densities[i].get().max(1. / OMEGA);

            let mut new_velocity = self.velocities[i].get() + acceleration * delta_time;

            let max_velocity = OMEGA;
            if new_velocity.length() > max_velocity {
                new_velocity = new_velocity.normalized() * max_velocity;
            }

            self.velocities[i].set(new_velocity);
        }

        self.apply_velocity(0.8, delta_time);
    }
}

fn main() {
    const WIDTH: i32 = 1600;
    const HEIGHT: i32 = 800;

    let (mut rl, thread) = raylib::init()
        .size(WIDTH, HEIGHT)
        .title("Confined Fluid Sim")
        .build();

    const FPS: u32 = 60;
    const DELTA_TIME: f32 = 1. / FPS as f32;

    rl.set_target_fps(FPS);

    let sim = Simulation::new_from_grid(
        1225,
        Vector2::new(10., 10.),
        35,
        15.,
        0.5,
        130.,
        0.004,
        0.95,
        Vector2::new(WIDTH as f32, HEIGHT as f32),
        0.7,
    );

    let mut average_framerate = 0.;
    let mut frame_count = 0;
    let test_time = Duration::from_secs_f32(40.);

    let start_time = Instant::now();

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        for _ in 0..2 {
            sim.simulate_step(DELTA_TIME);
        }
        for (_i, pos) in sim.positions.iter().enumerate() {
            let position = pos.get();
            d.draw_circle(position.x as i32, position.y as i32, 6.5, Color::BLUE);
        }

        println!(
            "{}",
            &sim.densities
                .iter()
                .map(|e| e.get())
                .reduce(|acc, e| acc + e)
                .unwrap()
                / sim.number_of_particles as f32
        );

        d.draw_fps(0, 0);

        average_framerate += d.get_frame_time();
        frame_count += 1;

        if Instant::now().duration_since(start_time) > test_time {
            println!(
                "average framerate: {}",
                average_framerate / frame_count as f32
            );
            break;
        }
    }
}
