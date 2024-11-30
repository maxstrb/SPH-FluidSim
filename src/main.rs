use raylib::prelude::*;

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(800, 600)
        .title("Shader Particles")
        .build();

    let fps = 120;
    rl.set_target_fps(fps);

    while !rl.window_should_close() {
        // Begin drawing
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        println!("{}", d.get_frame_time());
    }
}
