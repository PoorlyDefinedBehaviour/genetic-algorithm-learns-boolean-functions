use core::f32;

use rerun::{ViewCoordinates, external::glam::Vec3};

fn predict(input: &[f32], weights: &[f32]) -> f32 {
    assert_eq!(input.len(), weights.len());

    let y: f32 = input.iter().zip(weights).map(|(v, w)| v * w).sum();
    y
}

fn activation(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = 100;
    let initial_population_size = 500;

    let op = "AND";
    let data = if op == "AND" {
        [
            // Boolean AND
            (1.0, 1.0, 1.0_f32),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
    } else {
        [
            // Boolean OR
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ]
    };
    let mut population: Vec<_> = (0..initial_population_size)
        .map(|_| {
            [
                rand::random_range(-1.0..=1.0),
                rand::random_range(-1.0..=1.0),
                rand::random_range(-1.0..=1.0),
            ]
        })
        .collect();

    let rec = rerun::RecordingStreamBuilder::new("realtime_updates").spawn()?;
    rec.log(
        "world",
        &ViewCoordinates::LUF(), // or similar constant
    )?;

    let data_points: Vec<_> = data
        .iter()
        .map(|(_, _, expected)| Vec3::new(*expected, *expected, 0.0))
        .collect();

    for frame in 0..iterations {
        // Set time for animation
        rec.set_time_sequence("frame", frame as i64);

        population.sort_by(|w1, w2| {
            let mut total_error_w1 = 0.0;
            let mut total_error_w2 = 0.0;
            for (a, b, expected) in data {
                let y = activation(predict(&[1.0, a, b], w1));
                let error = expected - y;
                total_error_w1 += error.powf(2.0);

                let y = activation(predict(&[1.0, a, b], w2));
                let error = expected - y;
                total_error_w2 += error.powf(2.0);
            }

            total_error_w1.partial_cmp(&total_error_w2).unwrap()
        });

        let best = population[0];

        let prediction_points = data.iter().map(|(a, b, _)| {
            // Do not use an activation function here so we can see the points moving in the visualization.
            let y = predict(&[1.0, *a, *b], &best);
            Vec3::new(y, y, 0.0)
        });

        rec.log(
            "data",
            &rerun::Points3D::new(data_points.clone())
                .with_colors([rerun::Color::from_rgb(0, 0, 150)])
                .with_radii([0.1]),
        )?;

        rec.log(
            "prediction",
            &rerun::Points3D::new(prediction_points)
                .with_colors([rerun::Color::from_rgb(255, 100, 100)])
                .with_radii([0.1]),
        )?;

        for population in population.iter_mut().skip(10) {
            *population = population.map(|w| w + rand::random_range(-0.005..=0.005));
        }
    }

    let best = population[0];
    println! {"best weights: {best:?}"};
    for (a, b, _) in data {
        let y = activation(predict(&[1.0, a, b], &best));
        println!("{a} {op} {b} = {y}");
    }

    Ok(())
}
