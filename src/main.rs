use crate::image_transform::functions::read_rgb_image;
use crate::models::{load_model_architecture, ModelArchitecture};
use glob::glob;
use std::collections::HashMap;
use std::fs;
use std::str::FromStr;
use tract_onnx::prelude::*;

mod image_transform;
mod models;

fn test_imagenet() {}

fn main() -> Result<(), String> {
    let (model, pipeline) = load_model_architecture(ModelArchitecture::EfficientNetLite4);
    let mut n_good = 0;
    let mut n_bad = 0;

    for imagepath in glob("images/imagenet-sample-images/*.JPEG")
        .unwrap()
        .flatten()
    {
        let image = read_rgb_image(imagepath.to_str().unwrap());
        let image_tensor = pipeline
            .transform_image(&image)
            .expect("Cannot transform image");

        let result = model.run(tvec!(image_tensor)).unwrap();

        // find and display the max value with its index
        let best = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .zip(2..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let predicted_class = best.unwrap().1 - 2;

        let class_id_str = imagepath
            .to_str()
            .unwrap()
            .split('/')
            .last()
            .unwrap()
            .split('.')
            .nth(0)
            .unwrap();
        let true_class = i32::from_str(class_id_str).unwrap();

        if true_class == predicted_class {
            n_good += 1;
        } else {
            n_bad += 1;
        }

        println!(
            "{:} true {:} predicted {:} acc {:}",
            imagepath.to_str().unwrap(),
            true_class,
            predicted_class,
            (n_good as f32) / ((n_good + n_bad) as f32)
        );
    }

    Ok(())
}
