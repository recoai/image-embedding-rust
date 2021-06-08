mod models;

use clap::{AppSettings, Clap};
use std::error::Error;
use tract_onnx::prelude::*;
use std::time::Instant;

#[derive(Clap, Debug)]
struct Image {
    #[clap(long, default_value = "cat.jpeg")]
    image_path: String,
    #[clap(long, default_value = "Reshape_103")]
    layer_name: String,
    #[clap(long)]
    normalize: bool,
    #[clap(long, default_value = "224")]
    image_size: usize,
}

#[derive(Clap, Debug)]
enum SubCommand {
    Inspect,
    Embed(Image),
}

#[derive(Clap, Debug)]
#[clap(version = "0.1", author = "Paweł Jankiewicz")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    #[clap(long, default_value = "mobilenetv2-7.onnx")]
    model_path: String,
    #[clap(subcommand)]
    subcmd: SubCommand,
}

fn inspect_model(opts: Opts) -> Result<(), Box<dyn Error>> {
    let model = tract_onnx::onnx().model_for_path(opts.model_path)?;

    for name in model.node_names() {
        println!("{:?}", name);
    }

    Ok(())
}

fn embed(opts: &Opts, image_opt: &Image) -> Result<(), Box<dyn Error>> {
    let image_size = image_opt.image_size;
    let start = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(opts.model_path.clone())?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, image_size, image_size)),
        )?
        .with_output_names(vec![image_opt.layer_name.clone()])?
        .into_optimized()?
        .into_runnable()?;
    println!("Model loaded in {:?} seconds", start.elapsed().as_millis() as f64 / 1000.0);

    let image = image::open(&image_opt.image_path).unwrap().to_rgb8();
    let resized = image::imageops::resize(
        &image,
        image_size as u32,
        image_size as u32,
        ::image::imageops::FilterType::Triangle,
    );

    let image: Tensor = if image_opt.normalize {
        tract_ndarray::Array4::from_shape_fn((1, 3, image_size, image_size), |(_, c, y, x)| {
            let mean = [0.485, 0.456, 0.406][c];
            let std = [0.229, 0.224, 0.225][c];
            (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
        })
        .into()
    } else {
        tract_ndarray::Array4::from_shape_fn((1, 3, image_size, image_size), |(_, c, y, x)| {
            resized[(x as _, y as _)][c] as f32
        })
        .into()
    };

    // run the model on the input
    let start = Instant::now();
    let result = model.run(tvec!(image))?;
    println!("Predicted in {:?} seconds", start.elapsed().as_millis() as f64 / 1000.0);

    let best: Vec<_> = result[0].to_array_view::<f32>()?.iter().cloned().collect();

    println!("Layer size {:?}", best.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let opts: Opts = Opts::parse();
    match &opts.subcmd {
        SubCommand::Inspect => inspect_model(opts),
        SubCommand::Embed(image_opt) => embed(&opts, &image_opt),
    }
}
