use std::fs;
use std::fs::{copy, File};
use std::io::Write;
use std::path::Path;

use image::imageops::FilterType;
use tract_onnx::prelude::*;

use crate::image_transform::architectures::load_model_config;
use crate::image_transform::pipeline::{
    GenericTransform, ImageSize, MeanStdNormalization, ResizeRGBImage, ToArray, ToTensor,
    TransformationPipeline,
};
use crate::image_transform::utils::{model_filename, save_file_get};

pub type TractSimplePlan =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct ModelConfig {
    pub model_name: String,
    pub model_url: String,
    pub image_transformation: TransformationPipeline,
    pub image_size: ImageSize,
    pub layer_name: String,
}

pub struct ModelA {
    pub model_config: ModelConfig,
    pub model: TractSimplePlan,
}

pub fn load_model(config: &ModelConfig) -> TractSimplePlan {
    let name = config.model_name.clone();
    let url = config.model_url.clone();
    let filename = model_filename(&name);
    if !Path::new(&filename).exists() {
        println!("Downloading model file");
        save_file_get(&url, &filename);
    } else {
        println!("Skipping download");
    }

    let model = tract_onnx::onnx()
        .model_for_path(&filename)
        .expect("Cannot read model")
        .with_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec!(1, 3, config.image_size.width, config.image_size.height),
            ),
        )
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    model
}

pub fn load_model_architecture(model_arch: ModelArchitecture) -> TractSimplePlan {
    let model_config = load_model_config(model_arch);
    let model = load_model(&model_config);
    model
}

pub enum ModelArchitecture {
    MobileNetV2,
    ResNet152,
    EfficientNetLite4,
}
