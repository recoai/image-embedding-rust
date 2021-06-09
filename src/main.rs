use crate::models::{load_model_architecture, ModelArchitecture};
use std::collections::HashMap;
use std::fs;

mod image_transform;
mod models;

fn test_imagenet() {}

fn main() -> Result<(), String> {
    let model = load_model_architecture(ModelArchitecture::MobileNetV2);
    Ok(())
}
