use std::error::Error;

use image::imageops::{resize, FilterType};
use image::{ImageBuffer, Rgb, RgbImage};
use tract_onnx::prelude::{tract_ndarray, Tensor};

use crate::image_transform::functions::image_to_tensor;

pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub enum ImageTransformEnum {
    ResizeRGBImage(ResizeRGBImage),
    ToTensor,
}

pub enum ImageTransformResult {
    RgbImage(RgbImage),
    Tensor(Tensor),
}

impl From<RgbImage> for ImageTransformResult {
    fn from(rgb_image: RgbImage) -> Self {
        ImageTransformResult::RgbImage(rgb_image)
    }
}

impl From<Tensor> for ImageTransformResult {
    fn from(tensor: Tensor) -> Self {
        ImageTransformResult::Tensor(tensor)
    }
}

pub struct TransformationPipeline {
    pub steps: Vec<ImageTransformEnum>,
}

trait ImageTransform {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str>;
}

struct ResizeRGBImage {
    pub image_size: ImageSize,
    pub filter: FilterType,
}

impl ImageTransform for ResizeRGBImage {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => Ok(resize(
                &image,
                self.image_size.width as u32,
                self.image_size.height as u32,
                self.filter,
            )
            .into()),
            ImageTransformResult::Tensor(_) => Err("not implemented"),
        }
    }
}

struct ImageToTensor {
    pub image_size: usize,
}

impl ImageTransform for ImageToTensor {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                Ok(image_to_tensor(self.image_size, image).into())
            }
            ImageTransformResult::Tensor(_) => Err("not implemented"),
        }
    }
}
