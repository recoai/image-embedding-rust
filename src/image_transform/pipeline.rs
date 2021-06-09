use std::error::Error;

use image::{ImageBuffer, Rgb, RgbImage};
use image::imageops::{FilterType, resize};
use tract_onnx::prelude::{Tensor, tract_ndarray};
use tract_onnx::prelude::tract_ndarray::Array4;
use tract_onnx::tract_core::ndarray::Array;
use crate::image_transform::pipeline::tract_ndarray::Ix4;
use crate::image_transform::functions::image_to_tensor;

pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub enum ImageTransformResult {
    RgbImage(RgbImage),
    Array4(Array4<f32>),
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
    pub steps: Vec<Box<dyn GenericTransform>>,
}

pub trait GenericTransform {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str>;
}

pub struct ResizeRGBImage {
    pub image_size: ImageSize,
    pub filter: FilterType,
}

impl GenericTransform for ResizeRGBImage {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                Ok(resize(
                    &image,
                    self.image_size.width as u32,
                    self.image_size.width as u32,
                    FilterType::Triangle,
                ).into())
            },
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

pub struct MeanStdNormalization {
    pub means: [f32; 3],
    pub stds: [f32; 3]
}

impl GenericTransform for MeanStdNormalization {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(_) => Err("Not implemented"),
            ImageTransformResult::Tensor(tensor) => Err("Not implemented"),
            ImageTransformResult::Array4(arr) => {
                let mean = Array::from_shape_vec((1, 3, 1, 1), self.means.to_vec()).expect("Wrong conversion to array");
                let std = Array::from_shape_vec((1, 3, 1, 1), self.stds.to_vec()).expect("Wrong conversion to array");
                let new_arr = (arr - mean) / std;
                Ok(ImageTransformResult::Array4(new_arr))
            }
        }
    }
}

pub struct Transpose {
    pub axes: [usize; 4]
}

impl GenericTransform for Transpose {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(_) => Err("Not implemented"),
            ImageTransformResult::Array4(arr) => {
                let arr = arr.permuted_axes(self.axes);
                Ok(ImageTransformResult::Array4(arr))
            },
            ImageTransformResult::Tensor(tensor) => {
                // note that the same operation on Tensor is not safe as it is on Array4
                let tensor = tensor.permute_axes(&self.axes).expect("Transpose should match the shape of the tensor");
                Ok(ImageTransformResult::Tensor(tensor))
            }
        }
    }
}

pub struct ToTensor {}

impl GenericTransform for ToTensor {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                let shape = image.dimensions();
                let tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
                    (1 as usize, 3 as usize, shape.0 as usize, shape.1 as usize),
                    |(_, c, y, x)| image[(x as _, y as _)][c] as f32
                ).into();
                Ok(ImageTransformResult::Tensor(tensor))
            }
            ImageTransformResult::Tensor(tensor) => {
                // already a tensor
                Ok(ImageTransformResult::Tensor(tensor))
            }
            ImageTransformResult::Array4(arr4) => {
                Ok(ImageTransformResult::Tensor(arr4.into()))
            }
        }
    }
}

pub struct ToArray {}

impl GenericTransform for ToArray {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                let shape = image.dimensions();
                let arr = tract_ndarray::Array4::from_shape_fn(
                    (1 as usize, 3 as usize, shape.0 as usize, shape.1 as usize),
                    |(_, c, y, x)| image[(x as _, y as _)][c] as f32
                );
                Ok(ImageTransformResult::Array4(arr))
            }
            ImageTransformResult::Tensor(tensor) => {
                // already a tensor
                let dyn_arr = tensor.into_array::<f32>().expect("Cannot convert tensor to Array4");

                let arr4 = dyn_arr.into_dimensionality::<Ix4>().expect("Cannot convert dynamic Array to Array4");
                Ok(ImageTransformResult::Array4(arr4))
            }
            ImageTransformResult::Array4(arr4) => {
                Ok(ImageTransformResult::Tensor(arr4.into()))
            }
        }
    }
}

pub fn run_pipeline(pipeline: &TransformationPipeline, image: RgbImage) -> Result<Tensor, &'static str> {
    let mut result = ImageTransformResult::RgbImage(image.clone());

    for step in &pipeline.steps {
        result = step.transform(result)?;
    }

    let to_tensor = ToTensor{};
    result = to_tensor.transform(result)?;

    match result {
        ImageTransformResult::Tensor(t) => Ok(t),
        _ => Err("Should be converted to tensor already")
    }
}

#[cfg(test)]
mod tests {
    use image::imageops::FilterType;

    use crate::image_transform::pipeline::{ImageSize, ResizeRGBImage, TransformationPipeline};

    use super::*;

    #[test]
    fn test_pipeline() {
        let pipeline = TransformationPipeline {
            steps: vec![
                Box::new(ResizeRGBImage{ image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                Box::new(ToTensor {})
            ]
        };
    }
}
