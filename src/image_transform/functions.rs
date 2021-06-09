use image::imageops::{resize, FilterType};
use image::{ImageBuffer, Rgb, RgbImage};
use tract_onnx::prelude::{tract_ndarray, Tensor};

pub fn resize_rgb_image(image_size: usize, image: &RgbImage) -> RgbImage {
    resize(
        image,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    )
}

pub fn normalize_imagenet_to_tensor(image_size: usize, image: &RgbImage) -> Tensor {
    tract_ndarray::Array4::from_shape_fn((1, 3, image_size, image_size), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (image[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into()
}

pub fn image_to_tensor(image_size: usize, image: RgbImage) -> Tensor {
    tract_ndarray::Array4::from_shape_fn((1, 3, image_size, image_size), |(_, c, y, x)| {
        image[(x as _, y as _)][c] as f32
    })
    .into()
}

fn read_rgb_image(image_path: &String) -> RgbImage {
    image::open(image_path).unwrap().to_rgb8()
}
