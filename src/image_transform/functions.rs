use image::imageops::{resize, FilterType};
use image::{ImageBuffer, Rgb, RgbImage};
use tract_onnx::prelude::{tract_ndarray, Tensor};

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

pub fn read_rgb_image(image_path: &str) -> RgbImage {
    image::open(image_path).unwrap().to_rgb8()
}

// Better way to deal with mean and std
//
//     // Imagenet mean and standard deviation
//     let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.485, 0.456, 0.406])?;
//     let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.229, 0.224, 0.225])?;
//
//     let img = image::open("elephants.jpg").unwrap().to_rgb8();
//     let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
//     let image: Tensor =
//         ((tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
//             resized[(x as _, y as _)][c] as f32 / 255.0
//         }) - mean)
//             / std)
//             .into();
