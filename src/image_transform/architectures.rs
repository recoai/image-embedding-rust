use crate::image_transform::pipeline::{
    ImageSize, Normalization, ResizeRGBImage, ToArray, ToTensor, TransformationPipeline, Transpose,
};
use crate::models::{Channels, ModelArchitecture, ModelConfig};
use image::imageops::FilterType;

pub fn load_model_config(model: ModelArchitecture) -> ModelConfig {
    match model {
        // Top-1 accuracy 1000 imagenet: 55.7% (39ms per image)
        ModelArchitecture::SqueezeNet => ModelConfig {
            model_name: "SqueezeNet".into(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.1-7.onnx".into(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    Box::new(ResizeRGBImage { image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                    Box::new(ToArray {}),
                    Box::new(Normalization { sub: [0.485, 0.456, 0.406], div: [0.229, 0.224, 0.225], zeroone: true }),
                    Box::new(ToTensor {}),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("squeezenet0_pool3_fwd".to_string()),
            channels: Channels::CWH
        },
        // Top-1 accuracy 1000 imagenet: 79.8% (75ms per image)
        ModelArchitecture::MobileNetV2 => ModelConfig {
            model_name: "MobileNetV2".into(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx".into(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    Box::new(ResizeRGBImage { image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                    Box::new(ToArray {}),
                    Box::new(Normalization { sub: [0.485, 0.456, 0.406], div: [0.229, 0.224, 0.225], zeroone: true }),
                    Box::new(ToTensor {}),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("Reshape_103".to_string()),
            channels: Channels::CWH
        },
        // Top-1 accuracy 1000 imagenet: 89.6% (502ms per image)
        ModelArchitecture::ResNet152 => ModelConfig {
            model_name: "ResNet152".to_string(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v2-7.onnx".to_string(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    Box::new(ResizeRGBImage { image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                    Box::new(ToArray {}),
                    Box::new(Normalization { sub: [0.485, 0.456, 0.406], div: [0.229, 0.224, 0.225], zeroone: true }),
                    Box::new(ToTensor {}),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("resnetv27_flatten0_reshape0".to_string()),
            channels: Channels::CWH
        },
        // Top-1 accuracy 1000 imagenet: 83.1% (220ms per image)
        ModelArchitecture::EfficientNetLite4 => ModelConfig {
            model_name: "EfficientNet-Lite4".to_string(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx".to_string(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    Box::new(ResizeRGBImage { image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                    Box::new(ToArray {}),
                    Box::new(Normalization { sub: [127.0, 127.0, 127.0], div: [128.0, 128.0, 128.0], zeroone: false }),
                    Box::new(ToTensor {}),
                    Box::new(Transpose { axes: [0, 2, 3, 1] }),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("efficientnet-lite4/model/head/Squeeze".into()),
            channels: Channels::WHC
        }
    }
}
