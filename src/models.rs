use crate::image_transform::pipeline::{
    GenericTransform, ImageSize, MeanStdNormalization, ResizeRGBImage, ToArray, ToTensor,
};
use image::imageops::FilterType;

pub struct ModelConfig {
    pub model_name: String,
    pub model_url: String,
    pub image_transformation: Vec<Box<dyn GenericTransform>>,
    pub layer_name: String,
}

pub enum Model {
    MobileNetV2,
    ResNet152,
    EfficientNetLite4,
}

pub fn dispatch_model(model: Model) -> ModelConfig {
    match model {
        Model::MobileNetV2 => ModelConfig {
            model_name: "MobileNetV2".into(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx".into(),
            image_transformation: vec![
                Box::new(ResizeRGBImage { image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                Box::new(ToArray {}),
                Box::new(MeanStdNormalization {means: [0.485, 0.456, 0.406] , stds: [0.229, 0.224, 0.225]}),
                Box::new(ToTensor {}),
            ],
            layer_name: "Reshape_103".to_string()
        },
        Model::ResNet152 => ModelConfig {
            model_name: "ResNet152".to_string(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v2-7.onnx".to_string(),
            image_transformation: vec![
                Box::new(ResizeRGBImage{ image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                Box::new(ToTensor {})
            ],
            layer_name: "resnetv27_flatten0_reshape0".to_string()
        },
        Model::EfficientNetLite4 => ModelConfig {
            model_name: "EfficientNet-Lite4".to_string(),
            model_url: "https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx".to_string(),
            image_transformation: vec![
                Box::new(ResizeRGBImage{ image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }),
                Box::new(ToTensor {})
            ],
            layer_name: "efficientnet-lite4/model/head/Squeeze".into()
        }
    }
}
