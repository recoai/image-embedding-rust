pub enum Normalization {
    None,
    MeanStd([f64; 3], [f64; 3]),
    MinOnePlusOne
}

pub enum ImageShape {
    // Batch x Channel x Height x Width
    BCHW,
    // Batch x Height x Width x Channel
    BHWC,

}

pub struct ModelConfig {
    pub model_name: String,
    pub model_url: String,
    pub normalization: Normalization,
    pub image_shape: ImageShape,
    pub image_size: usize
}

pub enum Model {
    MobileNetV2,
    ResNet152,
    EfficientNetLite4
}

pub fn dispatch_model(model: Model) -> ModelConfig {
    match model {
        Model::MobileNetV2 => ModelConfig {
            model_name: "MobileNet V2".into(),
            model_url: "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx".into(),
            normalization: Normalization::MeanStd([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            image_shape: ImageShape::BCHW,
            image_size: 224
        },
        Model::ResNet152 => ModelConfig {
            model_name: "".to_string(),
            model_url: "".to_string(),
            normalization: Normalization::None,
            image_shape: ImageShape::BCHW,
            image_size: 0
        },
        Model::EfficientNetLite4 => ModelConfig {
            model_name: "".to_string(),
            model_url: "".to_string(),
            normalization: Normalization::None,
            image_shape: ImageShape::BCHW,
            image_size: 0
        }
    }
}