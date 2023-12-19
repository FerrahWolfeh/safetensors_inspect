use std::collections::HashMap;
use std::str::FromStr;

use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde_json::Value;

fn mspec_default_string() -> String {
    "N/A".to_string()
}

fn mspec_default_num() -> u8 {
    0
}

fn default_false() -> bool {
    false
}

fn default_none<T>() -> Option<T> {
    None
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Root {
    #[serde(rename = "__metadata__")]
    pub metadata: Metadata,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(rename = "modelspec.architecture", default = "mspec_default_string")]
    pub modelspec_architecture: String,
    #[serde(
        rename = "modelspec.encoder_layer",
        deserialize_with = "deserialize_number",
        default = "mspec_default_num"
    )]
    pub modelspec_encoder_layer: u8,
    #[serde(rename = "modelspec.implementation", default = "mspec_default_string")]
    pub modelspec_implementation: String,
    #[serde(rename = "modelspec.prediction_type", default = "mspec_default_string")]
    pub modelspec_prediction_type: String,
    #[serde(rename = "modelspec.resolution", default = "mspec_default_string")]
    pub modelspec_resolution: String,
    #[serde(rename = "modelspec.sai_model_spec", default = "mspec_default_string")]
    pub modelspec_sai_model_spec: String,
    #[serde(rename = "modelspec.title", default = "mspec_default_string")]
    pub modelspec_title: String,
    #[serde(deserialize_with = "deserialize_optional")]
    pub ss_adaptive_noise_scale: Option<f32>,
    #[serde(default = "mspec_default_string")]
    pub ss_base_model_version: String,
    #[serde(deserialize_with = "deserialize_bool")]
    pub ss_cache_latents: bool,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_caption_dropout_every_n_epochs: u8,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_caption_dropout_rate: f32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_caption_tag_dropout_rate: f32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_clip_skip: u8,
    #[serde(deserialize_with = "deserialize_string_json_object")]
    pub ss_dataset_dirs: HashMap<String, DatasetDirObj>,
    #[serde(deserialize_with = "deserialize_string_json_object")]
    pub ss_datasets: Vec<DatasetObj>,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_epoch: u8,
    #[serde(deserialize_with = "deserialize_optional")]
    pub ss_face_crop_aug_range: Option<String>,
    #[serde(deserialize_with = "deserialize_bool")]
    pub ss_full_fp16: bool,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_gradient_accumulation_steps: u8,
    #[serde(deserialize_with = "deserialize_bool")]
    pub ss_gradient_checkpointing: bool,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_learning_rate: f32,
    #[serde(deserialize_with = "deserialize_bool")]
    pub ss_lowram: bool,
    pub ss_lr_scheduler: String,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_lr_warmup_steps: u32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_max_grad_norm: f32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_max_token_length: u16,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_max_train_steps: u64,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_min_snr_gamma: f32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_multires_noise_discount: f32,
    #[serde(deserialize_with = "deserialize_optional")]
    pub ss_multires_noise_iterations: Option<f32>,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_network_alpha: u8,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_network_dim: u8,
    #[serde(deserialize_with = "deserialize_optional")]
    pub ss_network_dropout: Option<f32>,
    pub ss_network_module: String,
    pub ss_new_sd_model_hash: String,
    pub ss_new_vae_hash: Option<String>,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_noise_offset: f32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_num_batches_per_epoch: u16,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_num_epochs: u16,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_num_reg_images: u32,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_num_train_images: u32,
    pub ss_optimizer: String,
    pub ss_output_name: String,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_prior_loss_weight: f32,
    #[serde(deserialize_with = "deserialize_optional")]
    pub ss_scale_weight_norms: Option<f32>,
    pub ss_sd_model_name: String,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_seed: u64,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_steps: u32,
    #[serde(deserialize_with = "deserialize_string_json_object")]
    pub ss_tag_frequency: IndexMap<String, HashMap<String, u32>>,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_text_encoder_lr: f32,
    #[serde(deserialize_with = "deserialize_optional")]
    pub ss_training_comment: Option<String>,
    #[serde(deserialize_with = "deserialize_number")]
    pub ss_unet_lr: f32,
    #[serde(deserialize_with = "deserialize_bool")]
    pub ss_v2: bool,
    #[serde(default = "mspec_default_string")]
    pub ss_vae_hash: String,
    #[serde(default = "mspec_default_string")]
    pub ss_vae_name: String,
    #[serde(deserialize_with = "deserialize_bool", default = "default_false")]
    pub ss_zero_terminal_snr: bool,
    #[serde(
        deserialize_with = "deserialize_string_json_object",
        default = "default_none"
    )]
    pub ss_network_args: Option<IndexMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetDirObj {
    pub n_repeats: u8,
    pub img_count: u32,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetObj {
    pub is_dreambooth: bool,
    pub batch_size_per_device: u8,
    pub num_train_images: u32,
    pub num_reg_images: u32,
    pub resolution: [u16; 2],
    pub enable_bucket: bool,
    pub min_bucket_reso: u16,
    pub max_bucket_reso: u16,
    pub bucket_info: BucketInfo,
    pub subsets: Vec<Subset>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BucketInfo {
    pub buckets: IndexMap<String, Bucket>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bucket {
    pub resolution: [u16; 2],
    pub count: u32,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Subset {
    pub image_dir: String,
    pub img_count: u64,
    pub num_repeats: u8,
    pub color_aug: bool,
    pub flip_aug: bool,
    pub random_crop: bool,
    pub shuffle_caption: bool,
    pub keep_tokens: u8,
    pub class_tokens: Option<Vec<String>>,
    pub is_reg: bool,
}

fn deserialize_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "True" => Ok(true),
        "False" => Ok(false),
        _ => Err(serde::de::Error::custom("expected \"True\" or \"False\"")),
    }
}

fn deserialize_number<'de, D, N>(deserializer: D) -> Result<N, D::Error>
where
    D: Deserializer<'de>,
    N: FromStr,
{
    let s = String::deserialize(deserializer)?;
    match s.parse::<N>() {
        Ok(num) => Ok(num),
        Err(_) => Err(serde::de::Error::custom("failed to parse number")),
    }
}

fn deserialize_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "None" => Ok(None),
        "False" => Ok(None),
        _ => Ok(Some(match s.parse::<T>() {
            Ok(u) => u,
            Err(_) => {
                return Err(serde::de::Error::custom(
                    "failed to parse content of option",
                ))
            }
        })),
    }
}

fn deserialize_string_json_object<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let s = String::deserialize(deserializer)?;

    let v = serde_json::from_str::<Value>(&s).map_err(serde::de::Error::custom)?;

    let fht = serde_json::from_value::<T>(v).map_err(serde::de::Error::custom)?;

    Ok(fht)
}
