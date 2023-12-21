use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use bytesize::ByteSize;
use owo_colors::OwoColorize;
use sft_header::Metadata;
use std::env;

use crate::sft_header::Root;

mod sft_header;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path_arg = &args[1];

    let ayo = st_open(Path::new(path_arg), false).unwrap();

    let trained_resolutions = Vec::from_iter(
        ayo.ss_datasets
            .iter()
            .map(|ds| format!("{}x{}", ds.resolution[0], ds.resolution[1])),
    );

    let optimizer = ayo.ss_optimizer;

    let batch =
        ayo.ss_datasets.get(0).unwrap().batch_size_per_device * ayo.ss_gradient_accumulation_steps;

    let network_arch = match ayo.ss_network_module.as_str() {
        "networks.lora" => "Conventional LoRA",
        "lycoris.kohya" => {
            if let Some(args) = ayo.ss_network_args.clone() {
                if let Some(algo) = args.get("algo") {
                    match algo.as_str() {
                        "dylora" => "LyCORIS DyLoRA",
                        "full" => "LyCORIS Native Fine-Tune",
                        "glokr" => "LyCORIS GLoRA",
                        "glora" => "LyCORIS GLoKr",
                        "ia3" => "LyCORIS IA^3",
                        "locon" => "LyCORIS LoCon",
                        "loha" => "LyCORIS LoHa",
                        "lokr" => "LyCORIS LoKr",
                        "diag-oft" => "LyCORIS Diag-OFT",
                        "lora" => "LyCORIS LoRA",
                        _ => "Unknown",
                    }
                } else {
                    "LyCORIS LoCon"
                }
            } else {
                "Unknown"
            }
        }
        _ => "Unknown",
    };

    let model_arch = if ayo.modelspec_architecture != "N/A" {
        ayo.modelspec_architecture
    } else if ayo.ss_v2 {
        "StableDiffusion V2 LoRA".to_string()
    } else {
        "StableDiffusion V1.5 LoRA".to_string()
    };

    println!(
        "{}\n{} {model_arch}\n{} {network_arch}\n{} {}\n{} {}",
        "Network:".blue().bold().underline(),
        "SD architecture:".bold(),
        "Architecture:".bold(),
        "Linear dimension:".bold(),
        ayo.ss_network_dim,
        "Linear alpha:".bold(),
        ayo.ss_network_alpha
    );

    if ayo.ss_network_module.contains("lycoris") && ayo.ss_network_args.is_some() {
        let ay = ayo.ss_network_args.unwrap().clone();
        println!(
            "{} {}\n{} {}",
            "Conv dimension:".bold(),
            ay.get("conv_dim").unwrap(),
            "Conv alpha:".bold(),
            ay.get("conv_alpha").unwrap()
        )
    }

    println!(
        "{} {trained_resolutions:?}\n",
        "Trained Resolutions:".bold()
    );

    println!(
        "{}\n{} {optimizer}\n{} {}\n{} {}\n{} {}\n{} {batch:?}\n{} {}\n{} {}\n",
        "Training Parameters:".blue().bold().underline(),
        "Optimizer:".bold(),
        "UNet LR:".bold(),
        ayo.ss_unet_lr,
        "TE LR:".bold(),
        ayo.ss_text_encoder_lr,
        "Scheduler:".bold(),
        ayo.ss_lr_scheduler,
        "Total batch size:".bold(),
        "Noise Offset:".bold(),
        ayo.ss_noise_offset,
        "Scaled Weight Norms:".bold(),
        ayo.ss_scale_weight_norms.unwrap_or(0.0)
    );

    println!(
        "{}\n{} {}\n{} {}\n{} {}\n{} {}\n{} {}",
        "Dataset:".blue().bold().underline(),
        "Train images:".bold(),
        ayo.ss_num_train_images,
        "Configured Epochs:".bold(),
        ayo.ss_num_epochs,
        "Model Epoch:".bold(),
        ayo.ss_epoch,
        "Model Steps".bold(),
        ayo.ss_steps,
        "Effective steps trained:".bold(),
        ayo.ss_steps * batch as u32,
    );
}

fn st_open<P: AsRef<Path>>(filename: P, quiet: bool) -> io::Result<Metadata> {
    let filename = filename.as_ref();
    let metadata = filename.metadata()?;
    let file_size = metadata.size();

    if file_size < 9 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Length less than 9 bytes",
        ));
    }

    let mut file = File::open(filename)?;
    let mut buffer = [0; 9];

    file.read_exact(&mut buffer)?;

    let mut hdrsz = [0; 8];
    hdrsz.clone_from_slice(&buffer[..8]);

    if buffer[8] != 0x7B {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a valid safetensors file. Missing 0x7B byte at position 8",
        ));
    }

    let header_len = u64::from_le_bytes(hdrsz);
    if 8 + header_len > file_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Header is bigger than the file itself",
        ));
    }

    if !quiet {
        println!(
            "{}:\n{}: {}\n{}: {}\n",
            filename
                .file_name()
                .unwrap()
                .to_string_lossy()
                .blue()
                .bold()
                .underline(),
            "Model file size".bold(),
            ByteSize::b(file_size).to_string_as(true),
            "Model header length".bold(),
            ByteSize::b(header_len).to_string_as(true)
        );
    }

    let mut header_buf = vec![0; header_len as usize];
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut header_buf)?;

    let json: Root = serde_json::from_slice(&header_buf).unwrap();

    Ok(json.metadata)
}
