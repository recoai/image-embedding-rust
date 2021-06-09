use std::fs;
use std::io::Write;
use std::path::Path;

pub fn remove_non_alphanum(s: &String) -> String {
    let s_new: String = s
        .chars()
        .map(|x| match x {
            '0'..='9' => x,
            'A'..='Z' => x.to_ascii_lowercase(),
            'a'..='z' => x,
            ' ' => '_',
            _ => ' ',
        })
        .collect();
    s_new.replace(" ", "_")
}

pub fn model_filename(name: &String) -> String {
    let clean_name = remove_non_alphanum(name);
    format!("models/{}.onnx", clean_name)
}

pub fn save_file_get(url: &str, path: &str) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .referer(false)
        .build()
        .map_err(|e| e.to_string())?;

    let mut response = client.get(url).send().map_err(|e| e.to_string())?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().unwrap();
        return Err(if text.is_empty() {
            status.to_string()
        } else {
            text
        });
    }

    if !Path::new("models/").exists() {
        fs::create_dir("models").map_err(|e| e.to_string())?;
    }
    let mut out = fs::File::create(path).map_err(|e| e.to_string())?;
    out.write_all(&response.bytes().expect("Failed to convert to bytes"));

    Ok(())
}
