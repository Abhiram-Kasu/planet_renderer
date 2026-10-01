use naga::valid::{Capabilities, ValidationFlags, Validator};
use shaderc::{CompileOptions, Compiler, EnvVersion, ShaderKind, SourceLanguage, TargetEnv};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn shader_files(directory: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let entries = fs::read_dir(directory).expect("failed to read shaders directory");
    for entry in entries {
        let path = entry.expect("failed to read shader directory entry").path();
        if path.is_dir() {
            shader_files(&path, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "hlsl")
        {
            files.push(path);
        }
    }
}

fn stage_for(path: &Path) -> (ShaderKind, &'static str, &'static str) {
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("shader filename must be UTF-8");
    if file.ends_with("_vert.hlsl") {
        (ShaderKind::Vertex, "vs_main", "Vertex")
    } else if file.ends_with("_frag.hlsl") {
        (ShaderKind::Fragment, "fs_main", "Fragment")
    } else if file.ends_with("_comp.hlsl") {
        (ShaderKind::Compute, "cs_main", "Compute")
    } else {
        panic!(
            "shader filename must end with _vert.hlsl, _frag.hlsl, or _comp.hlsl: {}",
            path.display()
        );
    }
}

fn compile_stage(compiler: &Compiler, path: &Path, kind: ShaderKind, entry: &str) -> String {
    let source = fs::read_to_string(path).expect("failed to read HLSL shader");
    let source_name = path.to_str().expect("shader path must be UTF-8");
    let mut options = CompileOptions::new().expect("failed to create shader compiler options");
    options.set_source_language(SourceLanguage::HLSL);
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_0 as u32);
    let spirv = compiler
        .compile_into_spirv(&source, kind, source_name, entry, Some(&options))
        .unwrap_or_else(|error| panic!("HLSL compilation failed for {}: {error}", path.display()));
    let module = naga::front::spv::parse_u8_slice(
        spirv.as_binary_u8(),
        &naga::front::spv::Options::default(),
    )
    .unwrap_or_else(|error| panic!("SPIR-V parsing failed for {}: {error}", path.display()));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| panic!("shader validation failed for {}: {error}", path.display()));
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .unwrap_or_else(|error| panic!("WGSL generation failed for {}: {error}", path.display()))
}

fn rust_identifier(name: &str) -> String {
    let mut result = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_alphanumeric() || character == '_' {
            if index == 0 && character.is_ascii_digit() {
                result.push('_');
            }
            result.push(character.to_ascii_lowercase());
        } else if !result.ends_with('_') {
            result.push('_');
        }
    }
    let trimmed = result.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "shader".to_string()
    } else {
        trimmed
    }
}

fn main() {
    let compiler = Compiler::new().expect("failed to initialize shader compiler");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let mut files = Vec::new();
    shader_files(Path::new("shaders"), &mut files);
    files.sort();
    let mut declarations = String::new();
    let mut identifiers = std::collections::HashSet::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let (kind, entry, stage) = stage_for(&path);
        let stem = path
            .file_stem()
            .and_then(|name| name.to_str())
            .expect("shader filename must be UTF-8");
        let identifier = rust_identifier(stem);
        assert!(
            identifiers.insert(identifier.clone()),
            "shader names must be unique after conversion to Rust identifiers: {identifier}"
        );
        let wgsl = compile_stage(&compiler, &path, kind, entry);
        let wgsl_path = format!("{identifier}.wgsl");
        fs::write(output.join(&wgsl_path), wgsl).expect("failed to write generated WGSL");
        declarations.push_str(&format!(
            "#[allow(non_upper_case_globals)] pub const {identifier}: crate::Shader<crate::{stage}Shader> = crate::Shader::from_generated(include_str!(concat!(env!(\"OUT_DIR\"), \"/{wgsl_path}\")), \"{entry}\");\n"
        ));
    }
    fs::write(output.join("shaders.rs"), declarations)
        .expect("failed to write generated shader module");
}
