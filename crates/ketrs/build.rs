//! Compiles the GPU compute shaders (GLSL in `shaders/`) to SPIR-V with
//! glslc, into Cargo's `OUT_DIR`. Only for the opt-in `gpu` feature.
//!
//! glslc comes with the Vulkan SDK and with most distributions' `shaderc` or
//! `glslc` packages; set `GLSLC` to use a specific binary. A shader with
//! `// ketrs-variants: NAME = a b c` lines is compiled once per combination
//! of values, with `-DNAME=value`, to `<shader>_<value>[_<value>…].spv` (in
//! the order of the lines).

fn main() {
    #[cfg(feature = "gpu")]
    shaders::compile_all();
}

#[cfg(feature = "gpu")]
mod shaders {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::{env, fs};

    pub fn compile_all() {
        let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("shaders");
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        let glslc = env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
        println!("cargo:rerun-if-changed=shaders");
        println!("cargo:rerun-if-env-changed=GLSLC");

        for (name, file, define) in targets(&dir) {
            let mut cmd = Command::new(&glslc);
            cmd.args(["--target-env=vulkan1.2", "-O", "-I"]).arg(&dir);
            for d in define {
                cmd.arg(format!("-D{d}"));
            }
            cmd.arg("-o")
                .arg(out.join(format!("{name}.spv")))
                .arg(&file);
            let output = cmd.output().unwrap_or_else(|e| {
                panic!(
                    "ketrs's `gpu` feature compiles its shaders with glslc, but `{glslc}` could \
                     not be run ({e}).\nInstall glslc (Vulkan SDK, or your distribution's \
                     `shaderc`/`glslc` package), set GLSLC to its path, or build without the \
                     `gpu` feature."
                )
            });
            if !output.status.success() {
                panic!(
                    "glslc failed on shader {name}:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }

    /// `(output name, source file, defines)` for every binary, in order.
    fn targets(dir: &Path) -> Vec<(String, PathBuf, Vec<String>)> {
        let mut files: Vec<_> = fs::read_dir(dir)
            .expect("shaders directory")
            .map(|e| e.expect("directory entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "comp"))
            .collect();
        files.sort();
        let mut targets = Vec::new();
        for file in files {
            let stem = file.file_stem().unwrap().to_str().unwrap().to_owned();
            let source = fs::read_to_string(&file).expect("shader source");
            // Every combination of the variant lines' values.
            let mut combos = vec![(stem, Vec::new())];
            for spec in source
                .lines()
                .filter_map(|l| l.trim().strip_prefix("// ketrs-variants:"))
            {
                let (name, values) = spec.split_once('=').expect("NAME = values");
                combos = combos
                    .into_iter()
                    .flat_map(|(out, defines): (String, Vec<String>)| {
                        values.split_whitespace().map(move |v| {
                            let mut defines = defines.clone();
                            defines.push(format!("{}={v}", name.trim()));
                            (format!("{out}_{v}"), defines)
                        })
                    })
                    .collect();
            }
            targets.extend(
                combos
                    .into_iter()
                    .map(|(out, defines)| (out, file.clone(), defines)),
            );
        }
        targets
    }
}
