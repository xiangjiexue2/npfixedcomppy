//! Build script: compile the C++/Eigen core with MSVC (VS2022).
//!
//!  * Captures the MSVC environment by running `vcvars64.bat` (INCLUDE/LIB
//!    for the compiler + Windows SDK).
//!  * Compiles `cpp/npfc_core.cpp` as C++17, `/O2`, `/arch:AVX2` so Eigen
//!    selects its widest packet traits at compile time (the same source is
//!    recompiled for whatever the host supports).
//!  * OpenMP (optional): compile+link a probe with `/openmp`; when it
//!    succeeds the core is built with `/openmp` and linked against
//!    `libomp.lib` (the `vcomp140.dll` runtime ships with the VS runtime /
//!    Windows). When it fails the identical source compiles to serial loops.
//!
//! All C++ temporaries are RAII (Eigen / std::vector); the only raw pointer
//! the library hands out is the scratch pool, freed by `npfc_pool_free`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const VARS: &str = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat";
const VCROOT: &str =
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207";

/// Run `vcvars64.bat` in a temp batch file and parse its `set` output.
///
/// (Calling `cmd /c "<quoted path> && set"` directly is unreliable because
/// of cmd's quote-stripping rules; a .bat file with the path quoted inside
/// is the robust pattern.)
fn vcvars_env(out: &Path) -> HashMap<String, String> {
    let bat = out.join("vcvars_dump.bat");
    std::fs::write(&bat, format!("@echo off\r\ncall \"{VARS}\"\r\nset\r\n"))
        .expect("write vcvars_dump.bat");
    let o = Command::new("cmd.exe")
        .args(["/c", &bat.to_string_lossy()])
        .output()
        .expect("failed to run cmd for vcvars dump");
    let text = String::from_utf8_lossy(&o.stdout);
    let env: HashMap<String, String> = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    if !env.contains_key("INCLUDE") || !env.contains_key("LIB") {
        panic!("npfc build: vcvars64 did not export INCLUDE/LIB (VS2022 broken?)");
    }
    env
}

fn run_cl(env: &HashMap<String, String>, cwd: &std::path::Path, args: &[&str]) -> bool {
    let mut cmd = Command::new(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe");
    cmd.current_dir(cwd);
    for (k, v) in env {
        cmd.env(k, v);
    }
    for a in args {
        cmd.arg(a);
    }
    let st = cmd.status();
    match st {
        Ok(s) => s.success(),
        Err(e) => {
            println!("cargo:warning=npfc build: cl.exe failed to start: {e}");
            false
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=cpp/npfc_core.cpp");
    println!("cargo:rerun-if-changed=cpp/npfc_core.h");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let env = vcvars_env(&out);

    // ---- OpenMP probe: compile+link a trivial TU with /openmp ----
    let openmp = probe_openmp(&env, &out);
    if openmp {
        println!("cargo:warning=npfc build: OpenMP available -> parallel build (/openmp)");
    } else {
        println!("cargo:warning=npfc build: OpenMP unavailable -> serial build");
    }

    // ---- compile the core ----
    let cpp = manifest.join("cpp").join("npfc_core.cpp");
    let obj = out.join("npfc_core.obj");
    let inc_cpp = manifest.join("cpp");
    let inc_eigen = manifest.join("eigen");

    let mut args: Vec<String> = vec![
        "/nologo".into(),
        "/std:c++17".into(),
        "/O2".into(),
        "/arch:AVX2".into(),
        "/J".into(),
        "/EHsc".into(),
        "/Zc:__cplusplus".into(),
        format!("/I{}", inc_cpp.display()),
        format!("/I{}", inc_eigen.display()),
        "/c".into(),
        format!("/Fo{}", obj.display()),
        cpp.to_string_lossy().into_owned(),
    ];
    if openmp {
        args.push("/openmp".into());
    }
    let ok = run_cl(
        &env,
        &manifest,
        &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    assert!(ok, "cl.exe failed to compile cpp/npfc_core.cpp");

    // ---- link the object into the cdylib ----
    let obj_arg = obj.display().to_string().replace('\\', "/");
    println!("cargo:rustc-link-arg={obj_arg}");
    if openmp {
        println!("cargo:rustc-link-search=native={VCROOT}\\lib\\x64");
        println!("cargo:rustc-link-lib=static=libomp");
    }
}

fn probe_openmp(env: &HashMap<String, String>, out: &PathBuf) -> bool {
    std::fs::create_dir_all(out).ok();
    let probe = out.join("ompprobe.cpp");
    std::fs::write(
        &probe,
        "#include <omp.h>\nint main(){ return omp_get_max_threads() > 0 ? 0 : 1; }\n",
    )
    .ok();
    let probe_obj = out.join("ompprobe.obj");
    let probe_exe = out.join("ompprobe.exe");
    let mut ok = run_cl(
        env,
        out,
        &[
            "/nologo",
            "/std:c++17",
            "/openmp",
            "/EHsc",
            "/Fo",
            &probe_obj.display().to_string(),
            probe.to_string_lossy().as_ref(),
        ],
    );
    if !ok {
        return false;
    }
    // Link explicitly against libomp.lib the same way the real build does.
    ok = run_cl(
        env,
        out,
        &[
            "/nologo",
            "/openmp",
            &format!("/Fe:{}", probe_exe.display()),
            &probe_obj.display().to_string(),
            &format!("{VCROOT}\\lib\\x64\\libomp.lib"),
        ],
    );
    let _ = std::fs::remove_file(&probe);
    let _ = std::fs::remove_file(&probe_obj);
    let _ = std::fs::remove_file(&probe_exe);
    ok
}
