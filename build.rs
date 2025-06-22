use std::env;

fn main() {
    // Get the target triple, which is a string like "x86_64-unknown-linux-gnu"
    let target = env::var("TARGET").unwrap();
    println!("cargo:rerun-if-changed=build.rs");

    if target.starts_with("x86_64-unknown-linux-gnu") {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
