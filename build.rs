fn main() {
    // If we are on x86_64 then we can use the system's OpenBLAS

    #[cfg(target_arch = "x86_64")]
    {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");
        // add customized linkage path to the linker
        // link to openblas
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
