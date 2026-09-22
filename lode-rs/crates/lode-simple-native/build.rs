use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let vendor = manifest_dir.join("vendor");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("build output directory"));
    let pinyin_path = vendor.join("simple-0.7.1/contrib/pinyin.txt");
    let pinyin_data = fs::read(&pinyin_path).expect("read vendored pinyin data");
    let pinyin_header = out_dir.join("pinyin_data.h");
    let mut header = String::from("#ifndef LODE_PINYIN_DATA_H\n#define LODE_PINYIN_DATA_H\n\n");
    header.push_str("static const unsigned char LODE_PINYIN_DATA[] = {\n");
    for chunk in pinyin_data.chunks(16) {
        header.push_str("  ");
        for byte in chunk {
            write!(&mut header, "0x{byte:02x}, ").expect("write pinyin header");
        }
        header.push('\n');
    }
    header.push_str("};\n");
    writeln!(
        &mut header,
        "static const unsigned long LODE_PINYIN_DATA_LEN = {};",
        pinyin_data.len()
    )
    .expect("write pinyin length");
    header.push_str("\n#endif\n");
    fs::write(&pinyin_header, header).expect("write generated pinyin header");

    let sqlite_include = vendor.join("sqlite");
    let simple_src = vendor.join("simple-0.7.1/src");
    let cppjieba_include = vendor.join("cppjieba/include");
    let cppjieba_deps = vendor.join("cppjieba/deps");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        .define("USE_JIEBA", Some("1"))
        .define("SQLITE_CORE", Some("1"))
        .include(&sqlite_include)
        .include(&simple_src)
        .include(&cppjieba_include)
        .include(&cppjieba_deps)
        .include(&out_dir)
        .file(simple_src.join("pinyin.cc"))
        .file(simple_src.join("simple_tokenizer.cc"))
        .file(simple_src.join("simple_highlight.cc"))
        .file(simple_src.join("entry.cc"))
        .file(manifest_dir.join("src/native_entry.cc"))
        .warnings(false)
        .flag_if_supported("-fexceptions")
        .flag_if_supported("/EHsc")
        .compile("lode_simple_native");

    println!("cargo:rerun-if-changed={}", pinyin_path.display());
    println!("cargo:rerun-if-changed={}", simple_src.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/native_entry.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("cppjieba/include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("cppjieba/deps/limonp").display()
    );
    println!("cargo:rerun-if-changed={}", sqlite_include.display());
}
