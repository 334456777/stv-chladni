//! Windows 上给 stv.exe 嵌入图标和版本信息。

fn main() {
    println!("cargo:rerun-if-changed=assets/stv.ico");
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winresource::WindowsResource::new()
            .set_icon("assets/stv.ico")
            .set("ProductName", "stv")
            .set("FileDescription", "stv - speak to video")
            .set("LegalCopyright", "Copyright (c) 2026 JINGHAN YU")
            .compile()
            .expect("嵌入 Windows 资源失败（需要 Windows SDK 的 rc.exe）");
    }
}
