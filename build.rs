fn main() {
    let sub_dir = if cfg!(debug_assertions) { "debug" } else { "release" };

    // Copy the database file to the build target directory pre-build.
    if std::fs::copy("data.db", format!("target\\{}\\data.db", sub_dir)).is_err() {
        eprintln!("Error attempting to copy the database file.");
    }
}

#[cfg(windows)]
fn windows_only() {
}
