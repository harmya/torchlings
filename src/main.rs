use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, Command};
use colored::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use walkdir::{DirEntry, WalkDir};

fn main() -> Result<()> {
    let matches = Command::new("torchlings")
        .about("A CLI tool for learning PyTorch")
        .version("0.1.0")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("test")
                .about("Run all Python exercise files and report results")
                .arg(
                    Arg::new("exercises_path")
                        .short('e')
                        .long("exercises-path")
                        .value_name("PATH")
                        .help("Path to exercises directory")
                        .default_value("exercises"),
                )
                .arg(
                    Arg::new("verbose")
                        .short('v')
                        .long("verbose")
                        .help("Show verbose output")
                        .action(clap::ArgAction::SetTrue),
                ),
        )
        .get_matches();

    if let Some(sub) = matches.subcommand_matches("test") {
        let exercises_path = PathBuf::from(sub.get_one::<String>("exercises_path").unwrap());
        let verbose = sub.get_flag("verbose");
        run_tests(&exercises_path, verbose)?;
    }

    Ok(())
}

fn run_tests(exercises_path: &Path, verbose: bool) -> Result<()> {
    println!("{}", "Torchlings Test Runner".bright_yellow().bold());
    println!("{}", "=======================".bright_yellow());
    println!();
    let python_files = find_python_files(exercises_path)?;
    if python_files.is_empty() {
        println!("{}", "No Python files found in exercises directory!".bright_red());
        return Ok(());
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut failed_files = Vec::new();

    for file in python_files {
        let success = run_python_file(&file, verbose)?;
        if success {
            println!("{} {}", "✅".bright_green(), file.display().to_string().bright_green());
            passed += 1;
        } else {
            println!("{} {}", "❌".bright_red(), file.display().to_string().bright_red());
            failed += 1;
            failed_files.push(file);
        }
    }

    println!();
    println!("{}", "=== Test Summary ===".bright_yellow().bold());
    if failed == 0 {
        println!("{} All {} tests passed! 🎉", "✅".bright_green(), passed);
    } else {
        println!(
            "{} {} passed, {} failed",
            if passed > 0 { "✅" } else { "" },
            passed.to_string().bright_green(),
            failed.to_string().bright_red()
        );
        if !failed_files.is_empty() {
            println!();
            println!("{}", "Failed files:".bright_red().bold());
            for file in failed_files {
                println!("  {} {}", "❌".bright_red(), file.display());
            }
        }
    }

    Ok(())
}

fn find_python_files(exercises_path: &Path) -> Result<Vec<PathBuf>> {
    if !exercises_path.exists() {
        anyhow::bail!("Exercises directory does not exist: {}", exercises_path.display());
    }

    fn is_unwanted(entry: &DirEntry) -> bool {
        entry
            .path()
            .components()
            .any(|c| matches!(c.as_os_str().to_str(), Some(s) if s.starts_with('.') || s == "venv" || s == ".venv"))
    }

    let mut python_files = Vec::new();

    for entry in WalkDir::new(exercises_path)
        .into_iter()
        .filter_entry(|e| !is_unwanted(e))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("py"))
    {
        python_files.push(entry.path().to_path_buf());
    }

    python_files.sort();
    Ok(python_files)
}

fn run_python_file(file_path: &Path, verbose: bool) -> Result<bool> {
    if verbose {
        println!("Running: {}", file_path.display());
    }

    let output = StdCommand::new("python3")
        .args(&["-m", "pytest", "-q", "--maxfail=1"])
        .arg(file_path)
        .output()
        .context("failed to spawn pytest")?;

    if verbose {
        println!("STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("STDERR:\n{}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(output.status.success())
}
