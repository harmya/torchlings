use anyhow::{Context, Result};
use clap::{Arg, Command};
use colored::*;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

mod pretty;
mod venv;

fn main() -> Result<()> {
    let matches = Command::new("torchlings")
        .about("Exercises to get you used to reading and writing basic PyTorch code.")
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
        .subcommand(
            Command::new("init")
                .about("Initialize the exercises directory and create Python environment")
                .arg(
                    Arg::new("exercises_path")
                        .short('e')
                        .long("exercises-path")
                        .value_name("PATH")
                        .help("Path to exercises directory")
                        .default_value("exercises"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("test", sub)) => {
            let exercises_path = PathBuf::from(sub.get_one::<String>("exercises_path").unwrap());
            let verbose = sub.get_flag("verbose");

            // Ensure Python environment is set up before running tests
            setup_environment()?;
            run_tests(&exercises_path, verbose)?;
        }
        Some(("init", sub)) => {
            pretty::welcome_banner();
            let exercises_path = PathBuf::from(sub.get_one::<String>("exercises_path").unwrap());
            initialize_project(&exercises_path)?;
        }
        _ => unreachable!(), // clap ensures we have a subcommand
    }

    Ok(())
}

fn setup_environment() -> Result<()> {
    println!("{}", "Setting up Python environment...".bright_blue());

    venv::setup_python_environment()
        .map_err(|e| anyhow::anyhow!("Failed to setup Python environment: {:?}", e))?;

    println!("{}", "✅ Python environment ready!".bright_green());
    Ok(())
}

fn initialize_project(exercises_path: &Path) -> Result<()> {
    println!("{}", "Initializing Torchlings project...".bright_blue());

    // Create exercises directory if it doesn't exist
    if !exercises_path.exists() {
        std::fs::create_dir_all(exercises_path).context("Failed to create exercises directory")?;
        println!(
            "📁 Created exercises directory: {}",
            exercises_path.display()
        );
    }

    // Setup Python environment
    setup_environment()?;

    println!();
    println!(
        "{}",
        "🚀 Torchlings project initialized successfully!"
            .bright_green()
            .bold()
    );
    println!(
        "Run {} to start testing your exercises",
        "torchlings test".bright_cyan()
    );

    Ok(())
}

fn run_tests(exercises_path: &Path, verbose: bool) -> Result<()> {
    print_test_header();

    // Check if exercises directory exists
    if !exercises_path.exists() {
        println!("{}", "Exercises directory does not exist!".bright_red());
        println!(
            "Run {} to initialize the project",
            "torchlings init".bright_cyan()
        );
        return Ok(());
    }

    let python_files = find_python_files(exercises_path)?;
    if python_files.is_empty() {
        println!(
            "{}",
            "No Python files found in exercises directory!".bright_red()
        );
        println!(
            "Run {} to initialize the project",
            "torchlings init".bright_cyan()
        );
        return Ok(());
    }

    // Run pytest on the entire directory
    let success = run_pytest_on_directory(exercises_path, verbose)?;
    print_directory_test_summary(success, python_files.len());

    Ok(())
}

fn run_pytest_on_directory(exercises_path: &Path, verbose: bool) -> Result<bool> {
    println!("Running pytest on directory: {}", exercises_path.display());

    match venv::run_pytest(Some(exercises_path.to_str().unwrap())) {
        Ok(()) => Ok(true),
        Err(venv::SetupError::CommandFailed(_)) => Ok(false),
        Err(e) => {
            if verbose {
                eprintln!("pytest execution error: {:?}", e);
            }
            anyhow::bail!("Failed to run pytest: {:?}", e)
        }
    }
}

fn print_test_header() {
    println!("{}", "Torchlings Test Runner".bright_yellow().bold());
    println!("{}", "=======================".bright_yellow());
    println!();
}

fn print_directory_test_summary(success: bool, file_count: usize) {
    println!();
    println!("{}", "=== Test Summary ===".bright_yellow().bold());

    if success {
        println!(
            "{} All tests passed across {} Python files! 🎉",
            "✅".bright_green(),
            file_count
        );
    } else {
        println!(
            "{} Some tests failed across {} Python files",
            "❌".bright_red(),
            file_count
        );
        println!("Run with {} for detailed output", "--verbose".bright_cyan());
    }
}

fn find_python_files(exercises_path: &Path) -> Result<Vec<PathBuf>> {
    if !exercises_path.exists() {
        anyhow::bail!(
            "Exercises directory does not exist: {}\nRun 'torchlings init' to create it.",
            exercises_path.display()
        );
    }

    let python_files: Vec<PathBuf> = WalkDir::new(exercises_path)
        .into_iter()
        .filter_entry(|entry| !is_ignored_directory(entry))
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| is_python_file(entry.path()))
        .map(|entry| entry.path().to_path_buf())
        .collect();

    Ok(python_files)
}

fn is_ignored_directory(entry: &DirEntry) -> bool {
    entry.path().components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(name) if name.starts_with('.') ||
                         name == "venv" ||
                         name == "__pycache__" ||
                         name.ends_with("-venv")
        )
    })
}

fn is_python_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext == "py")
        .unwrap_or(false)
}
