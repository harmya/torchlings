use std::io;
use std::path::Path;
use std::process::Command;

pub const VENV_NAME: &str = ".torchlings";
pub const REQUIREMENTS: &[&str] = &["torch", "pytest", "numpy"];

#[derive(Debug)]
pub enum SetupError {
    CommandFailed(String),
    IoError(io::Error),
    UvNotFound,
    VenvCreationFailed,
    RequirementsInstallFailed,
}

impl From<io::Error> for SetupError {
    fn from(error: io::Error) -> Self {
        SetupError::IoError(error)
    }
}

/// Check if uv is installed and available
fn is_uv_installed() -> Result<bool, SetupError> {
    match Command::new("uv").arg("--version").output() {
        Ok(output) => Ok(output.status.success()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(SetupError::from(e)),
    }
}

/// Install uv using the official installer
fn install_uv() -> Result<(), SetupError> {
    let output = Command::new("sh")
        .arg("-c")
        .arg("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SetupError::CommandFailed(format!(
            "uv installation failed: {}",
            stderr
        )));
    }

    Ok(())
}

/// Check if virtual environment exists
fn venv_exists() -> bool {
    Path::new(VENV_NAME).exists()
}

/// Create virtual environment
fn create_venv() -> Result<(), SetupError> {
    let output = Command::new("uv").args(["venv", VENV_NAME]).output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SetupError::VenvCreationFailed);
    }

    Ok(())
}

/// Check if a specific requirement is installed in the venv
fn is_package_installed(package: &str) -> Result<bool, SetupError> {
    let output = Command::new("uv")
        .args(["pip", "list", "--format=freeze"])
        .env("VIRTUAL_ENV", VENV_NAME)
        .output()?;

    if !output.status.success() {
        return Ok(false);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout
        .lines()
        .any(|line| line.starts_with(&format!("{}==", package))))
}

/// Check if all requirements are installed
fn are_requirements_installed() -> Result<bool, SetupError> {
    for &requirement in REQUIREMENTS {
        if !is_package_installed(requirement)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Install requirements in the virtual environment
fn install_requirements() -> Result<(), SetupError> {
    let output = Command::new("uv")
        .args(["pip", "install"])
        .args(REQUIREMENTS.iter())
        .env("VIRTUAL_ENV", VENV_NAME)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SetupError::RequirementsInstallFailed);
    }

    Ok(())
}

/// Main setup function - ensures uv is installed, venv exists, and requirements are met
pub fn setup_python_environment() -> Result<(), SetupError> {
    // Step 1: Ensure uv is installed
    if !is_uv_installed()? {
        println!("Installing uv...");
        install_uv()?;

        // Verify installation
        if !is_uv_installed()? {
            return Err(SetupError::UvNotFound);
        }
    }

    // Step 2: Create venv if it doesn't exist
    if !venv_exists() {
        println!("Creating virtual environment: {}", VENV_NAME);
        create_venv()?;
    }

    // Step 3: Install requirements if not already installed
    if !are_requirements_installed()? {
        println!("Installing requirements: {:?}", REQUIREMENTS);
        install_requirements()?;
    }

    println!("Python environment setup complete!");
    Ok(())
}

/// Run pytest in the virtual environment
pub fn run_pytest(test_path: Option<&str>) -> Result<(), SetupError> {
    let mut cmd = Command::new("uv");
    cmd.args(["run", "pytest"]);

    if let Some(path) = test_path {
        cmd.arg(path);
    }

    cmd.env("VIRTUAL_ENV", VENV_NAME);

    let output = cmd.output()?;

    // Print output regardless of success/failure
    println!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
    }

    if !output.status.success() {
        return Err(SetupError::CommandFailed(
            "pytest execution failed".to_string(),
        ));
    }

    Ok(())
}
