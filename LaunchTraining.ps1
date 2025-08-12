# Fire Detection Training Launcher
Write-Host "Fire Detection Training Environment" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Ask user if they want to start training
$response = Read-Host "Start training? (Y/N)"

if ($response -eq "Y" -or $response -eq "y") {
    Write-Host "Starting training..." -ForegroundColor Green
    python Train.py
} elseif ($response -eq "N" -or $response -eq "n") {
    Write-Host "Training cancelled. Virtual environment is active." -ForegroundColor Cyan
    Write-Host "You can now enter Python commands or run scripts manually." -ForegroundColor Cyan
    Write-Host "Type 'exit' to close this session." -ForegroundColor Yellow
    
    # Keep PowerShell session open for manual commands
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
} else {
    Write-Host "Invalid input. Please enter Y or N." -ForegroundColor Red
}