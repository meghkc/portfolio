# Refresh dependency pins using pip-tools on Windows PowerShell
# Usage: ./scripts/update_pins.ps1

param(
    [string]$Python = "python"
)

Write-Host "[pins] Ensuring pip-tools is installed..."
& $Python -m pip install --upgrade pip | Out-Null
& $Python -m pip install pip-tools | Out-Null

Write-Host "[pins] Compiling constraints from pyproject.toml ..."
& pip-compile --generate-hashes --output-file constraints.txt pyproject.toml

Write-Host "[pins] Done. Review git diff and commit updated constraints.txt"
