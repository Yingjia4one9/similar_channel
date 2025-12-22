# Create desktop shortcut for start.bat
$currentPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$batchFile = Join-Path $currentPath "start.bat"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "YouTube相似频道后端.lnk"

# Check if start.bat exists
if (-not (Test-Path $batchFile)) {
    Write-Host "Error: Cannot find start.bat file!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create shortcut
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $batchFile
$Shortcut.WorkingDirectory = $currentPath
$Shortcut.Description = "Start YouTube Similar Channels Backend Service"
$Shortcut.IconLocation = "C:\Windows\System32\shell32.dll,21"
$Shortcut.Save()

Write-Host "Success! Shortcut created on desktop." -ForegroundColor Green
Write-Host "Location: $shortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now find 'YouTube相似频道后端' shortcut on your desktop." -ForegroundColor Yellow

Read-Host "Press Enter to exit"

