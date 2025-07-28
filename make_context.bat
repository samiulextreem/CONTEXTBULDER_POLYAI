@echo off
REM Make Context Batch File
REM Creates text files for AI context from documents in a specified folder
REM Self-contained with GUI folder selection

echo ========================================
echo PolycircleAI Context Maker
echo ========================================
echo.

REM Check if folder path is provided as command line argument
if not "%~1"=="" (
    set "INPUT_FOLDER=%~1"
    set "OUTPUT_FOLDER=%~1"
    goto :process_folder
)

REM If no command line argument, show GUI folder selection
echo No folder specified. Opening folder selection dialog...
echo.

REM Use PowerShell to show folder browser dialog
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog; $folderBrowser.Description = 'Select folder containing documents to process for AI context'; $folderBrowser.ShowNewFolderButton = $false; if ($folderBrowser.ShowDialog() -eq 'OK') { $folderBrowser.SelectedPath } else { exit 1 }" > "%TEMP%\selected_folder.txt"

REM Check if user cancelled the dialog
if %ERRORLEVEL% neq 0 (
    echo.
    echo Folder selection was cancelled.
    echo.
    pause
    exit /b 1
)

REM Read the selected folder path from temp file
set /p INPUT_FOLDER=<"%TEMP%\selected_folder.txt"
set "OUTPUT_FOLDER=%INPUT_FOLDER%"

REM Clean up temp file
del "%TEMP%\selected_folder.txt" >nul 2>&1

:process_folder
REM Check if the folder exists
if not exist "%INPUT_FOLDER%" (
    echo ERROR: Folder does not exist: "%INPUT_FOLDER%"
    echo.
    pause
    exit /b 1
)

echo Selected folder: "%INPUT_FOLDER%"
echo Output folder: "%OUTPUT_FOLDER%" (same as input)
echo.

echo Processing documents...
echo.

REM Run the text extractor with the same folder for input and output
python main.py "%INPUT_FOLDER%" "%OUTPUT_FOLDER%"

REM Check if Python command was successful
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Text extraction failed
    echo Make sure Python is installed and main.py is working
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Context creation completed!
echo ========================================
echo.
echo Output files created in: "%OUTPUT_FOLDER%"
echo - all_documents_combined.txt (ready for AI context)
echo - processed_files.json (tracking file)
echo.
echo You can now set this folder as your context folder in the web interface.
echo.
pause
