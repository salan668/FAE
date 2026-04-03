@echo off
setlocal

cd /d "%~dp0"

for /f "delims=" %%V in ('python -c "import re, pathlib; t=pathlib.Path(r'HomeUI\VersionConstant.py').read_text(encoding='utf-8'); a=re.search(r'^MAJOR = (\d+)', t, re.M).group(1); b=re.search(r'^MINOR = (\d+)', t, re.M).group(1); c=re.search(r'^PATCH = (\d+)', t, re.M).group(1); print(f'{a}.{b}.{c}')"') do set "VERSION=%%V"

if not defined VERSION (
    echo Failed to read version from HomeUI\VersionConstant.py
    exit /b 1
)

echo Building FAE v%VERSION%...

pyinstaller --distpath "C:\Users\SunsServer\Project\FAE\FAEv%VERSION%" --workpath "C:\Users\SunsServer\Project\FAE\build" --clean --noconfirm MainFrameCall.spec

if errorlevel 1 (
    echo Build failed.
    exit /b %errorlevel%
)

echo Build completed: C:\Users\SunsServer\Project\FAE\FAEv%VERSION%
