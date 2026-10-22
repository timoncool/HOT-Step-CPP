HOT-Step CPP
============

QUICK START
Extract the complete archive. Windows: double-click HOT-Step.bat.
Linux or macOS: run ./HOT-Step.sh.
Open http://localhost:3001 if the browser does not open automatically.
Download models from the Model Manager.

DATA AND UPGRADES
User data defaults to HOT-Step-Data beside the app folder. It includes library,
audio, training, models, adapters, logs, tools and settings/.env. Engine binaries,
UI, bundled runtime, shipped plugins and server/data catalogues stay in the app.

Your profile saves the data location in data-location.json:
  Windows: %LOCALAPPDATA%/HOT-Step
  Linux: $XDG_CONFIG_HOME/hot-step or ~/.config/hot-step
  macOS: ~/Library/Application Support/HOT-Step

Stop the app, back up the data folder, and extract the next version separately.
The new launcher reuses the saved location and downloaded weights.
With no pointer, an older install launched in place migrates to the sibling folder.
A fresh extraction elsewhere can import the old app folder through Settings.

Settings > Environment > Data folder provides Move current data, Use existing
folder, Create empty data folder and Import previous install. Preview the paths
and space required, finish active work, then confirm and restart. External
folder overrides stay external. Import needs an empty library and keeps the
older install. Repoint copies and deletes nothing.

If a drive is missing or a transfer stops, keep the pointer and data intact.
Read the explanation page, fix the reported problem and restart to resume.

CONFIGURATION
Settings writes <data folder>/settings/.env from the shipped .env.example.
Leave managed paths unset. A process HOT_STEP_DATA_ROOT override selects another
existing folder for that launch.

REQUIREMENTS
Use the platform and GPU variant matching your system. Node.js is bundled.
Models require additional disk space and internet for the first download.
Linux ROCm builds require compatible system ROCm libraries.
If macOS blocks the unsigned app: xattr -cr /path/to/extracted-app/

Documentation and support: https://github.com/scragnog/HOT-Step-CPP
