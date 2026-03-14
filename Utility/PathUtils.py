import os
import shutil
import sys
from pathlib import Path


APP_NAME = 'FAE'


def get_repo_root():
    return Path(__file__).resolve().parents[1]


def get_app_root():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return get_repo_root()


def get_resource_root():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return get_repo_root()


def get_resource_path(*parts):
    return str(get_resource_root().joinpath(*parts))


def get_user_data_root():
    local_app_data = os.environ.get('LOCALAPPDATA')
    base_dir = Path(local_app_data) if local_app_data else get_app_root()
    user_data_root = base_dir / APP_NAME
    user_data_root.mkdir(parents=True, exist_ok=True)
    return user_data_root


def get_user_data_path(*parts):
    target_path = get_user_data_root().joinpath(*parts)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return str(target_path)


def ensure_user_copy(*parts):
    source_path = get_resource_root().joinpath(*parts)
    target_path = get_user_data_root().joinpath(*parts)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.exists() and not target_path.exists():
        shutil.copy2(source_path, target_path)

    return str(target_path)
