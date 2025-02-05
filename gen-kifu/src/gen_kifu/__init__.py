import os
import pathlib
import time
from gen_kifu import wtb2h5

WTHOR_PATH = "wthor"

def main() -> None:
    print("Hello from gen-kifu!")

def get_wthor() -> None:
    dir_path = pathlib.Path(WTHOR_PATH)
    if dir_path.exists():
        print("WTHOR files already exist.")
    dir_path.mkdir()
    try:
        for year in range(1990, 2025):
            if 0 != os.system(f"wget https://www.ffothello.org/wthor/base_zip/WTH_{year}.ZIP"):
                raise Exception("Failed to download WTHOR files.")
            if 0 != os.system(f"unzip WTH_{year}.ZIP -d {dir_path / str(year)}"):
                raise Exception("Failed to unzip WTHOR files.")
            if 0 != os.system(f"rm WTH_{year}.ZIP"):
                raise Exception("Failed to remove WTHOR zip files.")
            time.sleep(1)
    except Exception as e:
        print(e)
        return
    print("Downloaded WTHOR files.")

def run_wtb2h5() -> None:
    wtb2h5.main()
