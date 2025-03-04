import os
import pathlib
import sys
import time
from gen_kifu import wtb2h5
from gen_kifu import mcts as mcts_gen
from gen_kifu import egaroucid as egaroucid_gen

WTHOR_PATH = "wthor"
EGAROUCID_PATH = "egaroucid"

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

def mcts() -> None:
    if len(sys.argv) > 2:
        print("Usage: uv run mcts [--resume (optional)]")
        raise ValueError(f"Invalid arguments: {sys.argv[1:]}")
    resume = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "--resume":
            resume = True
        else:
            print("Usage: uv run mcts [--resume (optional)]")
            raise ValueError(f"Invalid argument: {sys.argv[1]}")
    mcts_gen.main(resume=resume)

def get_egaroucid():
    dir_path = pathlib.Path(EGAROUCID_PATH)
    if dir_path.exists():
        print("Egaroucid files already exist.")
    dir_path.mkdir()
    try:
        if 0 != os.system("wget https://github.com/Nyanyan/Egaroucid/releases/download/training_data/Egaroucid_Train_Data.zip"):
            raise Exception("Failed to download Egaroucid files.")
        if 0 != os.system("unzip Egaroucid_Train_Data.zip -d egaroucid"):
            raise Exception("Failed to unzip Egaroucid files.")
        if 0 != os.system("rm Egaroucid_Train_Data.zip"):
            raise Exception("Failed to remove Egaroucid zip files.")
    except Exception as e:
        print(e)
        print("Failed to download Egaroucid files.")
        print("Removing Egaroucid files...")
        if 0 != os.system("rm -rf egaroucid"):
            print("Failed to remove Egaroucid files.")
        return
    print("Downloaded Egaroucid files.")

def egaroucid() -> None:
    egaroucid_gen.main()
