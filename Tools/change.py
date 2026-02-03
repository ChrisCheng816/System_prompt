import os
from collections import deque

ZERO_CHAR = "0"
FOUR_CHAR = "4"

def drop_last_char(text: str) -> str:
    chars = deque(text)
    chars.pop()
    return "".join(chars)

def rename_all_dirs(root_dir: str) -> None:
    for dirpath, dirnames, _filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if not dirname.endswith(ZERO_CHAR):
                continue

            old_path = os.path.join(dirpath, dirname)

            base = dirname.removesuffix(ZERO_CHAR)

            new_name = base + FOUR_CHAR
            new_path = os.path.join(dirpath, new_name)

            if os.path.exists(new_path):
                print("skip because target exists:", old_path, "to", new_path)
                continue

            os.rename(old_path, new_path)
            print("renamed:", old_path, "to", new_path)

if __name__ == "__main__":

    rename_all_dirs("../generation_results_codereval")