import os
import warnings

warnings.filterwarnings("ignore")

PROHIBITED = ["venv"]


def dfs(dir="."):
    current_dir_files = os.listdir(dir)
    for file in current_dir_files:
        if file.endswith(".py"):
            # print(f"{dir}/{file}")
            os.system(f"isort {dir}/{file}")
            os.system(f"black {dir}/{file}")

        if file not in PROHIBITED and os.path.isdir(f"{dir}/{file}"):
            dfs(f"{dir}/{file}")


if __name__ == "__main__":
    dfs()
