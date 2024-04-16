
# add help display for the script

# first program parameter is a script name


import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

import generators.torus_generator as tg
# create alias
import generators.dense_grid as dg



index = {
    "torus_generator": {
        "hook": tg,
        "description": "Generate colorful torus point cloud",
    },
    "dense_grid": {
        "hook": dg,
        "description": "Generate dense grid point cloud",
    },
    "single_torus": {
        "hook": tg.SingleTorus,
        "description": "create single torus point cloud",
    },
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <script_name> [script parameters]")
        print("Available scripts:")
        for key in index:
            desc = index[key]["description"]
            print(f"\t{key} - {desc}")
        return

    script_name = sys.argv[1]
    if script_name not in index:
        print(f"Error: script {script_name} not found")
        return

    script = index[script_name]
    script["hook"].script()

if __name__ == "__main__":
    main()