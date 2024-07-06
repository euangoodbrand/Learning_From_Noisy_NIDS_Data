import os
from pathlib import Path

def explore_directory(directory, indent=""):
    print(f"{indent}{directory.name}/")
    for item in sorted(directory.iterdir()):
        if item.is_dir():
            explore_directory(item, indent + "  ")
        elif not item.name.endswith(('.csv', '.png', '.pth', '.pt')):
            print(f"{indent}  {item.name}")

def main():
    current_dir = Path.cwd()
    root_dir = current_dir / 'LabelNoiseLearning'
    
    print(f"Exploring directory structure of: {root_dir}")
    explore_directory(root_dir)

if __name__ == "__main__":
    main()