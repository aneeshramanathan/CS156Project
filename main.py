import sys
from pathlib import Path
from src.main import main

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    main()

