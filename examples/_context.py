
from pathlib import Path

try:
    import eikonal
except ImportError:
    import sys
    sys.path.append('..')
    import eikonal

print("eikonal.__version__:", eikonal.__version__)

#tmp_dir = Path(__file__).parent / 'tmp'
#tmp_dir.mkdir(exist_ok=True)
