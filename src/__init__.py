import sys
from pathlib import Path

file = Path(__file__).resolve()
file0 = file.parents[1]
file1 = file0 / 'src'
file2 = file0 / 'ISMIR2020_U_Nets_SVS'
sys.path.extend([str(file0), str(file1), str(file2)])