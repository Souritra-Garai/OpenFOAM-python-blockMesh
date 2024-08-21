from pathlib import Path

import numpy as np

from modules.faces import writeFaces, writeBoundary, writeOwners, writeNeighbours
from modules.points import writePoints

def writePolyMesh(case_dir_path: Path, faces:dict, points:np.ndarray) -> None:

	case_dir_path = Path('.')
	(case_dir_path / 'constant' / 'polyMesh').mkdir(parents=True, exist_ok=True)

	boundary_dict = writeFaces(case_dir_path, faces)

	writeBoundary(case_dir_path, boundary_dict)

	writeOwners(case_dir_path, faces, boundary_dict)

	writeNeighbours(case_dir_path, faces)

	writePoints(case_dir_path, points)