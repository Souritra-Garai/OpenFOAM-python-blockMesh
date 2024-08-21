import logging
from pathlib import Path

import numpy as np

from modules.faces import writeFaces, writeBoundary, writeOwners, writeNeighbours
from modules.points import writePoints

def writePolyMesh(case_dir_path: Path, faces:dict, points:np.ndarray) -> None:

	polyMesh_dir_path = case_dir_path / 'constant' / 'polyMesh'

	logging.info('Writing polyMesh files to {0}'.format(polyMesh_dir_path))

	polyMesh_dir_path.mkdir(parents=True, exist_ok=True)

	logging.info('Writing faces file')

	boundary_dict = writeFaces(case_dir_path, faces)

	logging.info('Writing boundary file')

	writeBoundary(case_dir_path, boundary_dict)

	logging.info('Writing owners file')

	writeOwners(case_dir_path, faces, boundary_dict)

	logging.info('Writing neighbours file')

	writeNeighbours(case_dir_path, faces)

	logging.info('Writing points file')

	writePoints(case_dir_path, points)