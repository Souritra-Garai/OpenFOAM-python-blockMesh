from pathlib import Path

import numpy as np

from modules.openfoam_dictionary import addHeader, getPolyMeshFilePath, file_EOF

def writePoints(case_path: Path, points:np.ndarray) -> dict:

	file_path = getPolyMeshFilePath(case_path, 'points')

	addHeader(file_path, {
		'format'	: 'ascii',
		'class'		: 'vectorField',
		'location'	: '"constant/polyMesh"',
		'object'	: 'points',
	})

	with open(file_path, 'a') as file :

		file.write('\n')

		n = points.shape[0]

		file.write(str(n) + '\n')
		file.write('(\n')

		for point in points :

			file.write('\t(' + ' '.join([str(x) for x in point]) + ')\n')

		file.write(')\n')

		file.write('\n')

		file.write(file_EOF)

		file.write('\n')

	pass