from pathlib import Path

import numpy as np

from modules.openfoam_dictionary import addHeader, getPolyMeshFilePath, appendDictionary, file_EOF

def writeFaces(case_path: Path, face_dict: dict) -> dict:

	file_path = getPolyMeshFilePath(case_path, 'faces')

	addHeader(file_path, {
		'format'	: 'ascii',
		'class'		: 'faceList',
		'location'	: '"constant/polyMesh"',
		'object'	: 'faces',
	})

	num_faces = sum([len(faces['owner']) for faces in face_dict.values()])

	boundary_dict = {}

	with open(file_path, 'a') as file :

		file.write('\n')

		file.write(str(num_faces) + '\n')
		file.write('(\n')

		for vertices in face_dict['internal']['vertices'] :

			file.write('4({0} {1} {2} {3})\n'.format(*vertices))

		n = len(face_dict['internal']['owner'])

		for face in set(face_dict.keys()) - {'internal'} :

			for vertices in face_dict[face]['vertices'] :

				file.write('4({0} {1} {2} {3})\n'.format(*vertices))

			boundary_dict[face] = {'startFace':n, 'nFaces':len(face_dict[face]['owner']), 'type':'patch'}

			n += len(face_dict[face]['owner'])

		file.write(')\n')

		file.write('\n')

		file.write(file_EOF)

		file.write('\n')

	return boundary_dict

def writeBoundary(case_path: Path, boundary_dict: dict) -> None:

	file_path = getPolyMeshFilePath(case_path, 'boundary')

	addHeader(file_path, {
		'format'	: 'ascii',
		'class'		: 'polyBoundaryMesh',
		'location'	: '"constant/polyMesh"',
		'object'	: 'boundary',
	})
	
	with open(file_path, 'a') as file :

		file.write('\n')

		n = len(boundary_dict)

		file.write(str(n) + '\n')
		file.write('(\n')

		appendDictionary(file, boundary_dict, 1)

		# for boundary in boundary_dict :

		# 	file.write('\t' + boundary + '\n')
		# 	file.write('\t{\n')

		# 	appendDictionary(file, boundary_dict[boundary], 2)

		# 	file.write('\t}\n')

		# 	if n > 1 :

		# 		file.write('\n')

		# 	n -= 1

		file.write(')\n')

		file.write('\n')

		file.write(file_EOF)

		file.write('\n')

	pass

def getMeshStats(face_dict: dict) -> dict:

	n_points = max([max(face['vertices'].flatten(), default=0) for face in face_dict.values()])
	n_points += 1

	n_cells = max([max(face['owner'], default=0) for face in face_dict.values()])
	n_cells += 1

	n_faces = sum([len(faces['owner']) for faces in face_dict.values()])
	
	n_internal_faces = len(face_dict['internal']['owner'])

	return {'nPoints':n_points, 'nCells':n_cells, 'nFaces':n_faces, 'nInternalFaces':n_internal_faces}

def writeOwners(case_path: Path, face_dict: dict, boundary_dict: dict) -> None:

	file_path = getPolyMeshFilePath(case_path, 'owner')

	mesh_stats = getMeshStats(face_dict)

	addHeader(file_path, {
		'format'	: 'ascii',
		'class'		: 'labelList',
		'location'	: '"constant/polyMesh"',
		'object'	: 'owner',
		'note'		: '"nPoints: {0} nCells: {1} nFaces: {2} nInternalFaces: {3}"'.format(mesh_stats['nPoints'], mesh_stats['nCells'], mesh_stats['nFaces'], mesh_stats['nInternalFaces'])
	})

	with open(file_path, 'a') as file :

		file.write('\n')

		file.write(str(mesh_stats['nFaces']) + '\n')
		file.write('(\n')

		for owner in face_dict['internal']['owner'] :
			
			file.write(str(owner) + '\n')

		boundary_faces = sorted(boundary_dict.keys(), key=lambda x: boundary_dict[x]['startFace'])

		for boundary in boundary_faces :

			for owner in face_dict[boundary]['owner'] :
				
				file.write(str(owner) + '\n')

		file.write(')\n')

		file.write('\n')

		file.write(file_EOF)

		file.write('\n')

	pass

def writeNeighbours(case_path: Path, face_dict: dict) -> None:

	file_path = getPolyMeshFilePath(case_path, 'neighbour')

	mesh_stats = getMeshStats(face_dict)

	addHeader(file_path, {
		'format'	: 'ascii',
		'class'		: 'labelList',
		'location'	: '"constant/polyMesh"',
		'object'	: 'neighbour',
		'note'		: '"nPoints: {0} nCells: {1} nFaces: {2} nInternalFaces: {3}"'.format(mesh_stats['nPoints'], mesh_stats['nCells'], mesh_stats['nFaces'], mesh_stats['nInternalFaces'])
	})

	with open(file_path, 'a') as file :

		file.write('\n')

		file.write(str(mesh_stats['nInternalFaces']) + '\n')
		file.write('(\n')

		for neighbour in face_dict['internal']['neighbour'] :
			
			file.write(str(neighbour) + '\n')

		file.write(')\n')

		file.write('\n')

		file.write(file_EOF)

		file.write('\n')	

	pass

def groupFaces(face_dict: dict, faces:list[str], new_face_name:str='New Face') -> dict:

	vertices	= np.zeros((0, 4), dtype=int)
	owners		= np.zeros(0, dtype=int)

	for face in faces :

		face_data = face_dict.pop(face)

		vertices	= np.concatenate((vertices, face_data['vertices']), axis=0)
		owners		= np.concatenate((owners, face_data['owner']), axis=0)

	new_face = {'vertices':vertices, 'owner':owners}

	face_dict[new_face_name] = new_face

	return face_dict

def renameFaces(face_dict: dict, original_name:str, new_name:str) -> dict:

	face_dict[new_name] = face_dict.pop(original_name)

	return face_dict
