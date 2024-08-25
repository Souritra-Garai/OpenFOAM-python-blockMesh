import numpy as np

from .hex import Hex

hex_face_vertices = (
	(3, 0, 4, 7),
	(1, 2, 6, 5),
	(5, 4, 0, 1),
	(2, 3, 7, 6),
	(1, 0, 3, 2),
	(4, 5, 6, 7)
)

hex_edge_vertices = (
	(0, 1),
	(3, 2),
	(7, 6),
	(4, 5),
	(0, 3),
	(1, 2),
	(5, 6),
	(4, 7),
	(0, 4),
	(1, 5),
	(2, 6),
	(3, 7)
)

class ConnectedHexCollection :

	def __init__(self) -> None:
		
		self.collection_hex			= []
		
		self.connectivity				= np.zeros((0, 2), dtype=int)
		self.connectivity_vertex_map	= np.zeros((0, 2, 4), dtype=int)

		pass

	def addHex(self, hex:Hex) -> None:
		
		self.collection_hex.append(hex)

		pass

	def checkConnected(self, hex_1:int, vertices_hex_1:list[int, int, int, int]) -> bool:
		
		# Check if face is previously connected
		connected = False
		
		# Find indices of elements where hex_1 is present
		indices = np.where(self.connectivity == hex_1)

		# Check if the set of vertices is already connected
		for vertices in self.connectivity_vertex_map[indices] :
			
			if set(vertices) == set(vertices_hex_1):
				
				connected = True
				break

		return connected

	def connectHex(
		self,
		hex_1:int,
		hex_2:int,
		vertices_hex_1:list[int, int, int, int],
		vertices_hex_2:list[int, int, int, int]
	) -> None :
		
		assert self.checkConnected(hex_1, vertices_hex_1) == False, 'Hex 1 is already connected'
		assert self.checkConnected(hex_2, vertices_hex_2) == False, 'Hex 2 is already connected'

		assert self.collection_hex[hex_1].getFaceShape(vertices_hex_1) == self.collection_hex[hex_2].getFaceShape(vertices_hex_2), 'Face shapes do not match'

		assert np.isclose(self.collection_hex[hex_1].getFacePointsCoordinates(vertices_hex_1), self.collection_hex[hex_2].getFacePointsCoordinates(vertices_hex_2)).all(), 'Face coordinates do not match'

		hex_pair		= [hex_1, hex_2]
		vertices_pair	= [vertices_hex_1, vertices_hex_2]

		if hex_1 > hex_2 :

			hex_pair		= [hex_2, hex_1]
			vertices_pair	= [vertices_hex_2, vertices_hex_1]

		vertex_map	= np.zeros((2, 4), dtype=int)

		i = [set(face_vertices) for face_vertices in hex_face_vertices].index(set(vertices_pair[0]))

		vertex_map[0] = hex_face_vertices[i]
		vertex_map[1] = [vertices_pair[1][vertices_pair[0].index(vertex)] for vertex in hex_face_vertices[i]]

		self.connectivity				= np.append(self.connectivity, [hex_pair], axis=0)
		self.connectivity_vertex_map	= np.append(self.connectivity_vertex_map, [vertex_map], axis=0)
		
		pass

	def assignCellIndices(self, cell_index:int=0) -> None:
		
		for hex in self.collection_hex :

			cell_index = hex.setCellIndices(cell_index)

		return cell_index

	def assignVertexIndices(self, point_index:int=0) -> None:
		
		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			for i in range(4) :

				if self.collection_hex[hex_pair[0]].getVertexPointIndex(vertex_map[0, i]) == -1 :

					if self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]) == -1 :

						self.collection_hex[hex_pair[0]].setVertexPointIndex(vertex_map[0, i], point_index)
						self.collection_hex[hex_pair[1]].setVertexPointIndex(vertex_map[1, i], point_index)

						point_index += 1

					else :

						self.collection_hex[hex_pair[0]].setVertexPointIndex(vertex_map[0, i], self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]))

				else :
					
					if self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]) == -1 :

						self.collection_hex[hex_pair[1]].setVertexPointIndex(vertex_map[1, i], self.collection_hex[hex_pair[0]].getVertexPointIndex(vertex_map[0, i]))

					else :

						assert self.collection_hex[hex_pair[0]].getVertexPointIndex(vertex_map[0, i]) == self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]), 'Vertex indices do not match'

		for hex in self.collection_hex :

			for i in range(8) :

				if hex.getVertexPointIndex(i) == -1 :

					hex.setVertexPointIndex(i, point_index)
					point_index += 1

		return point_index
	
	def assignEdgePointsIndices(self, point_index:int=0) -> None:

		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			try :
			
				for i in range(4) :

					if np.all(self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]) == -1) :

						if np.all(self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4]) == -1) :

							num_new_points = self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]).shape[0]

							new_indices = np.arange(point_index, point_index + num_new_points)

							self.collection_hex[hex_pair[0]].setEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4], new_indices)
							self.collection_hex[hex_pair[1]].setEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4], new_indices)

							point_index += num_new_points

						else :

							self.collection_hex[hex_pair[0]].setEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4], self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4]))

					else :
						
						if np.all(self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4]) == -1) :

							self.collection_hex[hex_pair[1]].setEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4], self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]))

						else :

							assert np.all(self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]) == self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4])), 'Edge indices do not match'

			except Exception as e :

				print(hex_pair)
				print(vertex_map)
				raise e
			
		for hex in self.collection_hex :

			for edge in hex_edge_vertices :

				if np.all(hex.getEdgePointsIndices(edge[0], edge[1]) == -1) :

					num_new_points = hex.getEdgePointsIndices(edge[0], edge[1]).shape[0]

					new_indices = np.arange(point_index, point_index + num_new_points)

					hex.setEdgePointsIndices(edge[0], edge[1], new_indices)

					point_index += num_new_points
		
		return point_index

	def assignFacePointsIndices(self, point_index:int=0) -> None:

		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			if np.all(self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]) == -1) :

				if np.all(self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1]) == -1) :

					shape = self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]).shape

					num_new_points = np.prod(shape)

					new_indices = np.arange(point_index, point_index + num_new_points).reshape(shape, order='F')

					self.collection_hex[hex_pair[0]].setFacePointsIndices(vertex_map[0], new_indices)
					self.collection_hex[hex_pair[1]].setFacePointsIndices(vertex_map[1], new_indices)

					point_index += num_new_points

				else :

					self.collection_hex[hex_pair[0]].setFacePointsIndices(vertex_map[0], self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1]))

			else :
				
				if np.all(self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1]) == -1) :

					self.collection_hex[hex_pair[1]].setFacePointsIndices(vertex_map[1], self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]))

				else :

					assert np.all(self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]) == self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1])), 'Face indices do not match'

		for hex in self.collection_hex :

			for vertices in hex_face_vertices :

				if np.all(hex.getFacePointsIndices(tuple(vertices)) == -1) :

					shape = hex.getFacePointsIndices(vertices).shape

					num_new_points = np.prod(shape)

					new_indices = np.arange(point_index, point_index + num_new_points).reshape(shape, order='F')

					hex.setFacePointsIndices(vertices, new_indices)

					point_index += num_new_points

		return point_index

	def assignInternalIndices(self, point_index:int=0) -> None:

		for hex in self.collection_hex :

			point_index = hex.setInternalPointIndices(point_index)

		return point_index

	def getFaces(self) -> dict :

		faces = {'internal':{'vertices':np.zeros((0, 4), dtype=int), 'owner':np.zeros(0, dtype=int), 'neighbour':np.zeros(0, dtype=int)}}

		completed_hex_faces = [[] for _ in range(len(self.collection_hex))]

		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			vertices, owner			= self.collection_hex[hex_pair[0]].getFace(vertex_map[0])
			vertices_, neighbour	= self.collection_hex[hex_pair[1]].getFace(vertex_map[1])

			assert np.all(vertices == vertices_), 'Face vertices do not match'

			faces['internal']['vertices']	= np.append(faces['internal']['vertices'], vertices, axis=0)
			faces['internal']['owner']		= np.append(faces['internal']['owner'], owner)
			faces['internal']['neighbour']	= np.append(faces['internal']['neighbour'], neighbour)

			completed_hex_faces[hex_pair[0]].append(set(vertex_map[0]))
			completed_hex_faces[hex_pair[1]].append(set(vertex_map[1]))

		for i, hex in enumerate(self.collection_hex) :

			for j, face_vertices in enumerate(hex_face_vertices) :

				if set(face_vertices) not in completed_hex_faces[i] :

					vertices, owner = hex.getFace(face_vertices)

					faces[f'h{i}_f{j}'] = {'vertices':vertices, 'owner':owner}

			vertices, owner, neighbour = hex.getInternalFaces()

			faces['internal']['vertices']	= np.append(faces['internal']['vertices'], vertices, axis=0)
			faces['internal']['owner']		= np.append(faces['internal']['owner'], owner)
			faces['internal']['neighbour']	= np.append(faces['internal']['neighbour'], neighbour)

		return faces

	def getPoints(self) -> np.ndarray :

		num_points = max([hex.point_indices.max() for hex in self.collection_hex]) + 1

		points = np.zeros((num_points, 3))

		for hex in self.collection_hex :

			points[hex.point_indices.flatten(order='F')] = np.stack((
				hex.point_coordinates[..., 0].flatten(order='F'),
				hex.point_coordinates[..., 1].flatten(order='F'),
				hex.point_coordinates[..., 2].flatten(order='F')
			), axis=-1)

		return points