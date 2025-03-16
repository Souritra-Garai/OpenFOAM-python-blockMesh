import numpy as np

from .hex import Hex, hex_vertices_connectivity, getAxisAndOrientation

# Refer to Figure 5.3
# https://doc.cfd.direct/openfoam/user-guide-v12/blockmesh#x28-1620005.4.9
# for the hex vertices numbering

# Vertices that form a face of a hex
# Order of vertices is such that the normal points outwards
# Moreover, axis 0 cross axis 1 points outwards
# Axis 0 : vertex 0 -> vertex 1
# Axis 1 : vertex 1 -> vertex 2
hex_face_vertices = (
	(3, 0, 4, 7),
	(1, 2, 6, 5),
	(5, 4, 0, 1),
	(2, 3, 7, 6),
	(1, 0, 3, 2),
	(4, 5, 6, 7)
)

# Class to represent a collection of hexahedral blocks
# Two hexahedral blocks are connected if they share a face
# The face is defined by the vertices of the hexahedral blocks
class ConnectedHexCollection :

	def __init__(self) -> None:
		
		# List of hexahedral blocks
		self.collection_hex		= []
		
		# Connectivity information
		# Axis 0 represents a connected pair of hexahedral blocks
		# Axis 1 contain the indices of the connected hexahedral blocks
		self.connectivity		= np.zeros((0, 2), dtype=int)
		# Axis 2 contains the mapping of the vertices of the connected hexahedral blocks
		self.connectivity_vertex_map	= np.zeros((0, 2, 4), dtype=int)

		pass

	def addHex(self, hex:Hex) -> int:
		'''
		Add a hexahedral block to the collection
		Return the index of the new hexahedral block in the collection
		'''
		
		self.collection_hex.append(hex)

		return len(self.collection_hex) - 1

	def isHexFaceConnected(self, hex:int, vertices_hex:list[int, int, int, int]) -> bool:
		'''
		Check if the hexahedral block already shares the face defined by the vertices
		with another hexahedral block
		Return True if the face is already connected
		'''
		connected = False
		
		# Find indices of elements where hex is present
		# np.where returns a tuple of arrays of indices such that
		# self.connectivity[indices] == hex
		indices = np.where(self.connectivity == hex)

		# Similarly, self.connectivity_vertex_map[indices]
		# contains the vertices of the face that is connected
		for vertices in self.connectivity_vertex_map[indices] :
			
			if set(vertices) == set(vertices_hex):
				
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
		'''
		Connect two hexahedral blocks such that they share a face
		Vertices of the face are defined by vertices_hex_1 and vertices_hex_2
		'''
		
		# Verify that the hexahedral blocks are not already connected
		# Verify that the face shapes match
		# Verify that the face coordinates match
		# Verify that the hexahedral blocks are not the same
		# Verify that the hexahedral blocks are not connected to themselves
		assert hex_1 != hex_2, 'Hex 1 and Hex 2 are the same'
		assert self.isHexFaceConnected(hex_1, vertices_hex_1) == False, 'Hex 1 is already connected'
		assert self.isHexFaceConnected(hex_2, vertices_hex_2) == False, 'Hex 2 is already connected'

		assert self.collection_hex[hex_1].getFaceShape(vertices_hex_1) == self.collection_hex[hex_2].getFaceShape(vertices_hex_2), 'Face shapes do not match'

		assert np.isclose(self.collection_hex[hex_1].getFacePointsCoordinates(vertices_hex_1), self.collection_hex[hex_2].getFacePointsCoordinates(vertices_hex_2)).all(), 'Face coordinates do not match'

		# Use the hex with the lower index as the first hex
		hex_pair	= [hex_1, hex_2]
		vertices_pair	= [vertices_hex_1, vertices_hex_2]

		if hex_1 > hex_2 :

			hex_pair	= [hex_2, hex_1]
			vertices_pair	= [vertices_hex_2, vertices_hex_1]

		# Create a mapping of the vertices of the connected hexahedral blocks
		vertex_map	= np.zeros((2, 4), dtype=int)
		# Find the index of the face in hex_face_vertices
		# that matches the vertices of the first hexahedral block
		i = [set(face_vertices) for face_vertices in hex_face_vertices].index(set(vertices_pair[0]))
		# For the first hexahedral block, the vertices are the ordered
		# according to the order in 'hex_face_vertices'
		# For the second hexahedral block, the vertices are mapped
		# to the corresponding vertices of the first hexahedral block
		vertex_map[0] = hex_face_vertices[i]
		vertex_map[1] = [vertices_pair[1][vertices_pair[0].index(vertex)] for vertex in hex_face_vertices[i]]

		self.connectivity		= np.append(self.connectivity, [hex_pair], axis=0)
		self.connectivity_vertex_map	= np.append(self.connectivity_vertex_map, [vertex_map], axis=0)
		
		pass

	def assignCellIndices(self, cell_index:int=0) -> None:
		'''
		Assign cell indices to the hexahedral blocks
		Return the next available cell index
		'''
		
		for hex in self.collection_hex :

			cell_index = hex.setCellIndices(cell_index)

		return cell_index

	def assignVertexIndices(self, point_index:int=0) -> None:
		'''
		Assign point indices to the vertices of hexahedral blocks
		Return the next available point index
		'''
		
		# First assign indices to the vertices that are shared by the hexahedral blocks
		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			# For each of the 4 vertices of the shared face
			# Assign the same point index to the corresponding vertices
			# of the connected hexahedral blocks
			for i in range(4) :

				# If the vertex in hexahedral block 0 does not have an index
				if self.collection_hex[hex_pair[0]].getVertexPointIndex(vertex_map[0, i]) == -1 :

					# If the vertex in hexahedral block 1 does not have an index
					if self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]) == -1 :

						# Assign a new point index to the vertex in both hexahedral blocks
						self.collection_hex[hex_pair[0]].setVertexPointIndex(vertex_map[0, i], point_index)
						self.collection_hex[hex_pair[1]].setVertexPointIndex(vertex_map[1, i], point_index)

						point_index += 1

					# If the vertex in hexahedral block 1 has an index
					# Assign the same index to the vertex in hexahedral block 0
					else :

						self.collection_hex[hex_pair[0]].setVertexPointIndex(vertex_map[0, i], self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]))

				# If the vertex in hexahedral block 0 has an index
				else :
					
					# If the vertex in hexahedral block 1 does not have an index
					# Assign the same index to the vertex in hexahedral block 1
					if self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]) == -1 :

						self.collection_hex[hex_pair[1]].setVertexPointIndex(vertex_map[1, i], self.collection_hex[hex_pair[0]].getVertexPointIndex(vertex_map[0, i]))

					# If the vertex in hexahedral block 1 has an index
					# Verify that the indices match
					else :

						assert self.collection_hex[hex_pair[0]].getVertexPointIndex(vertex_map[0, i]) == self.collection_hex[hex_pair[1]].getVertexPointIndex(vertex_map[1, i]), 'Vertex indices do not match'

		# Assign indices to the vertices that are not shared by the hexahedral blocks
		for hex in self.collection_hex :

			# For each of the 8 vertices of the hexahedral block
			for i in range(8) :

				# Assign a new point index if the vertex does not have an index
				if hex.getVertexPointIndex(i) == -1 :

					hex.setVertexPointIndex(i, point_index)
					point_index += 1

		return point_index
	
	def assignEdgePointsIndices(self, point_index:int=0) -> None:
		'''
		Assign indices to the points on the edges of the hexahedral blocks
		Return the next available point index
		'''

		# Assign indices to the points on the edges that are shared by the hexahedral blocks
		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			try :
			
				# For each of the 4 edges of the shared face
				# Assign the same point indices to the corresponding edges
				# of the connected hexahedral blocks
				for i in range(4) :

					# i -> (i+1)%4 Since there are 4 vertices
					# For the last vertex, the next vertex is the 0th vertex

					# If all the points on the edge in hexahedral block 0 do not have an index
					if np.all(self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]) == -1) :

						# If all the points on the edge in hexahedral block 1 do not have an index
						if np.all(self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4]) == -1) :

							edge_axis, _ = getAxisAndOrientation(vertex_map[0, i], vertex_map[0, (i+1)%4])
							num_new_points = self.collection_hex[hex_pair[0]].point_indices.shape[edge_axis] - 2 # Exclude the vertices 

							new_indices = np.arange(point_index, point_index + num_new_points)

							self.collection_hex[hex_pair[0]].setEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4], new_indices)
							self.collection_hex[hex_pair[1]].setEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4], new_indices)

							point_index += num_new_points

						# If all the points on the edge in hexahedral block 1 have an index
						else :

							# Assign the same indices to the points on the edge in hexahedral block 0
							self.collection_hex[hex_pair[0]].setEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4], self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4]))

					else :
						# If all the points on the edge in hexahedral block 1 do not have an index
						if np.all(self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4]) == -1) :

							# Assign the same indices to the points on the edge in hexahedral block 1
							self.collection_hex[hex_pair[1]].setEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4], self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]))

						# If all the points on the edge in hexahedral block 1 have an index
						else :
							
							# Verify that the indices match
							assert np.all(self.collection_hex[hex_pair[0]].getEdgePointsIndices(vertex_map[0, i], vertex_map[0, (i+1)%4]) == self.collection_hex[hex_pair[1]].getEdgePointsIndices(vertex_map[1, i], vertex_map[1, (i+1)%4])), 'Edge indices do not match'

			except Exception as e :

				print(hex_pair)
				print(vertex_map)
				raise e

		# Assign indices to the points on the edges that are not shared by the hexahedral blocks			
		for hex in self.collection_hex :

			for edge in hex_vertices_connectivity.keys() :

				# If all the points on the edge do not have an index
				if np.all(hex.getEdgePointsIndices(edge[0], edge[1]) == -1) :

					edge_axis, _ = getAxisAndOrientation(edge[0], edge[1])
					num_new_points = hex.point_indices.shape[edge_axis] - 2 # Exclude the vertices

					new_indices = np.arange(point_index, point_index + num_new_points)

					hex.setEdgePointsIndices(edge[0], edge[1], new_indices)

					point_index += num_new_points
		
		return point_index

	def assignFacePointsIndices(self, point_index:int=0) -> None:
		'''
		Assign indices to the points on the faces of the hexahedral blocks
		Return the next available point index
		'''

		# Assign indices to the points on the faces that are shared by the hexahedral blocks
		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			# If all the points on the face in hexahedral block 0 do not have an index
			if np.all(self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]) == -1) :

				# If all the points on the face in hexahedral block 1 do not have an index
				if np.all(self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1]) == -1) :

					axis_0, _ = getAxisAndOrientation(vertex_map[0, 0], vertex_map[0, 1])
					axis_1, _ = getAxisAndOrientation(vertex_map[0, 1], vertex_map[0, 2])

					shape = (
						self.collection_hex[hex_pair[0]].point_indices.shape[axis_0] - 2,
						self.collection_hex[hex_pair[0]].point_indices.shape[axis_1] - 2
					)

					num_new_points = np.prod(shape)

					# Points are increasing faster along axis 0
					new_indices = np.arange(point_index, point_index + num_new_points).reshape(shape, order='F')

					# Assign the new indices to the points on the face in both hexahedral blocks
					self.collection_hex[hex_pair[0]].setFacePointsIndices(vertex_map[0], new_indices)
					self.collection_hex[hex_pair[1]].setFacePointsIndices(vertex_map[1], new_indices)

					point_index += num_new_points

				# If all the points on the face in hexahedral block 1 have an index
				else :

					# Assign the same indices to the points on the face in hexahedral block 0
					self.collection_hex[hex_pair[0]].setFacePointsIndices(vertex_map[0], self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1]))

			# If all the points on the face in hexahedral block 0 have an index
			else :
				
				# If all the points on the face in hexahedral block 1 do not have an index
				if np.all(self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1]) == -1) :

					# Assign the same indices to the points on the face in hexahedral block 1
					self.collection_hex[hex_pair[1]].setFacePointsIndices(vertex_map[1], self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]))

				# If all the points on the face in hexahedral block 1 have an index
				else :

					# Verify that the indices match
					assert np.all(self.collection_hex[hex_pair[0]].getFacePointsIndices(vertex_map[0]) == self.collection_hex[hex_pair[1]].getFacePointsIndices(vertex_map[1])), 'Face indices do not match'

		# Assign indices to the points on the faces that are not shared by the hexahedral blocks
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
		'''
		Assign indices to the points inside the hexahedral blocks
		Return the next available point index
		'''

		for hex in self.collection_hex :

			point_index = hex.setInternalPointIndices(point_index)

		return point_index

	def getFaces(self) -> dict :
		'''
		Returns the faces of the hexahedral blocks
		Internal faces are stored in the 'internal' key
		External faces are stored in the 'h{i}_f{j}' key
		'''

		# Initialize the faces dictionary
		faces = {'internal':{'vertices':np.zeros((0, 4), dtype=int), 'owner':np.zeros(0, dtype=int), 'neighbour':np.zeros(0, dtype=int)}}

		# List to store the faces of the hexahedral blocks
		# that have already been added to the faces dictionary
		# For each hexahedral block, store the set of vertices
		# of the faces that have been added
		completed_hex_faces = [[] for _ in range(len(self.collection_hex))]

		# Add the shared faces of the hexahedral blocks that are connected
		# to the faces dictionary
		for hex_pair, vertex_map in zip(self.connectivity, self.connectivity_vertex_map) :

			# getFace returns the vertices of the face and the owner
			# vertices are ordered according to vertex_map
			vertices, owner		= self.collection_hex[hex_pair[0]].getFace(vertex_map[0])
			vertices_, neighbour	= self.collection_hex[hex_pair[1]].getFace(vertex_map[1])
			
			# Vertices should ordered such that the normal points from hex_pair[0] to hex_pair[1]
			# vertex_map[0] is ordered such that the normal points outwards
			# vertex_map[1] is ordered such that the normal points inwards
			
			# Verify that the face vertices match
			assert np.all(vertices == vertices_), 'Face vertices do not match'

			faces['internal']['vertices']	= np.append(faces['internal']['vertices'], vertices, axis=0)
			faces['internal']['owner']	= np.append(faces['internal']['owner'], owner)
			faces['internal']['neighbour']	= np.append(faces['internal']['neighbour'], neighbour)

			# Mark the faces as added
			completed_hex_faces[hex_pair[0]].append(set(vertex_map[0]))
			completed_hex_faces[hex_pair[1]].append(set(vertex_map[1]))

		# Add the remaining faces of the hexahedral blocks that
		for i, hex in enumerate(self.collection_hex) :

			# For each of the 6 faces of the hexahedral block, 
			# add the face if it is a boundary face
			for j, face_vertices in enumerate(hex_face_vertices) :

				# If the face has not been added already
				if set(face_vertices) not in completed_hex_faces[i] :

					# getFace returns the vertices of the face and the owner
					vertices, owner = hex.getFace(face_vertices)

					# Again vertices should be ordered such that the normal points outwards
					# face_vertices is ordered such that the normal points outwards

					faces[f'h{i}_f{j}'] = {'vertices':vertices, 'owner':owner}

			# Add the internal faces of the hexahedral blocks
			vertices, owner, neighbour = hex.getInternalFaces()

			faces['internal']['vertices']	= np.append(faces['internal']['vertices'], vertices, axis=0)
			faces['internal']['owner']	= np.append(faces['internal']['owner'], owner)
			faces['internal']['neighbour']	= np.append(faces['internal']['neighbour'], neighbour)

		return faces

	def getPoints(self) -> np.ndarray :
		'''
		Return the coordinates of the points
		'''

		# Find the number of points
		# The maximum point index is the number of points - 1
		num_points = max([hex.point_indices.max() for hex in self.collection_hex]) + 1

		points = np.zeros((num_points, 3))

		# Assign the coordinates of the points
		# Coordinates are ordered according to the point indices
		# Points that are shared by the hexahedral blocks
		# have the same coordinates and the same index
		# So overwriting the coordinates is not an issue
		for hex in self.collection_hex :

			points[hex.point_indices.flatten(order='F')] = np.stack((
				hex.point_coordinates[..., 0].flatten(order='F'),
				hex.point_coordinates[..., 1].flatten(order='F'),
				hex.point_coordinates[..., 2].flatten(order='F')
			), axis=-1)

		return points