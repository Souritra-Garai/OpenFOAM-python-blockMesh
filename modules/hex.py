import numpy as np

# Refer to Figure 5.3
# https://doc.cfd.direct/openfoam/user-guide-v12/blockmesh#x28-1620005.4.9
# for the hex vertices numbering

#  0 represents start of an array
# -1 represents end of an array
# For a (n1, n2, n3, ...) np.array named A,
# A[hex_vertices_index[i]] represents the vertex i
hex_vertices_index = (
	( 0,  0,  0),
	(-1,  0,  0),
	(-1, -1,  0),
	( 0, -1,  0),
	( 0,  0, -1),
	(-1,  0, -1),
	(-1, -1, -1),
	( 0, -1, -1)
)

# Dictionary tells which vertices are connected to each other
# along which axis
# 0: x-axis, 1: y-axis, 2: z-axis
hex_vertices_connectivity = {
	(0, 1) : 0,
	(1, 2) : 1,
	(3, 2) : 0,
	(0, 3) : 1,
	(4, 5) : 0,
	(5, 6) : 1,
	(7, 6) : 0,
	(4, 7) : 1,
	(0, 4) : 2,
	(1, 5) : 2,
	(2, 6) : 2,
	(3, 7) : 2	
}

def getAxisAndOrientation(vertex_1:int, vertex_2:int) -> tuple[int, int] :
	'''
	Orientation is +1 if vertex_2 is in the positive direction of vertex_1
	along the corresponding axis, -1 otherwise
	'''

	if	(vertex_1, vertex_2) in hex_vertices_connectivity.keys() :

		return hex_vertices_connectivity[(vertex_1, vertex_2)], 1
	
	elif	(vertex_2, vertex_1) in hex_vertices_connectivity.keys() :

		return hex_vertices_connectivity[(vertex_2, vertex_1)], -1
	
	else :	raise ValueError('Vertices are not connected')

def verticesShareFaceAlongAxis(vertices:list[int], axis:int) -> bool :
	'''
	Checks if the vertices belong to a hex face
	that is normal to the axis
	'''

	return len(set([hex_vertices_index[vertex][axis] for vertex in vertices])) == 1

def getEdgePointsSlice(vertex_1:int, vertex_2:int) -> slice :
	'''
	Returns the slice of the 3D array that contains the points
	which are connected by the edge formed by the vertices
	excluding the vertices themselves
	'''

	axis, orientation = getAxisAndOrientation(vertex_1, vertex_2)

	# The axes that are not the edge axis
	# e.g. if the edge is along the x-axis,
	# then const_axes are y and z
	const_axes = {0, 1, 2} - {axis}

	slice_3d = [None, None, None]

	# The slice along the edge axis, excluding the vertices
	# Always in the positive direction
	slice_3d[axis] = slice(1, -1) if orientation == 1 else slice(-2, 0, -1)

	for const_axis in const_axes :

		# We know vertices form an edge
		assert verticesShareFaceAlongAxis([vertex_1, vertex_2], const_axis), f'Vertices {vertex_1} and {vertex_2} do not share a face along axis {const_axis}'

		slice_3d[const_axis] = hex_vertices_index[vertex_1][const_axis]

	return tuple(slice_3d)

def getThirdAxis(axis_1:int, axis_2:int) -> int :
	'''
	Returns the axis that is not axis_1 or axis_2
	'''

	return ({0, 1, 2} - {axis_1, axis_2}).pop()

def getFacePointsSlice(vertices:tuple[int, int, int, int]) -> slice :
	'''
	Returns the slice of the 3D array that represents
	the points on the face formed by the vertices.
	Excludes the vertices and edges of the face
	'''
	
	# For vertices [0, 1, 2, 3]
	# axis_1 = 0, axis_2 = 1
	# orientation_1 = 1, orientation_2 = 1
	axis_1, orientation_1 = getAxisAndOrientation(vertices[0], vertices[1])
	axis_2, orientation_2 = getAxisAndOrientation(vertices[1], vertices[2])
	# const_axis = 2
	const_axis = getThirdAxis(axis_1, axis_2)

	slice_3d = [None, None, None]

	slice_3d[axis_1] = slice(1, -1) if orientation_1 == 1 else slice(-2, 0, -1)
	slice_3d[axis_2] = slice(1, -1) if orientation_2 == 1 else slice(-2, 0, -1)

	assert verticesShareFaceAlongAxis(vertices, const_axis), f'Vertices {vertices} do not share a face along axis {const_axis}'

	slice_3d[const_axis] = hex_vertices_index[vertices[0]][const_axis]

	return tuple(slice_3d)

def getCustomSlice3D(axes_full:tuple[int, int], orientations:tuple[int, int], axis_custom:int, slice_custom:slice) -> slice :
	'''
	Returns a tuple of slices of length 3.
	axes_full: tuple of 2 axes along which complete slices are returned
	orientations: tuple of 2 orientations {1, -1}, representing the direction of the axes in axes_full
	axis_custom: the axis along which the custom slice is returned
	slice_custom: the custom slice to be returned along axis_custom
	'''

	slice_3d = [None, None, None]

	slice_3d[axes_full[0]] = slice(None, None, orientations[0])
	slice_3d[axes_full[1]] = slice(None, None, orientations[1])

	slice_3d[axis_custom] = slice_custom

	return tuple(slice_3d)

def getFaceOwners(cell_indices:np.ndarray, axes:tuple[int, int], orientations:tuple[int, int], axis:int) -> np.ndarray :
	'''
	Returns a slice of cell indices that represent
	face owners along positive direction of the axis.
	For example, if the face is along the x-axis,
	then returns cell_indices[:n-1, :, :]
	where n is the number of cells along the x-axis.
	'''
	return cell_indices[getCustomSlice3D(axes, orientations, axis, slice(0, -1))]

def getFaceNeighbours(cell_indices:np.ndarray, axes:tuple[int, int], orientations:tuple[int, int], axis:int) -> np.ndarray :
	'''
	Returns a slice of cell indices that represent
	face neighbours along positive direction of the axis.
	For example, if the face is along the x-axis,
	then returns cell_indices[1:, :, :]
	where n is the number of cells along the x-axis.
	'''
	return cell_indices[getCustomSlice3D(axes, orientations, axis, slice(1, None))]

def getFaceVertex1(points:np.ndarray, axis_1:int, axis_2:int, axis_face_normal:int, slice_face_normal:slice) -> int :
	'''
	Returns points representing vertex 1 of the face.
	axis_face_normal: the axis along which the face is normal
	slice_face_normal: the slice along the axis_face_normal
	axis_1, axis_2: Axes parallel to the face
	points: 3D array of point_ids
	Vertex 1 is the vertex with the lowest indices along axis_1 and axis_2
	For example, if axis_1 = 0, axis_2 = 1, then vertex 1 is the vertex with
	the lowest x and y indices
	'''
	slices_3d = [None, None, None]

	slices_3d[axis_face_normal] = slice_face_normal

	slices_3d[axis_1] = slice(0, -1)
	slices_3d[axis_2] = slice(0, -1)

	return points[tuple(slices_3d)]

def getFaceVertex2(points:np.ndarray, axis_1:int, axis_2:int, axis_face_normal:int, slice_face_normal:slice) -> int :
	'''
	Returns points representing vertex 2 of the face.
	axis_face_normal: the axis along which the face is normal
	slice_face_normal: the slice along the axis_face_normal
	axis_1, axis_2: Axes parallel to the face
	points: 3D array of point_ids
	Vertex 2 is the vertex with the highest index along axis_1 and the lowest index along axis_2
	'''
	slices_3d = [None, None, None]

	slices_3d[axis_face_normal] = slice_face_normal

	slices_3d[axis_1] = slice(1, None)
	slices_3d[axis_2] = slice(0, -1)

	return points[tuple(slices_3d)]

def getFaceVertex3(points:np.ndarray, axis_1:int, axis_2:int, axis_face_normal:int, slice_face_normal:slice) -> int :
	'''
	Returns points representing vertex 3 of the face.
	axis_face_normal: the axis along which the face is normal
	slice_face_normal: the slice along the axis_face_normal
	axis_1, axis_2: Axes parallel to the face
	points: 3D array of point_ids
	Vertex 3 is the vertex with the highest indices along axis_1 and axis_2
	'''
	slices_3d = [None, None, None]

	slices_3d[axis_face_normal] = slice_face_normal

	slices_3d[axis_1] = slice(1, None)
	slices_3d[axis_2] = slice(1, None)

	return points[tuple(slices_3d)]

def getFaceVertex4(points:np.ndarray, axis_1:int, axis_2:int, axis_face_normal:int, slice_face_normal:slice) -> int :
	'''
	Returns points representing vertex 4 of the face.
	axis_face_normal: the axis along which the face is normal
	slice_face_normal: the slice along the axis_face_normal
	axis_1, axis_2: Axes parallel to the face
	points: 3D array of point_ids
	Vertex 4 is the vertex with the lowest index along axis_1 and the highest index along axis_2
	'''
	slices_3d = [None, None, None]

	slices_3d[axis_face_normal] = slice_face_normal

	slices_3d[axis_1] = slice(0, -1)
	slices_3d[axis_2] = slice(1, None)

	return points[tuple(slices_3d)]


class Hex :
	
	def __init__(self, cell_shape:tuple[int, int, int]) -> None :
		'''
		cell_shape: tuple of number of cells / divisions along the 3 axes
		'''

		point_shape = tuple([i+1 for i in cell_shape])
		
		# Allocate memory for cell and point indices
		self.cell_indices	= np.zeros(cell_shape, dtype=int)
		self.point_indices	= np.zeros(point_shape, dtype=int)
		self.point_coordinates	= np.zeros(point_shape + (3,), dtype=float)

		# Fill the arrays with -1 and np.nan
		self.cell_indices.fill(-1)
		self.point_indices.fill(-1)
		self.point_coordinates.fill(np.nan)
		
		pass

	def setCellIndices(self, cell_index:int=0) -> int :
		'''
		Fills the cell indices with consecutive numbers
		starting from cell_index
		Returns the next cell index
		'''
		
		num_cells = np.prod(self.cell_indices.shape)
		
		# Fill the cell indices with consecutive numbers
		# starting from cell_index
		# 'F' ordering: Cell indices along axis 0 change the fastest
		# followed by axis 1 and then axis 2
		# i.e. cell_indices[0, 0, 0], cell_indices[1, 0, 0], cell_indices[2, 0, 0], ...
		self.cell_indices = np.arange(
			cell_index,
			cell_index + num_cells
		).reshape(self.cell_indices.shape, order='F')
		
		return cell_index + num_cells
	
	def setInternalPointIndices(self, point_index:int=0) -> int :
		'''
		Fills the internal points with consecutive numbers
		starting from point_index
		Internal points are the points that are not on the boundary
		[1:-1, 1:-1, 1:-1]
		Returns the next point index
		'''

		# Subtract 2 from each axis to exclude the boundary points
		shape_internal = tuple([i-2 for i in self.point_indices.shape])
		
		num_internal_points = np.prod(shape_internal)
		
		# Fill the internal points with consecutive numbers
		# starting from point_index
		# 'F' ordering: Point indices along axis 0 change the fastest
		# followed by axis 1 and then axis 2
		# i.e. point_indices[1, 1, 1], point_indices[2, 1, 1], point_indices[3, 1, 1], ...
		self.point_indices[1:-1, 1:-1, 1:-1] = np.arange(
			point_index, 
			point_index + num_internal_points
		).reshape(shape_internal, order='F')
		
		return point_index + num_internal_points
	
	def getFaceShape(self, vertex_indices:tuple[int, int, int, int]) -> tuple[int, int] :
		'''
		Returns the shape / number of divisions along each axis of the face
		formed by the vertices given by vertex_indices
		vertex_indices: tuple of 4 vertex indices that form a face
		(Order of vertices is important)
		'''
		
		axes_1, orientation_1 = getAxisAndOrientation(vertex_indices[0], vertex_indices[1])
		axes_2, orientation_2 = getAxisAndOrientation(vertex_indices[0], vertex_indices[3])

		return self.cell_indices.shape[axes_1], self.cell_indices.shape[axes_2]
	
	def getVertexPointIndex(self, vertex_num:int) -> int :
		'''
		Returns the point index of the hex vertex given by vertex_num
		'''
		return self.point_indices[hex_vertices_index[vertex_num]]
	
	def setVertexPointIndex(self, vertex_num:int, point_index:int) -> None :
		'''
		Sets the point index of the hex vertex given by vertex_num
		'''

		self.point_indices[hex_vertices_index[vertex_num]] = point_index

		pass
	
	def getEdgePointsIndices(self, vertex_1:int, vertex_2:int) -> np.ndarray :
		'''
		Returns the point indices of the points on the edge formed
		by the vertices vertex_1 and vertex_2
		Indices ordered in the direction from vertex_1 to vertex_2
		excluding the vertices themselves
		'''

		return self.point_indices[getEdgePointsSlice(vertex_1, vertex_2)]
		
	def setEdgePointsIndices(self, vertex_1:int, vertex_2:int, point_indices:np.ndarray) -> None :
		'''
		Sets the point indices of the points on the edge formed
		by the vertices vertex_1 and vertex_2
		Indices ordered in the direction from vertex_1 to vertex_2
		excluding the vertices themselves
		'''

		edge_point_indices = self.point_indices[getEdgePointsSlice(vertex_1, vertex_2)]

		assert edge_point_indices.shape == point_indices.shape, 'Point indices shape mismatch'
		
		edge_point_indices = point_indices

		pass
	
	def getFacePointsIndices(self, vertices:tuple[int, int, int, int]) -> np.ndarray :
		'''
		Returns the point indices of the points on the face formed
		by the vertices.
		Excludes the vertices and the edges of the face
		'''
		return self.point_indices[getFacePointsSlice(vertices)]
		
	def setFacePointsIndices(self, vertices:tuple[int, int, int, int], point_indices:np.ndarray) -> None :
		'''
		Sets the point indices of the points on the face formed
		by the vertices.
		Excludes the vertices and the edges of the face
		'''

		face_points_indices = self.point_indices[getFacePointsSlice(vertices)]

		assert face_points_indices.shape == point_indices.shape, 'Point indices shape mismatch'

		face_points_indices = point_indices

		pass

	def getFace(self, vertices:tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray] :
		'''
		Returns the point indices and cell indices of the face
		formed by the vertices.
		'''

		# Say the face lies on the x-y plane
		# All points / cells will have the same z index
		# It will be the same as the z index of the first vertex
		# In this case, x and y are axes 1 and 2 respectively,
		# and z is the const_axis

		axis_1, orientation_1 = getAxisAndOrientation(vertices[0], vertices[1])
		axis_2, orientation_2 = getAxisAndOrientation(vertices[1], vertices[2])

		const_axis = getThirdAxis(axis_1, axis_2)

		assert verticesShareFaceAlongAxis(vertices, const_axis), 'Vertices do not belong to a hex face'
		const_axis_value = hex_vertices_index[vertices[0]][const_axis]

		face_slice = getCustomSlice3D(
			(axis_1, axis_2),
			(orientation_1, orientation_2),
			const_axis,
			const_axis_value
		)

		points	= self.point_indices[face_slice]
		cells	= self.cell_indices[face_slice].flatten(order='F')

		# Points are ordered such that normal to the face is in
		# cross_product(axis 1, axis 2) direction
		# The positive direction of axis 1 is from vertex 1 to vertex 2
		# The positive direction of axis 2 is from vertex 2 to vertex 3
		face_points_indices = np.stack((
			points[:-1, :-1].flatten(order='F'),
			points[ 1:, :-1].flatten(order='F'),
			points[ 1:,  1:].flatten(order='F'),
			points[:-1,  1:].flatten(order='F')
		), axis=-1)

		return face_points_indices, cells
	
	def getInternalFaces(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray] :
		'''
		Returns the point indices and owner and neighbour cell indices
		of the internal faces
		'''

		point_indices	= np.zeros((0, 4), dtype=int)

		cell_owner	= np.zeros(0, dtype=int)
		cell_neighbour	= np.zeros(0, dtype=int)

		# Iterates cyclically over the axes
		# For example, if the face is along the z-axis (axis 2),
		# then the axes are (0, 1) and const_axis is 2
		for const_axis, axes in enumerate(((1, 2), (2, 0), (0, 1))) :

			# Owner is all cells except the last one along the axis
			# Neighbour is all cells except the first one along the axis
			# For example, if the face is along the z-axis (axis 2),
			# then owner is cell_indices[:, :, :-1] and neighbour is cell_indices[:, :, 1:]
			owner		= getFaceOwners(self.cell_indices, axes, (1, 1), const_axis)
			neighbour	= getFaceNeighbours(self.cell_indices, axes, (1, 1), const_axis)
			
			# Vertices are ordered such that normal to the face is in
			# cross_product(axis 0, axis 1) direction
			# This should match the direction from owner to neighbour
			point_indices_0 = getFaceVertex1(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))
			point_indices_1 = getFaceVertex2(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))
			point_indices_2 = getFaceVertex3(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))
			point_indices_3 = getFaceVertex4(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))

			# Flatten the 3D arrays such that
			# We iterate fastest along the axis 0 ('F' ordering)
			cell_owner	= np.concatenate((cell_owner, owner.flatten(order='F')))
			cell_neighbour	= np.concatenate((cell_neighbour, neighbour.flatten(order='F')))

			point_indices_stacked = np.stack((
				point_indices_0.flatten(order='F'),
				point_indices_1.flatten(order='F'),
				point_indices_2.flatten(order='F'),
				point_indices_3.flatten(order='F')
			), axis=-1)

			point_indices = np.append(point_indices, point_indices_stacked, axis=0)

		return point_indices, cell_owner, cell_neighbour

	def getFacePointsCoordinates(self, vertices:tuple[int, int, int, int]) -> np.ndarray :
		'''
		Returns the coordinates of the points on the face formed
		by the vertices.
		Points are ordered in a 3D array such that
		Axis 0 is from vertex 0 to vertex 1
		Axis 1 is from vertex 1 to vertex 2
		Axis 2 is coordinate (x, y, z)
		'''

		axis_1, orientation_1 = getAxisAndOrientation(vertices[0], vertices[1])
		axis_2, orientation_2 = getAxisAndOrientation(vertices[1], vertices[2])

		const_axis = getThirdAxis(axis_1, axis_2)

		assert verticesShareFaceAlongAxis(vertices, const_axis), 'Vertices do not belong to a hex face'
		const_axis_value = hex_vertices_index[vertices[0]][const_axis]

		slice_3d = getCustomSlice3D(
			(axis_1, axis_2),
			(orientation_1, orientation_2),
			const_axis,
			const_axis_value
		)

		return self.point_coordinates[slice_3d]	

	def assignPointCoordinates(self, coordinates:np.ndarray, axes:tuple[int, int, int]=(0, 1, 2), orientations:tuple[int, int, int]=(1, 1, 1)) -> None :
		'''
		Assign coordinates to the points along the axes
		coordinates: 3D array of coordinates
		axes: tuple of axes along which the coordinates are assigned
		orientations: tuple of orientations {1, -1}, representing the direction of the axes in axes
		'''

		slice_3d = [slice(None), slice(None), slice(None)]

		for axis, orientation in zip(axes, orientations) :

			slice_3d[axis] = slice(None, None, orientation)

		slice_3d = tuple(slice_3d)

		self.point_coordinates[slice_3d] = coordinates

		pass

	def getCoordinates(self) -> np.ndarray :
		'''
		Get flattened coordinates of all points - size (nx * ny * nz, 3)
		Fastest changing index is along axis 0
		'''

		points = np.zeros((np.prod(self.point_indices.shape), 3), dtype=float)

		points = np.stack((
				self.point_coordinates[:, :, :, 0].flatten(order='F'),
				self.point_coordinates[:, :, :, 1].flatten(order='F'),
				self.point_coordinates[:, :, :, 2].flatten(order='F')
			), axis=-1)
		
		indices = np.argsort(self.point_indices.flatten(order='F'))

		return points[indices]

if __name__ == '__main__' :

	nx, ny, nz = 3, 3, 3
	hex = Hex((nx-1, ny-1, nz-1))

	num_cells = 0
	num_cells = hex.setCellIndices(num_cells)

	num_points = 0
	num_points = hex.setInternalPointIndices(num_points)

	print(hex.cell_indices)
