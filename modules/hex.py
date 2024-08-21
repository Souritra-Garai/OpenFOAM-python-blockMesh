import numpy as np

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
	'''Orientation is +1 if vertex_2 is in the positive direction of vertex_1, -1 otherwise'''

	if	 (vertex_1, vertex_2) in hex_vertices_connectivity.keys() :

		return hex_vertices_connectivity[(vertex_1, vertex_2)], 1
	
	elif (vertex_2, vertex_1) in hex_vertices_connectivity.keys() :

		return hex_vertices_connectivity[(vertex_2, vertex_1)], -1
	
	else :

		raise ValueError('Vertices are not connected')

def checkConstantAxis(axis:int, vertices:list[int]) -> bool :

	return len(set([hex_vertices_index[vertex][axis] for vertex in vertices])) == 1

def getEdgePointsSlice(vertex_1:int, vertex_2:int) -> slice :

	axis, orientation = getAxisAndOrientation(vertex_1, vertex_2)

	const_axes = {0, 1, 2} - {axis}

	slice_3d = [None, None, None]

	slice_3d[axis] = slice(1, -1) if orientation == 1 else slice(-2, 0, -1)

	for const_axis in const_axes :

		assert checkConstantAxis(const_axis, [vertex_1, vertex_2]), 'Invalid edge'

		slice_3d[const_axis] = hex_vertices_index[vertex_1][const_axis]

	return tuple(slice_3d)

def getConstantAxis(axis_1:int, axis_2:int) -> int :

	axes_1 = {0, 1, 2} - {axis_1}
	axes_2 = {0, 1, 2} - {axis_2}

	const_axis = axes_1.intersection(axes_2)

	assert len(const_axis) == 1, 'Invalid vertices'

	return const_axis.pop()

def getFacePointsSlice(vertices:tuple[int, int, int, int]) -> slice :
	
	axis_1, orientation_1 = getAxisAndOrientation(vertices[0], vertices[1])
	axis_2, orientation_2 = getAxisAndOrientation(vertices[1], vertices[2])

	const_axis = getConstantAxis(axis_1, axis_2)

	slice_3d = [None, None, None]

	slice_3d[axis_1] = slice(1, -1) if orientation_1 == 1 else slice(-2, 0, -1)
	slice_3d[axis_2] = slice(1, -1) if orientation_2 == 1 else slice(-2, 0, -1)

	assert checkConstantAxis(const_axis, vertices), 'Vertices do not belong to a hex face'

	slice_3d[const_axis] = hex_vertices_index[vertices[0]][const_axis]

	return tuple(slice_3d)

def getCompleteSlice(axes:tuple[int, int], orientations:tuple[int, int], const_axis:int, const_axis_slice:slice) -> slice :

	slice_3d = [None, None, None]

	slice_3d[axes[0]] = slice(None, None, orientations[0])
	slice_3d[axes[1]] = slice(None, None, orientations[1])

	slice_3d[const_axis] = const_axis_slice

	return tuple(slice_3d)

def getFaceOwners(cell_indices:np.ndarray, axes:tuple[int, int], orientations:tuple[int, int], const_axis:int) -> np.ndarray :

	return cell_indices[getCompleteSlice(axes, orientations, const_axis, slice(0, -1))]

def getFaceNeighbours(cell_indices:np.ndarray, axes:tuple[int, int], orientations:tuple[int, int], const_axis:int) -> np.ndarray :

	return cell_indices[getCompleteSlice(axes, orientations, const_axis, slice(1, None))]
	
def getFaceVertex1(points:np.ndarray, axis_1:int, axis_2:int, const_axis:int, const_axis_slice:slice) -> int :

	slices_3d = [None, None, None]

	slices_3d[const_axis] = const_axis_slice

	slices_3d[axis_1] = slice(0, -1)
	slices_3d[axis_2] = slice(0, -1)

	return points[tuple(slices_3d)]

def getFaceVertex2(points:np.ndarray, axis_1:int, axis_2:int, const_axis:int, const_axis_slice:slice) -> int :

	slices_3d = [None, None, None]

	slices_3d[const_axis] = const_axis_slice

	slices_3d[axis_1] = slice(1, None)
	slices_3d[axis_2] = slice(0, -1)

	return points[tuple(slices_3d)]

def getFaceVertex3(points:np.ndarray, axis_1:int, axis_2:int, const_axis:int, const_axis_slice:slice) -> int :

	slices_3d = [None, None, None]

	slices_3d[const_axis] = const_axis_slice

	slices_3d[axis_1] = slice(1, None)
	slices_3d[axis_2] = slice(1, None)

	return points[tuple(slices_3d)]

def getFaceVertex4(points:np.ndarray, axis_1:int, axis_2:int, const_axis:int, const_axis_slice:slice) -> int :

	slices_3d = [None, None, None]

	slices_3d[const_axis] = const_axis_slice

	slices_3d[axis_1] = slice(0, -1)
	slices_3d[axis_2] = slice(1, None)

	return points[tuple(slices_3d)]


class Hex :
	
	def __init__(self, cell_shape:tuple[int, int, int]) -> None:

		point_shape = tuple([i+1 for i in cell_shape])
		
		self.cell_indices		= np.zeros(cell_shape, dtype=int)
		self.point_indices		= np.zeros(point_shape, dtype=int)
		self.point_coordinates	= np.zeros(point_shape + (3,), dtype=float)

		self.cell_indices.fill(-1)
		self.point_indices.fill(-1)
		self.point_coordinates.fill(np.nan)
		
		pass

	def setCellIndices(self, cell_index:int=0) -> int:
		
		# Get the number of cells in the volume
		num_cells = np.prod(self.cell_indices.shape)
		
		# Create a linear array of cells
		self.cell_indices = np.arange(cell_index, cell_index + num_cells).reshape(self.cell_indices.shape, order='F')
		
		return cell_index + num_cells
	
	def setInternalPointIndices(self, point_index:int=0) -> int:

		shape_internal = tuple([i-2 for i in self.point_indices.shape])
		
		# Get the number of internal points in the volume
		num_internal_points = np.prod(shape_internal)
		
		self.point_indices[1:-1, 1:-1, 1:-1] = np.arange(point_index, point_index + num_internal_points).reshape(shape_internal, order='F')
		
		return point_index + num_internal_points
	
	def getFaceShape(self, vertex_indices:tuple[int, int, int, int]) -> tuple[int, int]:
		'''Returns the shape of cell faces on the hex face'''
		
		axes_1, orientation_1 = getAxisAndOrientation(vertex_indices[0], vertex_indices[1])
		axes_2, orientation_2 = getAxisAndOrientation(vertex_indices[0], vertex_indices[3])

		return self.cell_indices.shape[axes_1], self.cell_indices.shape[axes_2]
	
	def getVertexPointIndex(self, vertex_num:int) -> int :

		return self.point_indices[hex_vertices_index[vertex_num]]
	
	def setVertexPointIndex(self, vertex_num:int, point_index:int) -> None :

		self.point_indices[hex_vertices_index[vertex_num]] = point_index

		pass
	
	def getEdgePointsIndices(self, vertex_1:int, vertex_2:int) -> np.ndarray :

		return self.point_indices[getEdgePointsSlice(vertex_1, vertex_2)]
		
	def setEdgePointsIndices(self, vertex_1:int, vertex_2:int, point_indices:np.ndarray) -> None :
		
		self.point_indices[getEdgePointsSlice(vertex_1, vertex_2)] = point_indices

		pass
	
	def getFacePointsIndices(self, vertices:tuple[int, int, int, int]) -> np.ndarray :

		return self.point_indices[getFacePointsSlice(vertices)]
		
	def setFacePointsIndices(self, vertices:tuple[int, int, int, int], point_indices:np.ndarray) -> None :

		self.point_indices[getFacePointsSlice(vertices)] = point_indices

		pass

	def getFace(self, vertices:tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray] :

		axis_1, orientation_1 = getAxisAndOrientation(vertices[0], vertices[1])
		axis_2, orientation_2 = getAxisAndOrientation(vertices[1], vertices[2])

		const_axis = getConstantAxis(axis_1, axis_2)

		assert checkConstantAxis(const_axis, vertices), 'Vertices do not belong to a hex face'
		const_axis_value = hex_vertices_index[vertices[0]][const_axis]

		face_slice = getCompleteSlice(
			(axis_1, axis_2),
			(orientation_1, orientation_2),
			const_axis,
			const_axis_value
		)

		points	= self.point_indices[face_slice]
		cells	= self.cell_indices[face_slice].flatten(order='F')

		face_points_indices = np.stack((
			points[:-1, :-1].flatten(order='F'),
			points[ 1:, :-1].flatten(order='F'),
			points[ 1:,  1:].flatten(order='F'),
			points[:-1,  1:].flatten(order='F')
		), axis=-1)

		return face_points_indices, cells
	
	def getInternalFaces(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray] :

		point_indices	= np.zeros((0, 4), dtype=int)

		cell_owner		= np.zeros(0, dtype=int)
		cell_neighbour	= np.zeros(0, dtype=int)

		for const_axis, axes in enumerate(((1, 2), (2, 0), (0, 1))) :

			owner		= getFaceOwners(self.cell_indices, axes, (1, 1), const_axis)
			neighbour	= getFaceNeighbours(self.cell_indices, axes, (1, 1), const_axis)
			
			point_indices_0 = getFaceVertex1(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))
			point_indices_1 = getFaceVertex2(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))
			point_indices_2 = getFaceVertex3(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))
			point_indices_3 = getFaceVertex4(self.point_indices, axes[0], axes[1], const_axis, slice(1, -1))

			cell_owner		= np.concatenate((cell_owner, owner.flatten(order='F')))
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

		axis_1, orientation_1 = getAxisAndOrientation(vertices[0], vertices[1])
		axis_2, orientation_2 = getAxisAndOrientation(vertices[1], vertices[2])

		const_axis = getConstantAxis(axis_1, axis_2)

		assert checkConstantAxis(const_axis, vertices), 'Vertices do not belong to a hex face'

		slice_3d = getCompleteSlice(
			(axis_1, axis_2),
			(orientation_1, orientation_2),
			const_axis,
			hex_vertices_index[vertices[0]][const_axis]
		)

		return self.point_coordinates[slice_3d]		

	def assignPointCoordinates(self, coordinates:np.ndarray, axes:tuple[int, int, int]=(0, 1, 2), orientations:tuple[int, int, int]=(1, 1, 1)) -> None :

		slice_3d = [slice(None), slice(None), slice(None)]

		for axis, orientation in zip(axes, orientations) :

			slice_3d[axis] = slice(None, None, orientation)

		slice_3d = tuple(slice_3d)

		self.point_coordinates[slice_3d] = coordinates

		pass

	def getCoordinates(self) -> np.ndarray :

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
