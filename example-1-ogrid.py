from pathlib import Path

import numpy as np

from scipy.interpolate import CubicHermiteSpline, splev

from modules.hex import Hex
from modules.connected_hex import ConnectedHexCollection

from modules.faces import groupFaces
from modules.polymesh import writePolyMesh

from interpolate import *

n_core	= 25
n_shell	= 25
n_z		= 100

###############################################################################

# Control points for BSpline
control_points = np.array([
	[ 1.,	 0.],
	[ 1.,	-5.],
	[ 1.,	-10.],
	[ 10.,	-15.],
	[ 10.,	-20.],
	[ 10.,	-25.]
])

# Taken from https://stackoverflow.com/a/39262872
def bspline(cv, n=100, degree=3):
	""" Calculate n samples on a bspline

		cv :      Array of control vertices
		n  :      Number of samples to return
		degree:   Curve degree
	"""
	cv = np.asarray(cv)
	count = cv.shape[0]

	# Prevent degree from exceeding count-1, otherwise splev will crash
	degree = np.clip(degree, 1, count-1)

	# Calculate knot vector
	# [0, 0, ..., 0 degree times, 0, 1, 2, ..., count - degree - 1, count - degree, count - degree, ..., count - degree degree times]
	kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree, dtype='int')

	# Calculate query range
	u = np.linspace(0, (count-degree), n)

	# Calculate result
	return np.array(splev(u, (kv, cv.T, degree))).T

points = bspline(control_points, n_z + 1, 5)

r_ext	= points[::-1, 0]
l_z		= points[::-1, 1]

r_core	= r_ext * 0.5

###############################################################################

hex_0 = Hex((n_core, n_core, n_z))

hex_1 = Hex((n_shell, n_core, n_z))
hex_2 = Hex((n_core, n_shell, n_z))
hex_3 = Hex((n_shell, n_core, n_z))
hex_4 = Hex((n_core, n_shell, n_z))

###############################################################################
# Hex 0

# Axis 0: x		- r_core / sqrt(2) -> r_core / sqrt(2)
# Axis 1: y		- r_core / sqrt(2) -> r_core / sqrt(2)
# Axis 2: z		0 -> l_z

s = 1 / np.sqrt(2)

tanp15 = np.tan(np.deg2rad(15))
tanm15 = np.tan(np.deg2rad(-15))

spline = CubicHermiteSpline([-s, s], [s, s], [tanp15, tanm15])

x = np.linspace(-s, s, n_core + 1)
y = spline(x)

edge_0 = -y
edge_1 = y

x = np.linspace(edge_0, edge_1, n_core + 1, axis=0)
y = np.linspace(edge_0, edge_1, n_core + 1, axis=1)

X = x[:, :, np.newaxis] * r_core[np.newaxis, np.newaxis, :]
Y = y[:, :, np.newaxis] * r_core[np.newaxis, np.newaxis, :]
Z = np.tile(l_z, (n_core + 1, n_core + 1, 1))

coordinates = np.stack((X, Y, Z), axis=-1)

hex_0.assignPointCoordinates(coordinates)

###############################################################################
# Hex 1

surface_0 = hex_0.getFacePointsCoordinates((1, 2, 6, 5))

theta = np.linspace(-np.pi / 4, np.pi / 4, n_core + 1)

# Axis 0: theta	- pi / 4 -> pi / 4
# Axis 1: z		0 -> l_z
x = r_ext[np.newaxis, :] * np.cos(theta[:, np.newaxis])
y = r_ext[np.newaxis, :] * np.sin(theta[:, np.newaxis])
z = np.tile(l_z, (n_core + 1, 1))

surface_1 = np.stack((x, y, z), axis=-1)

volume = interpolate_surface_to_surface(surface_0, surface_1, n_shell + 1)

hex_1.assignPointCoordinates(volume)

###############################################################################
# Hex 2

surface_0 = hex_0.getFacePointsCoordinates((3, 2, 6, 7))

theta = np.linspace(3 * np.pi / 4, np.pi / 4, n_core + 1)

# Axis 0: theta	3 * pi / 4 -> pi / 4
# Axis 1: z		0 -> l_z
x = r_ext[np.newaxis, :] * np.cos(theta[:, np.newaxis])
y = r_ext[np.newaxis, :] * np.sin(theta[:, np.newaxis])

surface_1 = np.stack((x, y, z), axis=-1)

# Axis 0: r		r_core -> r_ext
# Axis 1: theta	3 * pi / 4 -> pi / 4
# Axis 2: z		0 -> l_z
volume = interpolate_surface_to_surface(surface_0, surface_1, n_shell + 1)

# Axis 0: x | theta	3
# Axis 1: y | r		0 -> r_ext
# Axis 2: z			0 -> l_z
volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2])

hex_2.assignPointCoordinates(volume)

###############################################################################
# Hex 3

surface_1 = hex_0.getFacePointsCoordinates((0, 3, 7, 4))

theta = np.linspace(-3 * np.pi / 4, - 5 * np.pi / 4, n_core + 1)

# Axis 0: theta	- 3 * pi / 4 -> - 5 * pi / 4
# Axis 1: z		0 -> l_z
x = r_ext[np.newaxis, :] * np.cos(theta[:, np.newaxis])
y = r_ext[np.newaxis, :] * np.sin(theta[:, np.newaxis])

surface_0 = np.stack((x, y, z), axis=-1)

# Axis 0: x | r		r_ext -> r_core
# Axis 1: y | theta - 3 * pi / 4 -> - 5 * pi / 4
# Axis 2: z			0 -> l_z
volume = interpolate_surface_to_surface(surface_0, surface_1, n_shell + 1)

hex_3.assignPointCoordinates(volume)

###############################################################################
# Hex 4

surface_1 = hex_0.getFacePointsCoordinates((0, 1, 5, 4))

theta = np.linspace(-3 * np.pi / 4, - np.pi / 4, n_core + 1)

# Axis 0: theta	- 3 * pi / 4 -> - pi / 4
# Axis 1: z		0 -> l_z
x = r_ext[np.newaxis, :] * np.cos(theta[:, np.newaxis])
y = r_ext[np.newaxis, :] * np.sin(theta[:, np.newaxis])

surface_0 = np.stack((x, y, z), axis=-1)

# Axis 0: r_ext -> r_core
# Axis 1: theta - 3 * pi / 4 -> - pi / 4
# Axis 2: z		0 -> l_z
volume = interpolate_surface_to_surface(surface_0, surface_1, n_shell + 1)

# Axis 0: x | theta	- 3 * pi / 4 -> - pi / 4
# Axis 1: y | r		r_ext -> r_core
# Axis 2: z			0 -> l_z
volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2])

hex_4.assignPointCoordinates(volume)

###############################################################################

# print('Hex 1:')
# print(hex_1.getFacePointsCoordinates((0, 1, 5, 4)))
# print('Hex 4:')
# print(hex_4.getFacePointsCoordinates((3, 2, 6, 7)))

# print('Hex 2:')
# print(volume[0, :, :])

###############################################################################
# Connected Hex Collection

o_grid_mesh = ConnectedHexCollection()

o_grid_mesh.addHex(hex_0)

o_grid_mesh.addHex(hex_1)
o_grid_mesh.addHex(hex_2)
o_grid_mesh.addHex(hex_3)
o_grid_mesh.addHex(hex_4)

o_grid_mesh.connectHex(0, 1, (1, 2, 6, 5), (0, 3, 7, 4))

o_grid_mesh.connectHex(0, 2, (3, 2, 6, 7), (0, 1, 5, 4))
o_grid_mesh.connectHex(1, 2, (3, 2, 6, 7), (1, 2, 6, 5))

o_grid_mesh.connectHex(0, 3, (0, 3, 7, 4), (1, 2, 6, 5))
o_grid_mesh.connectHex(2, 3, (0, 3, 7, 4), (2, 3, 7, 6))

o_grid_mesh.connectHex(0, 4, (0, 1, 5, 4), (3, 2, 6, 7))
o_grid_mesh.connectHex(3, 4, (0, 1, 5, 4), (0, 3, 7, 4))
o_grid_mesh.connectHex(1, 4, (0, 1, 5, 4), (2, 1, 5, 6))

# o_mesh.connectHex(1, 2, (0, 1, 5, 4), (3, 2, 6, 7))
# o_mesh.connectHex(2, 3, (1, 2, 6, 5), (0, 3, 7, 4))
# o_mesh.connectHex(0, 3, (0, 1, 5, 4), (3, 2, 6, 7))

###############################################################################

num_cells = o_grid_mesh.assignCellIndices()

num_points = o_grid_mesh.assignVertexIndices()
num_points = o_grid_mesh.assignEdgePointsIndices(num_points)
num_points = o_grid_mesh.assignFacePointsIndices(num_points)
num_points = o_grid_mesh.assignInternalIndices(num_points)

faces	= o_grid_mesh.getFaces()
points	= o_grid_mesh.getPoints()

faces = groupFaces(faces, [f'h{i}_f4' for i in range(5)], 'INLET')
faces = groupFaces(faces, [f'h{i}_f5' for i in range(5)], 'OUTLET')

walls = [key for key in faces.keys() if key[0] == 'h' and key[-2] == 'f']
faces = groupFaces(faces, walls, 'WALL')

###############################################################################
# Write to file

writePolyMesh(Path('test'), faces, points)
