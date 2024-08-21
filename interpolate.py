import numpy as np

def interpolate_point_to_point(point_1:np.ndarray, point_2:np.ndarray, number_of_points:int) -> np.ndarray:
	
	# Creates a line of equally spaced points between two points

	line_of_points = np.linspace(point_1, point_2, number_of_points, dtype=float, axis=0)

	return line_of_points

def interpolate_line_to_line(line_1:np.ndarray, line_2:np.ndarray, number_of_points:int) -> np.ndarray:
	
	assert line_1.shape == line_2.shape, 'The two lines must have the same shape'

	surface_of_points = np.linspace(line_1, line_2, number_of_points, dtype=float, axis=0)

	return surface_of_points

def interpolate_surface_to_surface(surface_1:np.ndarray, surface_2:np.ndarray, number_of_points:int) -> np.ndarray:
	
	assert surface_1.shape == surface_2.shape, 'The two surfaces must have the same shape'

	volume_of_points = np.linspace(surface_1, surface_2, number_of_points, dtype=float, axis=0)

	return volume_of_points