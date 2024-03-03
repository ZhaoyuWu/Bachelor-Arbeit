from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
def preprocess_data(frontend_data, backend_data):
    """
    Use the DTW algorithm to map front-end handwriting data to back-end path data and calculate the average of the mapped points.
    frontend_data: a two-dimensional array representing the handwriting in the format [(x1, y1), (x2, y2), ...].
    backend_data: three-dimensional array, where the first two dimensions represent the path coordinates and the third dimension represents the width, in the format [[x1, y1, width1], [x2, y2, width2], ...]

    Return value: five-dimensional array, each element contains the horizontal coordinates of the stroke, the vertical coordinates of the stroke, the horizontal coordinates of the path, the vertical coordinates of the path, the width of the path
    """
    path_coords = [(point[0], point[1]) for point in backend_data]

    distance, path_mapping = fastdtw(frontend_data, path_coords, dist=euclidean)

    processed_data = np.zeros((len(backend_data), 5))

    for i, (_, backend_index) in enumerate(path_mapping):

        frontend_indices = [frontend_index for frontend_index, path_index in path_mapping if
                            path_index == backend_index]
        if frontend_indices:
            avg_x = np.mean([frontend_data[idx][0] for idx in frontend_indices])
            avg_y = np.mean([frontend_data[idx][1] for idx in frontend_indices])
        else:
            avg_x, avg_y = 0, 0  # If there is no mapping

        processed_data[backend_index, :] = [avg_x, avg_y, backend_data[backend_index][0],
                                            backend_data[backend_index][1], backend_data[backend_index][2]]
    # print(processed_data.tolist())
    return processed_data.tolist()


# test
# frontend_data = [(10, 20), (15, 25), (20, 30),(20,25)]
# backend_data = [[5, 10, 1], [10, 15, 2], [15, 20, 1]]
#
# processed_data = preprocess_data(frontend_data, backend_data)
# print("Processed Data:", processed_data)
