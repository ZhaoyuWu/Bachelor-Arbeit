import numpy as np
from DDPG import DDPG

a_dim = 1  # action dimension
s_dim = 15  # state dimension
a_bound = 1  # action bound

PIXELS_PER_CM = 96 / 2.54

ddpg_model = DDPG(a_dim, s_dim, a_bound)

"""
Input: preprocessed data(5 dimension: stroke horizontal coordinate, stroke vertical coordinate, path center horizontal coordinate, path center vertical coordinate, path width)
Output: DDPG applied data(3 dimension: path center horizontal coordinate, path center vertical coordinate, path width)
apply_model_to_processed_data: 
Step 1. Convert preprocessed data units to cm
Step 2. Sample the data every x steps
Step 3. Transform data into DDPG inputs and apply DDPG
Step 4. Convert result units to pixel
"""

def apply_model_to_processed_data(processed_data):
    ddpg_model.load_ckpt()

    # input to cm
    processed_data_cm = pixels_to_cm(np.array(processed_data), PIXELS_PER_CM)

    adjusted_data_cm = []

    # each X points apply once DDPG
    step_size = 1

    for i in range(0, len(processed_data_cm), step_size):
        window_data = processed_data_cm[max(i-2, 0):i+1]
        while len(window_data) < 3:
            window_data = np.vstack([window_data[0], window_data])
        state = np.concatenate(window_data).flatten()

        action = ddpg_model.choose_action(state.reshape(1, -1))

        for j in range(i, min(i+step_size, len(processed_data_cm))):
            adjusted_point_cm = [
                processed_data_cm[j][2],  # x
                processed_data_cm[j][3],  # y
                max(processed_data_cm[j][4] + action[0],0.1) # width + action
            ]
            print(action[0])
            adjusted_data_cm.append(adjusted_point_cm)

    # result to pixel
    adjusted_data_pixels = cm_to_pixels(np.array(adjusted_data_cm), PIXELS_PER_CM)
    return adjusted_data_pixels


def width_mapping(width):
    steep_factor = 10
    sigmoid_width = (width - 1.5) / 1.5 * steep_factor

    sigmoid_width = 1 / (1 + np.exp(-sigmoid_width))

    mapped_width = sigmoid_width * 3.9 + 0.1

    return mapped_width

def pixels_to_cm(data, conversion_rate):

    return data / conversion_rate

def cm_to_pixels(data, conversion_rate):

    return data * conversion_rate


# Test

processed_data = [[-6.62129874,  4.14477269, -6.29734142,  5.16526392,  0.75056286],
 [-4.42819322, -6.94921914,  0.83801895, -9.50826167,  0.75142882],
 [-6.45979031,  1.5257672 ,  7.45891672, -9.55752897,  0.55500714],
 [-8.22594932,  2.13430093,  4.64449773, -3.52779562,  1.60460302],
 [-7.58728258, -1.51738657,  6.13122296, -0.22713619,  1.49570679],
 [-0.78442464,  4.72888471,  3.17566733,  5.40814836,  1.21194631],
 [-5.87332563,  8.6873403 ,  3.84553129,  3.66590753,  1.76625567],
 [-2.71460278,  8.51137026,  6.98391303, -1.08194587,  1.70850523],
 [ 0.06834542, -0.98321257, -5.00663982, -4.52746667,  1.37803155],
 [ 3.80789657, -7.73523908, -0.21150073,  9.94249   ,  1.80240692]]



print(apply_model_to_processed_data(processed_data))

