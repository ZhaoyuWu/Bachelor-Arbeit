import numpy as np
from DDPG import DDPG

a_dim = 1  # action dimension
s_dim = 15  # state dimension
a_bound = 1  # action bound

PIXELS_PER_CM = 1 # 96 * 2

VAR = 0.2

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

    # print(processed_data)
    # input to cm
    processed_data_cm = pixels_to_cm(np.array(processed_data), PIXELS_PER_CM)

    adjusted_data_cm = []

    step_size = 5  # Apply DDPG every 10 steps
    window_size = 3  # Size of the window to create a state
    batch_size = 50

    ddpg_model.load_ckpt()

    width_scalar = 1
    for batch_start in range(0, len(processed_data_cm), batch_size):
        batch_end = min(batch_start + batch_size, len(processed_data_cm))
        batch_data = processed_data_cm[batch_start:batch_end]

        path_x_offset = batch_data[0][2]
        path_y_offset = batch_data[0][3]

        for k in range(len(batch_data)):
            batch_data[k][0] -= path_x_offset
            batch_data[k][1] -= path_y_offset
            batch_data[k][2] -= path_x_offset
            batch_data[k][3] -= path_y_offset

        for i in range(0, batch_size - 2, step_size):
            indices = [i, i + 1, i + 2]
            window_data = batch_data[indices]

            state = window_data.flatten()

            # 'state' should be exactly of length 15 (3 points * 5 dimensions each)
            action = ddpg_model.choose_action(state.reshape(1, -1))

            # action = np.clip(np.random.normal(action, VAR), -1, 1)

            action *= 1 # Scaling factor

            if(i==0):
                width_scalar = processed_data_cm[0][4] / (processed_data_cm[0][4] + action[0])

            for j in range(i, min(i + step_size, len(processed_data_cm))):

                adjusted_point_cm = [
                    batch_data[j][2] + path_x_offset,  # x
                    batch_data[j][3] + path_y_offset,  # y
                    min(max((batch_data[j][4] + action[0]),0.2),2) # width + action is limited between 0.5 and 2
                ]
                if(j==i):
                    # print(processed_data_cm[j][0],processed_data_cm[j][1],processed_data_cm[j][2],processed_data_cm[j][3],processed_data_cm[j][4])
                    # print(processed_data_cm[j][4] + action[0])
                    print(action[0])
                adjusted_data_cm.append(adjusted_point_cm)


    while len(adjusted_data_cm) < len(processed_data_cm):
        adjusted_data_cm.append(adjusted_data_cm[-1])

    # result to pixel
    adjusted_data_pixels = cm_to_pixels(np.array(adjusted_data_cm), PIXELS_PER_CM)

    return adjusted_data_pixels


# def width_mapping(width):
#     steep_factor = 10
#     sigmoid_width = (width - 1.5) / 1.5 * steep_factor
#
#     sigmoid_width = 1 / (1 + np.exp(-sigmoid_width))
#
#     mapped_width = sigmoid_width * 3.9 + 0.1
#
#     return mapped_width

def pixels_to_cm(data, conversion_rate):

    return data / conversion_rate

def cm_to_pixels(data, conversion_rate):

    return data * conversion_rate


if __name__ == "__main__":
    # Test

    processed_data = [[-6,  -4, -6,  -5,  2],
     [-6,  -5, -6,  -4,  2],
     [-6,  -6, -6,  -3,  2],
     [-6,  -4, -6,  -2,  2],
     [-5,  -4, -5,  -2,  2],
     [-4,  -4, -4,  -2,  2],
     [-3,  -4, -3,  -2,  2],
                      [-2, -3, -3, -3, 2],
                      [-2, -2, -2, -2, 2],
                      [-1, -2, -2, -3, 2],
                      [0, -2, -3, -4, 2],
                      [1, -1, -4, -5, 2],
     [2,  0, -5,  -6,  2],
     [2,  1, -4,  -6,  2],
     [2,  2, -4,  -5,  2],
     [3,  2, -3,  -4,  2],
     [3,  3, -3,  -3,  2],
     [2,  3, -3,  -2,  2],
     [2,  2, -2,  -2,  2],
     [2,  1, -1,  -1,  2],
     [1,  1, -1,  0,  2],
     [1,  0, 0,  0,  2],
     [0,  0, 0,  0,  2],
     [-1,  -1, -1,  0,  2]]



    print(apply_model_to_processed_data(processed_data))

