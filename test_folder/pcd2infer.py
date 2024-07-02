import numpy as np

input_npy = np.load('./test_folder/full_pyramid_peg_scaled.npy', allow_pickle=True)

print(input_npy.min())
print(input_npy.shape)
print('-------')
print(input_npy[0][0])

converted_dict = {
    'xyz': np.array(input_npy, dtype=np.float32),
    'K': np.ones((3, 3), dtype=np.float32)
}

np.save('./test_folder/full_pyramid_peg_preprocess.npy', converted_dict)