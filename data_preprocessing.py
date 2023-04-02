from scipy.io import loadmat
import os
import cv2
import numpy as np


# airs
max_v = 20.793742623020364
min_v = -2.305145422816422
# path = 'handwritingBCI/handwritingBCIData/RNNTrainingSteps/Step3_SyntheticSentences/HeldOutBlocks'
# a = loadmat('handwritingBCI/handwritingBCIData/RNNTrainingSteps/Step3_SyntheticSentences/HeldOutBlocks/t5.2019.05.08_snippets_interpolated.mat')
# a_set = a['a'][0]
# x = (a_set - min_v) / (max_v - min_v)
# y = np.zeros((x.shape[0], x.shape[1], 3))
# y[:, :, 0] = x
# y[:, :, 1] = x
# y[:, :, 2] = x

# cv2.imwrite('project/try1.png', y * 255.0)
# a = cv2.imread('project/try1.png')
# print(a.shape)

source_path = 'handwritingBCI/handwritingBCIData/RNNTrainingSteps/Step3_SyntheticSentences/HeldOutTrials'
target_path = 'project/data/val'

for letter in ['a', 'i', 'r', 's']:
    index = 0
    for i in os.listdir(source_path):
        dict_all = loadmat(os.path.join(source_path, i))
        set_letter = dict_all[letter]
        for j in range(set_letter.shape[0]):
            instance = set_letter[j]
            instance_normalized = (instance - min_v) / (max_v - min_v) * 255.0
            img = np.zeros((instance_normalized.shape[0], instance_normalized.shape[1], 3))
            img[:, :, 0] = instance_normalized
            img[:, :, 1] = instance_normalized
            img[:, :, 2] = instance_normalized
            cv2.imwrite(os.path.join(target_path, letter, str(index).zfill(4) + '.png'), img)
            index = index + 1
