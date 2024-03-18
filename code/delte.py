import numpy as np
trial = np.array([[0.        , 0.71856283],
       [0.68054677, 0.        ],
       [0.        , 0.        ],
       [0.        , 0.36728372],
       [0.        , 0.        ],
       [0.        , 0.53906036]])
print(trial)
dtIds = [1, 3, 10, 13, 17, 20]
gtIds = [362, 363]

for i in range(len(gtIds)):
    iou_matches = sum(trial[:, i] > 0.3)
    print(iou_matches)