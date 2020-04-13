import os
import numpy as np
import json
from PIL import Image


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    '''
    BEGIN YOUR CODE
    '''

    def norm_crosscorr(img, ker, stride=2):
        """Cross-correlation but with normalization within each window."""
        out = np.zeros(((img.shape[0] - ker.shape[0]), (img.shape[1] - ker.shape[1])))
        for i in range(0, out.shape[0], stride):
            for j in range(0, out.shape[1], stride):
                window = img[i:i + ker.shape[0], j:j + ker.shape[1]]
                out[i][j] = np.sum(window / np.mean(window) * ker)
        return out

    def extract_maxes(score_maps, minscore=0.75, mindist=40):
        """Extract local maxima from score maps."""
        maxes = []
        pts = []
        scores = []
        for k, score_map in enumerate(score_maps):
            hi = np.where(score_map > minscore)
            for i in range(len(hi[0])):
                pts.append(([hi[0][i], hi[1][i]], k))
            scores += list(-score_map[hi])
        idxs = np.argsort(scores)
        for i in idxs:
            pt = np.array([pts[i][0][0], pts[i][0][1]])
            far = True
            for mpt, mk in maxes:
                if np.linalg.norm(pt - mpt) < mindist:
                    far = False
            if far:
                maxes.append(pts[i])
        return maxes

    scores = []
    for ker in kers:
        scores.append(norm_crosscorr(I, ker) / ker.size)

    tl_and_ks = extract_maxes(scores)
    for tl, k in tl_and_ks:
        bounding_boxes.append([int(tl[0]), int(tl[1]),
                               int(tl[0] + kers[k].shape[0]),
                               int(tl[1] + kers[k].shape[1])])

    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

# set the path to the downloaded data:
data_path = '../RedLights2011_Medium'

# Hand-selected kernels
kers = []
I = Image.open(os.path.join(data_path, 'RL-003.jpg'))
kers.append(np.asarray(I)[202:221, 335:342])
I.close()
I = Image.open(os.path.join(data_path,'RL-011.jpg'))
kers.append(np.asarray(I)[67:133, 349:378])
I.close()
I = Image.open(os.path.join(data_path,'RL-044.jpg'))
kers.append(np.asarray(I)[283:310, 468:481])
I.close()
for i, ker in enumerate(kers):
    kermean = np.mean(ker)
    kers[i] = (ker.astype(np.float32) - kermean) / kermean

# set a path for saving predictions:
preds_path = './hw01_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

    if i % 10 == 0:  # 525
        print(i)


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
    json.dump(preds, f)
