import os
import json
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def plot_bbox(img, coords):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 6)
    I = Image.open(os.path.join(data_path, img))
    ax.set_title(img)
    ax.imshow(I)
    for coord in coords:
        tli, tlj, bri, brj = coord

        x, y = tlj - 1, tli - 1
        w, h = brj - tlj + 1, bri - tli + 1
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')
        
    plt.savefig(os.path.join(preds_path, img[:-4] + '_out.png'))
    plt.show()
    plt.close()


data_path = '../RedLights2011_Medium'
preds_path = './hw01_preds'

with open(os.path.join(preds_path, 'preds.json'), 'r') as f:
    preds = json.load(f)
test_imgs = ['RL-138.jpg', 'RL-215.jpg', 'RL-229.jpg', 'RL-303.jpg',
             'RL-168.jpg', 'RL-101.jpg', 'RL-288.jpg', 'RL-190.jpg']  # 271, 138
for im in test_imgs:
    plot_bbox(im, preds[im])