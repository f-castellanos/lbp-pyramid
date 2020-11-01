import numpy as np
from matplotlib import pyplot as plt
import itertools


def plot_lbp(img, img_lbp, r, method):
    if method == 'riu2':
        max_val = r*8+1
    else:
        max_val = 2**(r*8)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_lbp, cmap='gray', vmax=max_val)
    plt.title('LBP Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.hist(img_lbp.ravel(), bins=list(range(max_val + 1)), ec='white')
    plt.title('Histogram')
    plt.show()


def roll_pattern(pattern, k):
    rolled = np.roll(pattern, k)
    return sum(rolled * (2 ** np.arange(len(rolled))[::-1]))


def lbp_pixel_calc(img, x, y, r, method):
    center = img[x, y]
    matrix = img[x - r:x + r + 1, y - r:y + r + 1]
    if matrix.shape[0]*matrix.shape[1] != (r*2+1)**2:
        '''
        When there are no neighboring pixels for the given radius,
        a zero value is given to some virtual pixels outside the edges.
        '''
        img_resized = np.zeros(np.array(img.shape) + 2*r)
        img_resized[r:-r, r:-r] = img
        x += r
        y += r
        matrix = img[x - r:x + r + 1, y - r:y + r + 1]
    arr = np.concatenate((matrix[0, 1:-1][::-1], matrix[:, 0], matrix[-1, 1:-1], matrix[0:, -1][::-1]))
    arr = np.where(arr < center, 1, 0)
    if method == 'riu':
        n_arr = np.array(list(arr)*len(arr)).reshape(len(arr), -1)
        return min(np.array([roll_pattern(v, i) for i, v in enumerate(n_arr)]))
    elif method == 'riu2':
        u = len(list(itertools.groupby(np.append(arr, arr[0]), lambda bit: bit == 0))) - 1
        if u > 2:
            return r*8 + 1
        else:
            return sum(arr)
    else:
        return sum(arr * (2 ** np.arange(len(arr))[::-1]))


def lbp(img, r=1, method='riu2', plot=False):
    height, width = img.shape
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        img_lbp[i, :] = [lbp_pixel_calc(img, i, j, r, method) for j in range(0, width)]
    if plot:
        plot_lbp(img, img_lbp, r, method)
    return img_lbp
