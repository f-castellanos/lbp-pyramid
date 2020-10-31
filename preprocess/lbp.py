# Original code: https://github.com/arsho/local_binary_patterns
import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools


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
            return len(arr) + 1
        else:
            return sum(arr)
    else:
        return sum(arr * (2 ** np.arange(len(arr))[::-1]))


def show_output(img_gray, img_lbp):
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [10], [0, 10])
    output_list = [{
        "img": img_gray,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "Gray Image",
        "type": "gray"
    }, {
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    }, {
        "img": hist_lbp,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
    }]
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i + 1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap=plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color="black")
            current_plot.set_xlim([0, 10])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list, rotation=90)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lbp(img_gray, r=1, method='riu2', plot=False):
    height, width = img_gray.shape
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_pixel_calc(img_gray, i, j, r, method)
    if plot:
        show_output(img_gray, img_lbp)
    return img_lbp
