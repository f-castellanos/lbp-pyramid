# Original code: https://github.com/arsho/local_binary_patterns
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_pixel(img, center, x, y):
    try:
        if img[x, y] >= center:
            return 1
        else:
            return 0
    except IndexError:
        return 0


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val = np.array(
        [get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
         get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
         get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1),
         get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y)]
    )
    return sum(val*(2**np.arange(len(val))))


def show_output(img_gray, img_lbp):
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
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
            current_plot.set_xlim([0, 260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list, rotation=90)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lbp(img_gray, plot=False):
    height, width = img_gray.shape
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    if plot:
        show_output(img_gray, img_lbp)
    return img_lbp
