import matplotlib.pyplot as plt
import sys

def DrawBoundaryOfBinaryMask(image, ROI):
    plt.imshow(image, cmap='Greys_r')
    plt.contour(ROI, colors='y')
    plt.show()

def LoadWaitBar(total, progress):
    '''
    Show the wait bar
    :param total: the number of the total steps
    :param progress: the number of the current step
    :return:
    '''
    barLength, status = 20, ""
    raw_progress = progress
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% ({}/{}) {} ".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0), raw_progress, total,
        status)
    sys.stdout.write(text)
    sys.stdout.flush()