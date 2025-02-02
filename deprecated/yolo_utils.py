import numpy as np

def prepare_bbox_string(label,x,y,h,w,dx,dy):
    # Box coordinates are converted into normalized coordinates (0,1)

    # x,y need to be moved to the box center

    h_n = h / dy
    w_n = w / dx

    x = x + w/2 # move to center
    y = y + h/2 # move to center

    x_n = x / dx
    y_n = y / dy

    bbox_str = "%d %.2f %.2f %.2f %.2f" % (label, x_n, y_n, w_n, h_n)

    return bbox_str
