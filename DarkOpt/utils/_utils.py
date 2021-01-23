import matplotlib.patches as mpatches
import numpy as np



def calc_angles(det):
    """
    Function to calculate the angles of the lines dividing the Al Fins.
    Currently only works for 4 fin designs
    
    Parameters:
    -----------
    det : detector object
    
    Returns:
    --------
    angles : array
    the 4 angle in degrees, measured
    from the 1st quadrant.
   
    """


    b = det.QET.l_fin + det.QET.TES.l/2
    a = det.QET.l_fin + det.QET.TES.wempty_tes + det.QET.TES.w/2


    angle1 = np.arctan(b/a*np.tan(np.pi/4))*360/(2*np.pi)
    angle2 = (90 - angle1)*2 + angle1
    angle3 = 2*angle1 + angle2
    angle4 = 360 - angle1

    return np.array([angle1, angle2, angle3, angle4])

def _line(angle, x):
    return x/np.cos(angle*(2*np.pi)/360)


def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, color='xkcd:purple', **kwargs):
    """
    Function to generate a partial ellipse
    """

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0], 
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    poly.set_facecolor(color)

    return poly
