import numpy as np
import math
import shapely.geometry as geom
from Near_point import NearestPoint

def force(cx, cy, radi, coord1, coord2):
    ########## initialize variables ###########
    """force_feed = 0
    direction = np.zeros((1, 2))
    go_up = False
    go_down = False
    go_right = False
    case = 0"""
    k = 1
    ######### point = center of object ##########
    p = [cx, cy]
    ############## lower bownd ##################
    line = geom.LineString(coord1)

    np1 = NearestPoint(line, p)

    d1 = np1.distance

    px1 = np1.point_line.x
    py1 = np1.point_line.y
    ############### upper bownd #################
    line = geom.LineString(coord2)

    np2 = NearestPoint(line, p)

    d2 = np2.distance

    px2 = np2.point_line.x
    py2 = np2.point_line.y
    ############ create left bownd ##############
    coords3 = np.zeros((len(coord1[:, 0]), 2))
    coords3[:, 0] = coord1[0, 0]
    coords3[:, 1] = np.linspace(coord1[0, 1], coord2[0, 1], len(coords3[:, 0]))

    line = geom.LineString(coords3)

    np3 = NearestPoint(line, p)

    d3 = np3.distance
    px3 = np3.point_line.x
    py3 = np3.point_line.y
    ##########################################################################################################################

    if (cy <= py1) and (cx - radi >= coord1[0, 0]):
        # print('1')
        #go_up = True
        force_feed = k * (radi + d1)
        if d1 == 0:
            direction = [0, 1]
        else:
            direction = [(px1 - cx) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2),
                         (py1 - cy) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2)]
        #case = 2
    elif ((cy > py1) and (d1 < radi)) and (cx - radi >= coord1[0, 0]):
        # print('2')
        #go_up = True
        force_feed = k * (radi - d1)
        if d1 == 0:
            direction = [0, 1]
        else:
            direction = [-(px1 - cx) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2),
                         -(py1 - cy) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2)]
        #case = 9
    elif (cy >= py2) and (cx - radi >= coord2[0, 0]):
        # print('3')
        #go_down = True
        force_feed = k * (radi + d2)
        if d2 == 0:
            direction = [0, -1]
        else:
            direction = [(px2 - cx) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2),
                         (py2 - cy) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2)]
        #case = 1
    elif ((cy < py2) and (d2 < radi)) and (cx - radi >= coord2[0, 0]):
        # print('4')
        #go_down = True
        force_feed = k * (radi - d2)
        if d2 == 0:
            direction = [0, -1]
        else:
            direction = [-(px2 - cx) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2),
                         -(py2 - cy) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2)]
        #case = 9
    elif (cx - radi < coord1[0, 0]) and (cy < py2) and (cy > py1) and (d2 > radi) and (d1 > radi):
        # print('5')
        #go_right = True
        if cx >= coord1[0, 0]:
            force_feed = k * (radi - d3)
            direction = [1, 0]
            #case = 9
        else:
            force_feed = k * (radi + d3)
            direction = [1, 0]
            #case = 3
    ################################################################################################################################
    elif (cy >= py2) and (cx - radi < coord1[0, 0]):
        # print('6')
        if d2 <= d3:
            #go_down = True
            force_feed = k * (radi + d2)
            if d2 == 0:
                direction = [0, -1]
            else:
                direction = [(px2 - cx) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2),
                             (py2 - cy) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2)]
            #case = 1
        else:
            #go_right = True
            force_feed = k * (radi + d3)
            if d2 == 0:
                direction = [1, 0]
            else:
                direction = [(px3 - cx) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2),
                             (py3 - cy) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2)]
            #case = 5
    elif (d3 < radi) and (cy < py2) and (d2 < radi):
        # print('7')
        if d2 <= d3:
            #go_down = True
            force_feed = k * (radi - d2)
            if d2 == 0:
                direction = [0, -1]
            else:
                direction = [-(px2 - cx) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2),
                             -(py2 - cy) / math.sqrt((px2 - cx) ** 2 + (py2 - cy) ** 2)]
            #case = 9
        else:
            if cx > coord1[0, 0]:
                #go_right = True
                force_feed = k * (radi - d3)
                if d3 == 0:
                    direction = [1, 0]
                else:
                    direction = [-(px3 - cx) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2),
                                 -(py3 - cy) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2)]
                #case = 9
            else:
                #go_right = True
                force_feed = k * (radi + d3)
                if d3 == 0:
                    direction = [1, 0]
                else:
                    direction = [(px3 - cx) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2),
                                 (py3 - cy) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2)]
                #case = 3
    ################################################################################################################################
    elif (cy <= py1) and (cx - radi < coord1[0, 0]):
        # print('8')
        if d1 <= d3:
            #go_up = True
            force_feed = k * (radi + d1)
            if d1 == 0:
                direction = [0, 1]
            else:
                direction = [(px1 - cx) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2),
                             (py1 - cy) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2)]
            #case = 2
        else:
            #go_right = True
            force_feed = k * (radi + d3)
            if d3 == 0:
                direction = [1, 0]
            else:
                direction = [(px3 - cx) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2),
                             (py3 - cy) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2)]
            #case = 5
    elif (d3 < radi) and (cy > py1) and (d1 < radi):
        # print('9')
        if d1 <= d3:
            #go_up = True
            force_feed = k * (radi - d1)
            if d1 == 0:
                direction = [0, 1]
            else:
                direction = [-(px1 - cx) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2),
                             -(py1 - cy) / math.sqrt((px1 - cx) ** 2 + (py1 - cy) ** 2)]
            #case = 9
        else:
            if cx > coord1[0, 0]:
                #go_right = True
                force_feed = k * (radi - d3)
                if d3 == 0:
                    direction = [1, 0]
                else:
                    direction = [-(px3 - cx) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2),
                                 -(py3 - cy) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2)]
                #case = 9
            else:
                #go_right = True
                force_feed = k * (radi + d3)
                if d3 == 0:
                    direction = [1, 0]
                else:
                    direction = [(px3 - cx) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2),
                                 (py3 - cy) / math.sqrt((px3 - cx) ** 2 + (py3 - cy) ** 2)]
                #case = 3
    else:
        force_feed = 0
        direction = [0, 0]

    observ = [force_feed, direction[0], direction[1]]

    # return force, direction, go_up, go_down, go_right
    return force_feed, direction, observ
