import shapely.geometry as geom

class NearestPoint:
    point_line = 0
    distance = 0

    def __init__(self, line, p):
        self.line = line
        self.x, self.y = p[0], p[1]
        x = p[0]
        y = p[1]
        # ax.figure.canvas.mpl_connect('button_press_event', self)
        point = geom.Point(x, y)
        NearestPoint.distance = self.line.distance(point)
        self.line.distance(point)
        self.draw_segment(point)
        # print ('Distance to line:', distance)

    def draw_segment(self, point):
        point_on_line = self.line.interpolate(self.line.project(point))
        # print("Point on line is:", point_on_line.x, point_on_line.y)
        # self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y],
        #             color='red', marker='o', scalex=False, scaley=False)
        # self.fig.canvas.draw()
        NearestPoint.point_line = point_on_line
