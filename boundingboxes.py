class Vec:
    def __init__(self,x,y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

class BoundingBox:
    def __init__(self,array):
        self.points = array
        x_points = []
        y_points = []
        for item in self.points:
            x_points.append(item[0])
            y_points.append(item[1])
        
        self._min_points = (min(x_points),min(y_points))
        self._max_points = (max(x_points),max(y_points))

    @property
    def min_points(self):
        return Vec(self._min_points[0],self._min_points[1])
    
    @property
    def max_points(self):
        return Vec(self._max_points[0],self._max_points[1])
    
if __name__ is "__main__":
    bbox = BoundingBox([(0,0), (1,2), (-5,6), (-3,2), (0.5,-1)])
    bbox