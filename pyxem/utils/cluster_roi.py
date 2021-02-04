from hyperspy.roi import CircleROI
from matplotlib.pyplot import Circle

class Cluster(CircleROI):
    """This is a real space cluster of like symmetry diffraction
    patterns in a sample. Eventually it may allow for more diverse
    shapes than circles but this is a useful tool to getting cluster
    information from different signals by allowing the user to easily
    slice all forms of some dataset.
    """
    def __init__(self,
                 x=None,
                 y=None,
                 time=None,
                 radius=None,
                 k=None,
                 symmetry=None,
                 **kwargs):
        """ Initializes some cluster. The coordinates x and y are the real space
        coordinates defining the circle of interest

        Parameters
        -----------
        x: float
            The real space x position of the cluster
        y: float
            The real space y position of the clusters
        r: float
            The radius of the cluster
        k: The k
        """
        super().__init__(cx=x, cy=y, r=radius,  **kwargs)
        self.k = k
        self.time = time
        self.symmetry = symmetry

    def __str__(self):
        return ("Position: <" + str(self.cx) +", " +
               str(self.cy) + ">  k: "+ str(self.k) +
               " radius: "+str(self.r)+" Symmetry: " +str(self.symmetry))

    def to_circle(self,
                  linewidth=2,
                  fill=False,
                  color="blue",
                  alpha=None,
                  **kwargs):
        """This takes the object and turns it into a matplotlib.Circle object
        """
        return Circle(xy=(self.cy,self.cx),
                      radius=self.r,
                      linewidth=linewidth,
                      fill=fill,
                      color=color,
                      alpha=alpha,
                      **kwargs)

