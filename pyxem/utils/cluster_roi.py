from hyperspy.roi import CircleROI
    from matplotlib.pyplot import Circle
import hyperspy.api as hs
from skimage.draw import circle


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
                 kx=None,
                 ky=None,
                 symmetry=None,
                 correlation=None,
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
        k: float
            The inverse spacing for the cluster
        correlation: Array-like
            The saved angular correlation for the center of the cluster at some k.
        """
        super().__init__(cx=x, cy=y, r=radius, **kwargs)
        self.kx = kx
        self.ky = ky
        self.time = time
        self.symmetry = symmetry
        if correlation is not None:
            self.correlation = correlation.inav[x, y].isig[:, k]
            self.correlation.axes_manager.signal_axes[0].offset = 0
        else:
            self.correlation = None

    def __str__(self):
        return ("Position: <" + str(self.cx) + ", " +
                str(self.cy) + ">  kx: " + str(self.kx) + "ky" + str(self.ky) +
                " radius: " + str(self.r) + " Symmetry: " + str(self.symmetry))

    def to_circle(self,
                  linewidth=2,
                  fill=False,
                  color="blue",
                  alpha=None,
                  **kwargs):
        """This takes the object and turns it into a matplotlib.Circle object
        """
        return Circle(xy=(self.cy, self.cx),
                      radius=self.r,
                      linewidth=linewidth,
                      fill=fill,
                      color=color,
                      alpha=alpha,
                      **kwargs)

    def get_mean(self,
                 signal):
        return self(signal).nansum()

    def get_kernel(self,
                   radius,
                   signal):
        shape = tuple(reversed(signal.axes_manager.signal_shape))
        mask = np.zeros(shape, dtype=bool)
        rr, cc = circle(self.ky, self.kx, radius, shape)
        print(rr, cc)
        mask[rr, cc] = True
        data = signal.data
        data[~mask] = 0
        return data

    def plot(self, **kwargs):
        markers = [hs.markers.vertical_line(j * 6.28 / self.symmetry) for j in range(self.symmetry)]
        self.correlation.add_marker(markers, permanent=True, plot_marker=True)


class Clusters(list):
    """This is a group of clusters for some experiment.  This class is designed to organize clusters into a unit for
    easier processing.
    """

    def __init__(self,
                 cluster_list):
        """ Initializes a cluster based on a list of clusters.
        """
        super().__init__(cluster_list)

    def __str__(self):
        return ("Number of Clusters: <" + len(self) + " >")

    def to_markers(self, navigation_shape,
                   **kwargs):
        """This takes the object and turns it into a matplotlib.Circle object
        """
        xindexes = np.zeros(navigation_shape)
        yindexes = np.zeros(navigation_shape)

        for cluster in self:
            xindexes[int(cluster.cy), int(cluster.cx)] = cluster.kx
            yindexes[int(cluster.cy), int(cluster.cx)] = cluster.ky

        markers = hs.plot.markers.point(xindexes, yindexes, **kwargs)
        xx, yy = [int(c.cx) for c in self], [int(c.cy) for c in self]
        nav_markers = hs.plot.markers.point(yy, xx, **kwargs)

        return markers, nav_markers

    def to_signal(self,
                  shape,
                  ):
        data = np.zeros(shape, dtype=bool)
        data = np.zeros(shape)
        for c in self:
            rr, cc = circle(c.ky, c.kx, 4, shape=shape[-2:])
            data[int(c.cx), int(c.cy), rr, cc] = True
        data = sci_gaussian_filter(data, (1, 1, 0, 0))
        return hs.signals.Signal2D(data)