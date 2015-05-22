'''
Created on Oct 20, 2014

@author: sunilrao
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pylab import *
from skimage import img_as_float
import sys
from skimage.filter import hsobel, vsobel
from skimage.color import rgb2gray
from PIL import Image
from scipy.misc import toimage


# =========================================================================
# Note: - The conventions used and shortest path algorithm used
# is in line with the Shortest Path algorithm discussed in textbook
# - Algorithms 4th Edition, Robert Sedgewick and Kevin Wayne
# Except we do not use EdgeWeightedDigraph but
# use a single dimension array weights[v] whose index
# has a value = row *width + col
# =========================================================================


class Seamcarver(object):

    def __init__(self, img):
        super(Seamcarver, self).__init__()
        self._img = img
        self._width = self._img.shape[:2][0]
        self._height = self._img.shape[:2][1]
        self._border_energy = 3 * (2)
        # individual node weights
        self._weights = None
        # cumulative distance is captured in distTo
        self._distTo = None
        # given an index determine its
        # parent (expressed matrix (col * row) as x integer of the total
        # (col * row) nodes
        self._edgeTo = None
        # initialize the search vectors defined above
        self._hR = None
        self._hG = None
        self._hB = None
        self._vR = None
        self._vG = None
        self._vB = None

    # Helper function to be called within plot_seam
    def dual_gradient_energy(self, img):
        returnDouble = np.zeros([self._width, self._height])
        for i in xrange(0, self._width, 1):
            for j in xrange(0, self._height, 1):
                returnDouble[i][j] = self.energy(i, j)
        return returnDouble

    # an array of H integers, for each row return the column of the seam
    def find_seam(self, img):
        self._img = img
        self.initialize(self.node(0, 0), self.node(self._width - 1, 0), 1)
        for startCol in xrange((self._width - 1), (-1 * self._height), -1):
            if (startCol >= 0):
                row = 0
                col = startCol
            else:
                row = -1 * startCol
                col = 0
            while ((row < self._height - 1) and (col < self._width)):
                v = self.node(col, row)
                if (col > 0):
                    self.relax(v, self.node(col - 1, row + 1))
                self.relax(v, self.node(col, row + 1))
                if (col < (self._width - 1)):
                    self.relax(v, self.node(col + 1, row + 1))
                row += 1
                col += 1
        endOfSeam = self.argmin(
            self.node(0, (self._height - 1)), self._width * self._height, 1)
        return self.verticalPath(endOfSeam)

    # Core method of shortest path algo of relaxation
    def relax(self, start, stop):
        if (self._distTo[stop] > self._distTo[start] + self._weights[stop]):
            self._distTo[stop] = self._distTo[start] + self._weights[stop]
            self._edgeTo[stop] = start

    # Return index of the minimum energy
    def argmin(self, start, stop, skip):
        if(stop <= start or start < 0 or len(self._distTo) == 0):
            return None
        minTemp = sys.maxint
        argmin = start
        if (stop > len(self._distTo)):
            stop = len(self._distTo)
        for i in xrange(start, stop, skip):
            if(self._distTo[i] < minTemp):
                minTemp = self._distTo[i]
                argmin = i
        return argmin

    # Analogous to shortest path algo of traversing back to the source
    # parent = edgeTo[child] - so go on till you hit prev <= 0
    def verticalPath(self, stop):
        seam = [int] * self._height
        seam[self.row(stop)] = self.col(stop)
        prev = self._edgeTo[stop]
        while(prev >= 0):
            seam[self.row(prev)] = self.col(prev)
            prev = self._edgeTo[prev]
        return seam

    # Reinitializing as remove seam would have done the cleanup
    # Returns both energy pic matrix and seam overlaid energy pic matrix
    def plot_seam(self, img, seam):
        self._img = img
        self.initialize(self.node(0, 0), self.node(self._width - 1, 0), 1)
        energyMatrix = self.dual_gradient_energy(img)
        wdth = energyMatrix.shape[0]
        ht = energyMatrix.shape[1]
        max_val = np.max(energyMatrix)
        temp_img_energy = np.zeros([self._width, self._height, 3])
        for i in xrange(0, wdth, 1):
            for j in xrange(0, ht, 1):
                normalizedGrayValue = energyMatrix[i][j] / max_val
                temp_img_energy[i, j, :] = [
                    normalizedGrayValue, normalizedGrayValue,
                    normalizedGrayValue]
        overlaidPic = np.copy(temp_img_energy)
        for j in xrange(0, ht, 1):
            overlaidPic[seam[j], j, :] = [1, 0, 0]
        return (temp_img_energy, overlaidPic)

    # This function gets called after find_seam
    # Note this is vertical seam meaning - width gets reduced by 1 unit
    def remove_seam(self, img, seam):
        if (self._width == 0 or len(seam) != self._height):
            return None
        lastcol = seam[0]
        yet_another_temp_img = np.zeros([self._width - 1, self._height, 3])
        for row in xrange(0, self._height, 1):
            if (seam[row] < (lastcol - 1) or seam[row] > (lastcol + 1)):
                return None
            if (seam[row] < 0 or seam[row] >= self._width):
                return None
            lastcol = seam[row]
            temp_img_sub_row_plot = np.delete(
                self._img[:, row, :], lastcol, axis=0)
            yet_another_temp_img[:, row, :] = np.copy(temp_img_sub_row_plot)

        print "New Shape of the removed image is ", yet_another_temp_img.shape
        # Set image to new one and do cleanup
        self._img = yet_another_temp_img
        self._distTo = None
        self._edgeTo = None
        self._weights = None
        return yet_another_temp_img

    # converting 2 dimensional to uni dimensional
    def node(self, col, row):
        return (row * self._width) + col
    # Given single dimensional index return the column

    def col(self, node):
        return node % self._width
    # Given single dimensional index return the row

    def row(self, node):
        return node / self._width

    # Obtain the pre-calculated values of the horizontal sobel filter(2d array)
    # for given x, y co-ordinates
    def gradient_h(self, x, y):
        return np.square(np.asarray([self._hR[x, y],
                                     self._hG[x, y], self._hB[x, y]]))

    # Obtain the pre-calculated values of the vertical sobel filter(2d array)
    # for given x, y co-ordinates
    def gradient_v(self, x, y):
        return np.square(np.asarray([self._vR[x, y],
                                     self._vG[x, y], self._vB[x, y]]))

    # returns energy for the pixel(x,y)
    def energy(self, x, y):
        if (x < 0 or x >= self._width or y < 0 or y >= self._height):
            return None
        if (x == 0 or y == 0 or x == self._width - 1 or y == self._height - 1):
            return self._border_energy
        retVal_h = self.gradient_h(x, y)
        retVal_v = self.gradient_v(x, y)
        # Gradient magnitude as defined in Sobel filter of wikipedia
        finalVal = np.sum(np.sqrt(np.add(retVal_h, retVal_v)))
        return finalVal

    # initialize the search vectors
    def initialize(self, start, stop, skip):
        self._width = self._img.shape[:2][0]
        self._height = self._img.shape[:2][1]
        size = self._width * self._height
        self._weights = [float] * size
        self._edgeTo = [int] * size
        self._distTo = [float] * size
        R = self._img[:, :, 0]
        G = self._img[:, :, 1]
        B = self._img[:, :, 2]

        self._vR = vsobel(R)
        self._vG = vsobel(G)
        self._vB = vsobel(B)

        self._hR = hsobel(R)
        self._hG = hsobel(G)
        self._hB = hsobel(B)
        for v in xrange(0, size, 1):
            if ((v >= start) and (v < stop) and (v - start) % skip == 0):
                self._distTo[v] = 0
            else:
                self._distTo[v] = sys.maxint
            self._edgeTo[v] = -1
            self._weights[v] = self.energy(self.col(v), self.row(v))


def main():
    print "PP03 CST 515 - Begin"
    img = mpimg.imread(sys.argv[1])
    img = img_as_float(img)
    print "Shape of the input image is ", img.shape
    seamCarvr = Seamcarver(img)
    seam = seamCarvr.find_seam(img)
    print "Check len(seam array) = height of image", len(seam)
    reduced_img = seamCarvr.remove_seam(img, seam)
    # See if the image is good after removing the seam
    toimage(reduced_img).show()
    retTuple = seamCarvr.plot_seam(img, seam)
    # See the energy pic of the image
    toimage(retTuple[0]).show()
    # See the overlayed image on top of energy pic the seam values removed
    # For vertical seam the length of the seam should be equal to height
    toimage(retTuple[1]).show()
    pass


if __name__ == '__main__':
    main()
