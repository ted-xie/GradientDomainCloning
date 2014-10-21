__author__ = 'Ted Xie'

import numpy as np
import pylab
import skimage
import skimage.io
import skimage.transform
import scipy
import sys

class GradientDomainClone:
    def __init__(self, fg, bg, matte):
        """Reads in foreground, background, and matte images as Numpy arrays, and also
        instantiates a new canvas."""
        self.foreground = skimage.img_as_float(skimage.io.imread(fg))
        self.background = skimage.img_as_float(skimage.io.imread(bg))
        self.matte = skimage.img_as_float(skimage.io.imread(matte))
        self.canvas = np.zeros((self.foreground.shape[0],
                                self.foreground.shape[1],
                                3))
        print "Foreground image: ", fg
        print "Background image: ", bg
        print "Matte image:      ", matte

    def naive_solution(self):
        """The naive solution simply pastes the foreground on top of the background."""
        return self.background + self.matte * (self.foreground - self.background)

    def gradient_domain_clone(self):
        """Smoothly blends the foreground region into the background region using gradient domain cloning."""
        # Number of unknowns = size of foreground
        n = 0
        # The foreground region. The Omega list indices are used
        # later to match pixels together
        Omega = []
        
        # Populate Omega, the foreground region
        for y in range(self.canvas.shape[0]):
            for x in range(self.canvas.shape[1]):
                # If (x,y) is a foreground pixel...
                if np.sum(self.matte[y][x]) != 0:
                    Omega.append((x,y))
                else:
                    # Fill in background
                    self.canvas[y][x] = self.background[y][x]
        print "Omega populated"

        n = len(Omega)
        # A (nxn) and b (nx1) hold information to solve the Poisson equation to get the unknown vector u
        A = scipy.sparse.lil_matrix((n,n), dtype=float)
        # Create an index map that maps x and y coordinates to the appropriate unknown pixel
        index_map = np.zeros((self.canvas.shape[0],self.canvas.shape[1]))
        b = np.zeros(n)

        # Solve A*u=b for all three channels, one at a time
        for channel in range(3):
            print "Solving channel", channel
            # Iterate through all foreground pixels, applying the differential Poisson equation
            for index in range(len(Omega)):
                x = Omega[index][0]
                y = Omega[index][1]
                # All neighboring pixels
                Np = []
                # Neighboring pixels that are in the foregroud
                Np_intersect_Omega = []
                # Neighboring pixels not in the foreground
                dOmega = []
                # Left pixel
                if x-1 >= 0:
                    Np.append((x-1,y))
                    if (x-1,y) not in Omega:
                        dOmega.append((x-1,y))
                    else:
                        Np_intersect_Omega.append((x-1,y))
                # Right pixel
                if x+1 < self.canvas.shape[1]:
                    Np.append((x+1,y))
                    if (x+1,y) not in Omega:
                        dOmega.append((x+1,y))
                    else:
                        Np_intersect_Omega.append((x+1,y))
                # Upper pixel
                if y-1 >= 0:
                    Np.append((x,y-1))
                    if (x,y-1) not in Omega:
                        dOmega.append((x,y-1))
                    else:
                        Np_intersect_Omega.append((x,y-1))
                # Lower pixel
                if y+1 < self.canvas.shape[1]:
                    Np.append((x,y+1))
                    if (x,y+1) not in Omega:
                        dOmega.append((x,y+1))
                    else:
                        Np_intersect_Omega.append((x,y+1))

                # Add all neighboring pixels to the index map
                # Only need to do this once (generate A once)
                if channel == 0:
                    for i in range(index,n):
                        x_val = Omega[i][0]
                        y_val = Omega[i][1]
                        index_map[y_val][x_val] = i
                        
                # Generate b-value. This is the right hand side of the differential Poisson equation
                b_value = 0
                for (i,j) in set(Np).intersection(set(dOmega)):
                    b_value += self.background[j][i][channel]
                for (i,j) in Np:
                    b_value += self.foreground[y][x][channel] - self.foreground[j][i][channel]
                b[index] = b_value
                # Only need to compute the A matrix once
                if channel == 0:
                    index_map[y][x] = index
                    # Diagonal entries are all 4
                    A[index,index] = 4
                    # Mark neighboring pixels in foreground as -1
                    # Mark at (i,j) and (j,i) to make A symmetric
                    for (i,j) in Np_intersect_Omega:
                        neighbor_idx = index_map[j][i]
                        A[index,neighbor_idx] = -1
                        A[neighbor_idx,index] = -1

            # Use SciPy's conjugate gradient solver to solve for the unknown channel values inside the new foreground region
            print "solving for colors"
            [u,flag] = scipy.sparse.linalg.cg(A,b)
            print "mixing colors"

            # Paint new foreground region with newly computed color values
            for idx in range(len(u)):
                x_val = Omega[idx][0]
                y_val = Omega[idx][1]

                self.canvas[y_val][x_val][channel] = u[idx]

            np.clip(self.canvas,0,1,self.canvas)
            print "Omega region painted"
        
        print "Done"
        return self.canvas

if __name__ == '__main__':
    args = sys.argv
    foreground = str(args[1])
    background = str(args[2])
    matte = str(args[3])

    i = GradientDomainClone(foreground, background, matte)
    composite = i.gradient_domain_clone()
    pylab.imshow(composite)
    pylab.show()
