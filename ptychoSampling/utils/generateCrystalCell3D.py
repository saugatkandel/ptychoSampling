# Author - Saugat Kandel
# coding: utf-8


import numpy as np
import scipy
from scipy import spatial


def generateRandomPhase(img, mult):
    x, y, z = np.meshgrid(np.arange(img.shape[0]),
                          np.arange(img.shape[1]),
                          np.arange(img.shape[2]))
    normmax = lambda a: (a - np.mean(a)) / (np.max(a - np.mean(a)))

    x2 = normmax(x)
    y2 = normmax(y)
    z2 = normmax(z)
    randm = np.random.random((3, 3))

    _, R = np.linalg.eig(0.5 * (randm + randm.T))
    pts = R @ np.array([x2.reshape(-1), y2.reshape(-1), z2.reshape(-1)])
    phas = np.reshape(-np.pi + 2 * np.pi *
                      np.sin(mult * 2 * np.pi *
                             np.sum((R @ pts) ** 2, axis=0)),
                      np.shape(img))
    img_complex = img * np.exp(1j * phas)
    return img_complex


def getVoronoiCell(array_size, pts):
    pts = np.vstack((np.round(array_size / 2), pts))
    x, y, z = np.meshgrid(array_size[0] * np.linspace(0, 1, array_size[0]),
                          array_size[1] * np.linspace(0, 1, array_size[1]),
                          array_size[2] * np.linspace(0, 1, array_size[2]))
    samplePts = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    dist = scipy.spatial.distance.cdist(samplePts, pts)
    img = np.reshape(dist[:, 0] == np.min(dist, axis=1), (array_size))
    x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]))
    temp = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    temp = temp[img.ravel() != 0]
    centroid = np.mean(temp, axis=0)
    shift = np.round(-centroid + array_size / 2)
    img = np.roll(img, shift.astype('int'))
    return img


def generateCrystalCell(N=25, x_points=128, y_points=128, z_points=70):
    """
    Parameters:
    N = 25 # Number of Delaunay mesh points
    arr = np.array([x_points, y_points, z_points]) # desired array size
    """
    arr = np.array([x_points, y_points, z_points])

    # Generating random distribution of polar coordinates, normally distributed
    # in the radial direction and uniform in azimuthal and polar directions.
    cosTheta = -1 + 2 * np.random.random((1, N))
    sinTheta = np.sin(np.arccos(cosTheta))
    phi = np.pi * (-1 + 2 * np.random.random((1, N)))
    r = np.min(arr) / 3.5 + 0.5 * np.random.random((1, N))

    # 'pts' contains Delaunay mesh points
    pts = np.vstack([r * sinTheta * np.cos(phi),
                     r * cosTheta * np.sin(phi),
                     r * cosTheta * np.ones(phi.shape)])

    # rotating mesh points by a random rotation
    R, _, _ = np.linalg.svd(np.random.random((3, 3)))
    pts = (R @ pts).T

    # adding central point at origin
    pts = np.append([[0, 0, 0]], pts, axis=0)
    pts = pts + np.repeat(arr[None, :] / 2, N + 1, axis=0)

    img = generateRandomPhase(getVoronoiCell(arr, pts), 2)

    return img

def trimAndPadCell(cell):
    """Trimming and then ading a padding of 1 pixel to the generated cell."""
    c = cell[~np.all(cell==0, axis=(1, 2))]
    c = c[:, ~np.all(c==0, axis=(0, 2))]
    c = c[:, :, ~np.all(c==0, axis=(0, 1))]
    c = np.pad(c, [[1, 1 + c.shape[0] % 2], [1, 1 + c.shape[1] % 2], [1, 1 + c.shape[2] % 2]], mode='constant')
    return c