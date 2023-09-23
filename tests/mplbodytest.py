"""Tests the points on the body using matplotlib"""
import sys
import time

from ..modules import driver
from ..modules import body
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def clamp(mi, ma, x):
    return max(mi, min(ma, x))


modes = {
    "01000": "draw",
    "00100": "close",
    "01100": "line",
    "01111": "rubber"
}

plt.ion()
plt.show()

driver = driver.Driver(debug=False, modules=["body"])
n = 0
while True:
    n += 1
    driver.calculate(1, 1)
    if driver.full_body_results is not None:
        body_points = driver.full_body_results.landmark
        x = [p.x for p in body_points]
        y = [p.y for p in body_points]
        z = [p.z for p in body_points]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        # Show all the points
        for i in range(21):
            ax.scatter(x[i], y[i], z[i], color=("red"))
        plt.pause(0.001)

