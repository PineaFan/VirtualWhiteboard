"""Shows hand points using matplotlib"""
import sys
import time

import driver
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

driver = driver.Driver(debug=False, modules=["hands"])
n = 0
while True:
    n += 1
    driver.calculate(1, 1)
    if driver.fullHandLandmarks:
        handPoints = driver.fullHandLandmarks[0].landmark
        x = [p.x for p in handPoints]
        y = [p.y for p in handPoints]
        z = [p.z for p in handPoints]
        connections = [(x, x + 1) for x in range(1, 20) if x % 4 != 0]
        connections += [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        # Show all the points
        for i in range(21):
            value = driver.raisedFingers[0][max((i-1)//4, 0)] * 100
            ax.scatter(x[i], y[i], z[i], color=("red" if value < 0 else "green"))
        # Draw lines on all connections
        for connection in connections:
            ax.plot(
                [x[connection[0]], x[connection[1]]],
                [y[connection[0]], y[connection[1]]],
                [z[connection[0]], z[connection[1]]],
            color="red")
        # Get the current mode by converting the driver.raisedFingers to a binary number
        modeInteger = "".join(["1" if x > 0 else "0" for x in driver.raisedFingers[0]])
        mode = modes.get(modeInteger, "none")
        if mode == "close":
            plt.close()
            sys.exit()
        plt.pause(0.001)

