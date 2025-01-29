import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Ambiguity:

    def __init__(self, sampleDist, minDist):
        self.sampleDist = sampleDist
        self.minDist = minDist
        self.points = None
        return

    def buildStructure(self, G):

        #sample

        #G.Graph['width']

        width = G.graph['xmax']
        height = G.graph['ymax']

        w_bins = int(width / self.minDist) + 1
        h_bins = int(height / self.minDist) + 1

        points = [ [[]] * w_bins for i in range(h_bins)]
        PY = []
        PX = []
        for u,v,data in G.edges(data=True):
            if 'X_CP' and 'Y_CP' in data:
                X = data['X_CP']
                Y = data['Y_CP']

                remainder = self.sampleDist
                x_last = X[0]
                y_last = Y[0]

                for i in range(1, len(X)):
                    x_next = X[i]
                    y_next = Y[i]

                    d_x = x_next - x_last
                    d_y = y_next - y_last

                    d = np.sqrt(d_x**2 + d_y**2)
                    d_x = d_x / d
                    d_y = d_y / d       
                    #d /= self.sampleDist

                    nextD = remainder
                    while nextD < d:
                        p_x = x_last + d_x * nextD
                        p_y = y_last + d_y * nextD

                        nextD = nextD + self.sampleDist

                        PX.append(p_x)
                        PY.append(p_y)

                        w_bin = int(p_x / self.minDist)
                        h_bin = int(p_y / self.minDist)

                        a = None

                        points[h_bin][w_bin].append((p_x, p_y, u, v, a))

                    remainder = nextD - d

                    x_last = x_next
                    y_last = y_next
            else:
                print('Error: Missing data')
                return


        self.points = points
        self.plotPoints(PX,PY, G)


    def plotPoints(self, X, Y, G):

        width = G.graph['xmax']
        height = G.graph['ymax']

        fig, ax = plt.subplots(figsize=(G.graph['xmax'] / 96, G.graph['ymax'] / 96), dpi=96)
        ax.axis('off')
        #fig.canvas.set_window_title(self.name)

        ax.plot(X, Y, linestyle="none", color='black', marker='.', markersize = 1.0)

        x = 0
        while x <= width:
            ax.plot([x, x], [0, height], color='red')

            x += self.minDist

        y = 0
        while y <= height:
            ax.plot([0, width], [y, y], color='red')

            y += self.minDist

        plt.savefig(f'points_{datetime.now().strftime("%H:%M:%S")}.png')
        plt.close(fig)