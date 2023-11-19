import time
import numpy as np
import matplotlib.pyplot as plt

class tamatrix_importer:
    def __init__(self):
        self.startnm = 0
        self.endnm = 1000

    def import_data_silent(self, filename):
        # Load firstcol wave and find startrow and endrow
        # filename = input("Enter the filename for firstcol wave: ")
        startnm = 0
        endnm = 2000
        self.filename = filename
        firstcol = np.loadtxt(self.filename)[:, 1]
        if startnm < np.min(firstcol):
            startrow = np.argmin(firstcol)
        else:
            for index in range(len(firstcol)):
                if firstcol[index] > startnm:
                    startrow = index
                    break
        if endnm > np.max(firstcol):
            endrow = np.argmax(firstcol)
        else:
            for index in range(len(firstcol)):
                if firstcol[index] > endnm:
                    endrow = index
                    break

        # Load TAwavelength waves
        self.tawavelength = np.loadtxt(
            self.filename, skiprows=startrow, max_rows=endrow-startrow)[:, 1]
        # np.savetxt(self.filename+"_tawavelength",tawavelength,fmt='%1.5f')

        # Trim TAtime wave
        self.tatime = np.loadtxt(self.filename)[:, 0]
        idx = np.loadtxt(self.filename).shape[1]-2
        self.tatime = self.tatime[:idx]
        # np.savetxt(self.filename+"_tatime",tatime,fmt='%1.5f')

        # Load TAmatrix waves
        self.tamatrix = np.loadtxt(self.filename, skiprows=startrow,
                                   max_rows=endrow-startrow, usecols=np.arange(2, idx+2))
        # np.savetxt(self.filename+"_tamatrix",self.tamatrix,fmt='%1.5f')

    def auto_bgcorr(self, points):
        npavg = 0
        self.bgcorr = self.tamatrix.copy()
        for i in range(points):
            npavg += self.tamatrix[:, i]

        print("The number of time points taken as background: "+str(i+1))
        npavg /= points
        #np.savetxt(self.filename+"_tamatrix_npavg", npavg, fmt='%1.5f')
        for x in range(self.tamatrix.shape[1]):
            self.bgcorr[:, x] = self.tamatrix[:, x] - npavg

        return self.bgcorr
    # zero time correction

filename = input("input the tamatrix filename without _tamatrix\n")
line_file = input("input the filename of output line\n")
matrix = tamatrix_importer()
matrix.import_data_silent(filename)
TAtime = matrix.tatime
TAwavelength=matrix.tawavelength
TAmatrix=matrix.auto_bgcorr(20)

fig, ax = plt.subplots()
# Create contour plot
Y, X = np.meshgrid(TAtime, TAwavelength)
contour= ax.contour(X,Y,TAmatrix,[-0.005,-0.001,-0.0005,0,0.0005,0.001,0.005])
plt.ylim(-1,1)

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


#plt.waitforbuttonpress()

while True:
    pts = []
    tellme('Left click to draw. \nRight click to remove. \nMiddle button to confirm')
    pts = np.asarray(plt.ginput(-1, timeout=-1,show_clicks=True))
    if len(pts) < 5:
        tellme('Too few points, starting over\n')
        time.sleep(1)  # Wait a second

    ln = plt.plot(pts[:, 0], pts[:, 1],'xr-')

    tellme('Happy? Key click for yes, mouse click for no\n')

    if plt.waitforbuttonpress():
        break

    # Get rid of line
    for p in ln:
        p.remove()

#Draw the correction line
np.savetxt(line_file,pts,fmt='%1.5f')


