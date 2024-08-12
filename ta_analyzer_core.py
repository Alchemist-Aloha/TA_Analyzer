import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget #uncomment for interactive plot
from matplotlib.colors import ListedColormap
import lmfit
from scipy.stats import norm
from tqdm import tqdm
import xarray as xr
'''

from glotaran.optimization.optimize import optimize
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.project.scheme import Scheme
'''
# load all and average the matrix


def mat_avg(name, select):
    """Average the TAmatrix of multiple experiments

    Args:
        name (str): The name of the file to be loaded. e.g. "expt_"
        select (list): A list of the selected experiments to be loaded. e.g. [0,2,3,5]. Note this will load expt_1, expt_3, expt_4, expt_6.

    Returns:
        str, str: The averaged matrix, the matrix array with all experiments loaded.
    """
    first_array = np.loadtxt(name+str(select[0]+1))
    rows, columns = first_array.shape
    mat_array = np.zeros((rows, columns, len(select)))
    for i, x in enumerate(select):
        mat_array[:, :, i] = np.loadtxt(name+str(x+1))
    sum_array = np.sum(mat_array, axis=2)
    avg_array = sum_array / len(select)
    np.savetxt(name+"averaged", avg_array, fmt="%f", delimiter="\t")
    return avg_array, mat_array

# load TaTime0


def load_tatime(mat):
    """Load the time axis of the TA matrix

    Args:
        mat (2darray): The TA matrix as a numpy array

    Returns:
        1darray: The time axis of the TA matrix
    """
    tatime = mat[:mat.shape[1]-2, 0]
    return tatime

# load tawavelength


def load_tawavelength(mat):
    """Load the wavelength axis of the TA matrix

    Args:
        mat (2darray): The TA matrix as a numpy array

    Returns:
        1darray: The wavelength axis of the TA matrix
    """
    tawavelength = mat[:, 1]
    return tawavelength


class load_single:
    """Class to load a single experiment TA spectrum
    """

    def __init__(self, file_name):
        """Initialize the class with the file name

        Args:
            file_name (str): The name of the file to be loaded. e.g. "expt_1"
        """
        data = np.loadtxt(file_name)
        self.tawavelength = data[:, 0]
        self.spec_ta = data[:, 1]
        self.spec_on = data[:, 2]
        self.spec_off = data[:, 3]
        self.ax = None

    def plot(self, ylim=None):
        """Plot the TA spectrum, ON and OFF spectrum

        Args:
            ylim (tuple, optional): y axis limit of TA spectrum. Defaults to None.
        """
        fig, self.ax = plt.subplots(nrows=2)
        self.ax[1].plot(self.tawavelength, self.spec_ta, label='TA')
        self.ax[0].plot(self.tawavelength, self.spec_on, label='ON')
        self.ax[0].plot(self.tawavelength, self.spec_off, label='OFF')
        self.ax[0].legend()
        self.ax[1].set_xlabel('Wavelength (nm)')
        self.ax[0].set_ylabel('ΔOD')
        self.ax[1].set_ylabel('ΔOD')
        self.ax[1].set_ylim(ylim)
        self.ax[1].set_title('TA spectrum')
        self.ax[0].set_title('On and Off spectrum')
        plt.show()


class load_spectra:
    def __init__(self, file_inp, num_spec=None, select=None):
        """ class to include single or multiple experiments (average) TA matrix

        Args:
            file_inp (str): The name of the file to be loaded. e.g. "expt_". if num_spec = 1, file_inp should use full name, e.g. "expt_3".
            num_spec (int, optional): num_spec is the number of experiments to be loaded. e.g. 5. Note this will load expt_1, expt_2, expt_3, expt_4, expt_5. Defaults to None.
            select (list, optional):  select is a list of the selected experiments to be loaded. e.g. [0,2,3,5]. 
            Note this will load expt_1, expt_3, expt_4, expt_6. select CANNOT be a one element list. 
            Use num_spec = 1 instead for single experiment. Defaults to None.
        """

        self.file_inp = file_inp
        if select is None and (num_spec is None or num_spec == 1):
            self.num_spec = 1
            self.tamatrix_avg = np.loadtxt(self.file_inp)
            self.tatime = load_tatime(self.tamatrix_avg)
            self.tawavelength = load_tawavelength(self.tamatrix_avg)
        elif select is not None:
            self.select = select
            self.num_spec = len(self.select)
            self.tamatrix_avg, self.mat_array = mat_avg(
                self.file_inp, self.select)
            # load tatime and tawavelength axes
            self.tatime = load_tatime(self.tamatrix_avg)
            self.tawavelength = load_tawavelength(self.tamatrix_avg)
        else:
            self.num_spec = num_spec
            # average the matrix
            self.select = range(self.num_spec)
            self.tamatrix_avg, self.mat_array = mat_avg(
                self.file_inp, self.select)
            # load tatime and tawavelength axes
            self.tatime = load_tatime(self.tamatrix_avg)
            self.tawavelength = load_tawavelength(self.tamatrix_avg)

    def mat_sub(self, obj_bg, modifier=None):
        """Subtract background from the TA matrix

        Args:
            obj_bg (load_spectra): load_spectra object of the blank background TA matrix
            modifier (float, optional): modifier applied to the blank for subtraction. Defaults to None.
        """
        if modifier is None:
            modifier = 1
        self.tamatrix_avg = self.tamatrix_avg - obj_bg.tamatrix_avg * modifier
        self.mat_array = self.mat_array - \
            obj_bg.tamatrix_avg[:, :, np.newaxis] * modifier

    # plot 1ps spectrum
    def get_1ps(self):
        """Get the 1ps spectrum and plot it

        Returns:
            1darray: 1ps spectrum
        """
        diff = np.abs(self.tatime - 1)
        pt = pt = np.argmin(diff)
        self.spec_1ps = self.tamatrix_avg[:, pt+2]
        self.fig_s, self.ax_s = plt.subplots()
        self.ax_s.plot(self.tawavelength, self.spec_1ps)
        self.ax_s.set_title(self.file_inp)
        self.ax_s.set_xlabel('Wavelength (nm)')
        self.ax_s.set_ylabel('ΔOD')
        return self.spec_1ps

    # plot multiple parallel traces to see photodamage
    def get_traces(self, wavelength, disable_plot=None):
        """Get the traces at a specific wavelength and plot them

        Args:
            wavelength (num): The wavelength to be plotted
            disable_plot (_type_, optional): Not in use currently. Defaults to None.

        Returns:
            2darray: The traces at the specific wavelength
        """
        self.fig_k, self.ax_k = plt.subplots()
        self.trace_array = np.zeros((len(self.tatime), self.num_spec))
        diff = np.abs(self.tawavelength - wavelength)
        pt = np.argmin(diff)
        if self.num_spec == 1:
            self.trace_avg = self.tamatrix_avg[pt, 2:]
            self.ax_k.plot(np.log(self.tatime), self.trace_avg,
                           label=f'{wavelength} nm trace')
        else:
            for i, x in enumerate(self.select):
                self.trace_array[:, i] = self.mat_array[pt, 2:, i]
                self.ax_k.plot(
                    np.log(self.tatime), self.trace_array[:, i], label=f'{wavelength} nm trace {x+1}')

            self.trace_avg = self.tamatrix_avg[pt, 2:]
            self.ax_k.plot(np.log(self.tatime), self.trace_avg,
                           label=f'{wavelength} nm trace averaged')
        self.ax_k.legend()
        self.ax_k.set_xlabel('Time (Log scale ps)')
        self.ax_k.set_ylabel('ΔOD')
        self.ax_k.set_title(self.file_inp)

        return self.trace_avg

    def correct_burn(self, wavelength, disable_plot=None):
        """Correct the sample burning (degredation) according to selected wavelength in the TA matrix. Savethe corrected matrix as a new TA matrix file

        Args:
            wavelength (num): The wavelength to be sampled for burning correction
            disable_plot (_type_, optional): Not in use. Defaults to None.
        """
        self.fig_b, self.ax_b = plt.subplots()
        self.trace_array = np.zeros((len(self.tatime), self.num_spec))
        burn_correction = np.zeros_like(self.tatime)
        pts_time = np.arange(len(self.tatime))
        diff = np.abs(self.tawavelength - wavelength)
        pt = np.argmin(diff)
        diff2 = np.abs(self.tatime - 1)
        pt2 = np.argmin(diff2)
        if self.num_spec == 1:
            print("single experiment. No burn correction")
        else:
            percent_per_point = (self.mat_array[pt, pt2+2, 0]
                                 - self.mat_array[pt, pt2+2, len(self.select)-1])/self.mat_array[pt, pt2+2, 0]/(len(self.tatime)*(len(self.select)-1))
            burn_correction = 1 + percent_per_point*pts_time
            self.ax_b.plot(pts_time, burn_correction, label='Burn correction')
            self.ax_b.legend()
            self.ax_b.set_xlabel('time point')
            self.ax_b.set_ylabel('correction')
            self.tamatrix_avg_burncorr = self.tamatrix_avg.copy()
            self.tamatrix_avg_burncorr[:, 2:] *= burn_correction
            np.savetxt(self.file_inp+"avg_burncorrected",
                       self.tamatrix_avg_burncorr, fmt="%f", delimiter="\t")


class compare_traces:
    """compare traces from load_spectra object
    """

    def __init__(self, obj, wavelength):
        """initialize the class with the first load_spectra object and wavelength to compare

        Args:
            obj (load_spectra): load_spectra object
            wavelength (num): wavelength to be compared
        """
        self.wavelength = wavelength
        self.tatime = obj.tatime
        trace = obj.get_traces(wavelength, disable_plot=True).reshape(1, -1)
        self.trace_array = np.empty((0, len(self.tatime)))
        print(self.trace_array.size)
        print(trace.size)
        self.trace_array = np.append(self.trace_array, trace, axis=0)
        self.wavelength_list = [self.wavelength]
        self.name_list = [obj.file_inp]

    def add_trace(self, obj, wavelength=None):
        """add traces from another load_spectra object

        Args:
            obj (Load_spectra): load_spectra object
            wavelength (num, optional): wavelength to be added if want to compare traces at diff wavelength. Defaults to None will use the wavelength from first object.
        """
        self.name_list.append(obj.file_inp)
        if wavelength is None:
            trace_toadd = obj.get_traces(
                self.wavelength, disable_plot=True).reshape(1, -1)
            self.wavelength_list.append(self.wavelength)
        else:
            try:
                trace_toadd = obj.get_traces(
                    wavelength, disable_plot=True).reshape(1, -1)
                self.wavelength_list.append(wavelength)
            except:
                print('Invalid wavelength')
                return
        self.trace_array = np.append(self.trace_array, trace_toadd, axis=0)

    def plot(self):
        """plot the loaded traces
        """
        self.fig, self.ax = plt.subplots()
        for i in range(len(self.trace_array)):
            self.ax.plot(np.log(self.tatime), self.trace_array[i, :]/np.max(np.abs(
                self.trace_array[i, :])), label=f'{self.name_list[i]} @ {self.wavelength_list[i]} nm')
        self.ax.legend()
        self.ax.set_title('Normalized traces with logarithmic time axis')
        self.ax.set_xlabel('Time (Log scale ps)')
        self.ax.set_ylabel('ΔOD')


class glotaran:
    """Class to export the IGOR generated TAmatrix to Glotaran input format
    """

    def __init__(self, matrix_corr, tatime, tawavelength):
        """Initialize the class with the TA matrix (Output from IGOR macro auto_tcorr. Without time and wavelength axis.
        NOT original TAMatrix like file), time axis and wavelength axis. Use SaveMatrix()macro in IGOR to get those inputs.
        Output file will be named as matrix_corr+"glo.ascii"

        Args:
            matrix_corr (str): The filename of the TA matrix file to be loaded. 
            tatime (str): The filename of the time axis
            tawavelength (str): The filename of the wavelength axis
        """
        self.filename = matrix_corr
        self.tatime = np.loadtxt(tatime)
        self.tawavelength = np.loadtxt(tawavelength)
        # np.genfromtext will read nan as nan, avoid size mismatch with np.loadtxt
        self.output_matrix = np.genfromtxt(
            matrix_corr, delimiter='\t', filling_values=np.nan)
        self.output_matrix = np.append(
            self.tatime.reshape(1, -1), self.output_matrix, axis=0)
        self.output_matrix = np.append(
            np.append("", self.tawavelength).reshape(1, -1).T, self.output_matrix, axis=1)
        self.header = self.filename + \
            '\n\nTime explicit\nintervalnr ' + str(len(self.tatime))
        np.savetxt(self.filename+"glo.ascii", self.output_matrix,
                   header=self.header, fmt='%s', comments='', delimiter='\t')


class load_glotaran:
    """Class to load the Glotaran input file. Output will be the time axis, wavelength axis and the TA matrix without time and wavelength axis
    """

    def __init__(self, dir):
        """Initialize the class with the Glotaran input file

        Args:
            dir (str): The filename of the Glotaran input file
        """
        self.filename = dir
        matrix = np.loadtxt(dir, skiprows=4, delimiter='\t', dtype=str)
        matrix[matrix == ''] = np.nan
        matrix = matrix.astype(np.float64)
        self.tatime = matrix[0, 1:]
        self.tawavelength = matrix[1:, 0]
        self.tamatrix = matrix[1:, 1:]


class plot_glotaran:
    def __init__(self, dir):
        """Initialize the class with the Glotaran output file. plot both traces and DASs
        Files: "_traces.ascii", "_DAS.ascii", "_summary.txt"
        dir is the directory of the file without the extension
        """
        self.filename = dir
        with open(dir + "_summary.txt", 'r') as file:
            find_rate = False
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith("Estimated Kinetic parameters: Dataset1:"):
                    # Split the line by spaces or commas and convert to float
                    rate_list = [
                        value for value in stripped_line.replace(',', ' ').split()]
                    find_rate = True
                if find_rate is True and stripped_line.startswith("Standard errors:"):
                    error_list = [
                        value for value in stripped_line.replace(',', ' ').split()]
                    find_rate = False
        # Convert the list of rate and error to a NumPy array
        self.rate_array = np.array(rate_list[4:]).astype(float)
        self.error_array = np.array(error_list[2:]).astype(float)
        # Load the DAS and traces data
        self.das = np.loadtxt(dir + "_DAS.ascii", skiprows=1)
        self.traces = np.loadtxt(dir + "_traces.ascii", skiprows=1)
        self.fig_das, self.ax_das = plt.subplots(figsize=(6, 3))
        self.fig_das.subplots_adjust(left=0.2)
        self.fig_traces, (self.ax_traces, self.ax_traces_2) = plt.subplots(
            1, 2, width_ratios=[0.3, 0.7], sharey=True, facecolor='w', figsize=(6, 3))
        self.fig_traces.subplots_adjust(wspace=0.1)
        if self.das.shape[1] != 2*self.rate_array.shape[0]:
            print("das and rate array size mismatch")
        for i in range(int(self.das.shape[1]/2)):
            self.ax_das.plot(self.das[:, 2*i], self.das[:, 2*i+1], label='Long-term' if 1 /
                             self.rate_array[i] > 10000.0 else f'{1/self.rate_array[i]:.2f} ps')
            colorwaves(self.ax_das)
            self.ax_das.legend()
            self.ax_das.set_xlabel('Wavelength (nm)')
            self.ax_das.set_ylabel('DAS')
            # print(self.das[:,i], self.das[:,i+1])
        self.ax_das.axhline(y=0, c="black", linewidth=0.5, zorder=0)

        for i in range(int(self.traces.shape[1]/2)):
            # p = find_closest_value([5],self.traces[:,0])[0]
            # time_log = np.concatenate((self.traces[:p,2*i],np.log10(self.traces[p:,2*i])),axis=0)
            self.ax_traces.plot(
                self.traces[:, 2*i], self.traces[:, 2*i+1], label=f'Trace {1/self.rate_array[i]:.2f} ps')
            self.ax_traces_2.plot(
                self.traces[:, 2*i], self.traces[:, 2*i+1], label=f'Trace {1/self.rate_array[i]:.2f} ps')
            self.ax_traces.set_xlim(-1, 1)
            self.ax_traces_2.set_xlim(1, len(self.traces[:, 2*i]))
            self.ax_traces.spines['right'].set_visible(False)
            self.ax_traces_2.spines['left'].set_visible(False)
            self.ax_traces.yaxis.tick_left()
            self.ax_traces.tick_params(labelright=False)
            self.ax_traces_2.tick_params(axis='y', labelleft=False)
            self.ax_traces_2.yaxis.tick_right()
            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            self.ax_traces.plot(
                [1, 1], [1, 0], transform=self.ax_traces.transAxes, **kwargs)
            self.ax_traces_2.plot(
                [0, 0], [0, 1], transform=self.ax_traces_2.transAxes, **kwargs)
            colorwaves(self.ax_traces)
            colorwaves(self.ax_traces_2)
            # self.ax_traces.plot(time_log, self.traces[:,2*i+1], label=f'Trace {1/self.rate_array[i]:.2f} ps')
            self.ax_traces_2.legend(loc='center right')
            self.ax_traces_2.set_xscale('log')
            self.ax_traces_2.set_xlabel('Time (ps)')
            self.ax_traces_2.xaxis.set_label_coords(0.2, -0.1)
            self.ax_traces.set_ylabel('Amplitude')


class tamatrix_importer:
    """Class to import the TA matrix from the file. The input may vary from filename to Load_spectra object or load_glotaran object
    """

    def __init__(self, filename=None, startnm=None, endnm=None, load_spectra=None, load_glotaran=None, tamatrix=None, tatime=None, tawavelength=None, name=None):
        """Initialize the class with the filename, start and end wavelength to be loaded. If no filename is given, the object can be loaded from Load_spectra object or load_glotaran object

        Args:
            filename (str, optional): The filename of the TA matrix file to be loaded. 
            startnm (num, optional): The start wavelength to be loaded. Defaults to 0.
            endnm (num, optional): The end wavelength to be loaded. Defaults to 1200 nm.
            load_spectra (load_spectra, optional): The load_spectra object to be loaded. 
            load_glotaran (load_glotaran, optional): The load_glotaran object to be loaded. 
            tamatrix (str, optional): The filename of the TA matrix file to be loaded. NOT WORKING.
            tatime (str, optional): The filename of the time axis file to be loaded. NOT WORKING.
            tawavelength (str, optional): The filename of the wavelength axis file to be loaded. NOT WORKING.
            name (str, optional): The name of the object. Defaults to filename.
        """
        if startnm is None:
            self.startnm = 0
        else:
            self.startnm = startnm
        if endnm is None:
            self.endnm = 1200
        else:
            self.endnm = endnm
        # load from file if no object is given
        if filename is not None:
            # Load firstcol wave and find startrow and endrow
            # filename = input("Enter the filename for firstcol wave: ")
            self.filename = filename
            firstcol = np.loadtxt(self.filename)[:, 1]
            if self.startnm < np.min(firstcol):
                self.startrow = np.argmin(firstcol)
            else:
                for index in range(len(firstcol)):
                    if firstcol[index] > self.startnm:
                        self.startrow = index
                        break
            if self.endnm > np.max(firstcol):
                self.endrow = np.argmax(firstcol)
            else:
                for index in range(len(firstcol)):
                    if firstcol[index] > self.endnm:
                        self.endrow = index
                        break

            # Load TAwavelength waves
            self.tawavelength = np.loadtxt(
                self.filename, skiprows=self.startrow, max_rows=self.endrow-self.startrow)[:, 1]
            # np.savetxt(self.filename+"_tawavelength",tawavelength,fmt='%1.5f')

            # Trim TAtime wave
            self.tatime = np.loadtxt(self.filename)[:, 0]
            idx = np.loadtxt(self.filename).shape[1]-2
            self.tatime = self.tatime[:idx]
            # np.savetxt(self.filename+"_tatime",tatime,fmt='%1.5f')

            # Load TAmatrix waves
            self.tamatrix = np.loadtxt(self.filename, skiprows=self.startrow,
                                       max_rows=self.endrow-self.startrow, usecols=np.arange(2, idx+2))
            # np.savetxt(self.filename+"_tamatrix",self.tamatrix,fmt='%1.5f'

        elif load_spectra is not None:
            self.tawavelength = load_spectra.tawavelength
            self.tatime = load_spectra.tatime
            self.tamatrix = load_spectra.tamatrix_avg[:, 2:]
            self.filename = load_spectra.file_inp

        elif load_glotaran is not None:
            self.tawavelength = load_glotaran.tawavelength
            self.tatime = load_glotaran.tatime
            self.tcorr = load_glotaran.tamatrix
            self.filename = load_glotaran.filename
        """ else:
            self.tawavelength = np.loadtxt(tawavelength)
            self.tatime = np.loadtxt(tatime)
            self.tcorr = np.loadtxt(tamatrix)
            self.filename = tamatrix """

        if name is not None:
            self.filename = name

        self.fit_results = {}

    def contour(self, time_min, time_max):
        """Create a contour plot

        Args:
            time_min (num): lower limit of time axis
            time_max (num): upper limit of time axis
        """
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        plt.contourf(X, Y, self.tamatrix,
                     [-0.005, -0.001, -0.0005, 0, 0.0005, 0.001, 0.005], cmap='rainbow')
        plt.ylim(time_min, time_max)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Time (ps)")
        plt.colorbar()
        plt.show()

    def save_all(self, filename, mat):
        """Save the time axis, wavelength axis and TA matrix. Saved files will be named as filename+"_tatime", filename+"_tawavelength", filename+"_tamatrix"

        Args:
            filename (str): directory to save the files. e.g. "C:/Users/xxx"
            mat (str): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        """
        if mat == 'original':
            matrix = self.tamatrix.copy()
            print("Original matrix is selected\n")
        elif mat == 'bgcorr':
            matrix = self.bgcorr.copy()
            print("Background corrected matrix is selected\n")
        else:
            matrix = self.tcorr.copy()
            print("Background and Zero time corrected matrix is selected\n")
        np.savetxt(filename+"_tawavelength", self.tawavelength, fmt='%1.5f')
        print(filename+"_tawavelength has been saved\n")
        np.savetxt(filename+"_tatime", self.tatime, fmt='%1.5f')
        print(filename+"_tatime has been saved\n")
        np.savetxt(filename+"_tamatrix", matrix, fmt='%1.5f')
        print(filename+"_tatime has been saved\n")

    def save_tamatrix(self, filename, mat=None):
        """Save the TA matrix. Saved file will be named as filename+"_tamatrix"

        Args:
            filename (str): directory to save the file. e.g. "C:/Users/xxx"
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to tcorr.
        """
        if mat is None:
            matrix = self.tcorr.copy()
            print("Background and Zero time corrected matrix is selected\n")
        if mat == 'original':
            matrix = self.tamatrix.copy()
            print("Original matrix is selected\n")
        elif mat == 'bgcorr':
            matrix = self.bgcorr.copy()
            print("Background corrected matrix is selected\n")
        else:
            matrix = self.tcorr.copy()
            print("Background and Zero time corrected matrix is selected\n")
        if filename == self.filename:
            print("Cannot overwrite original matrix, choose another name\n")
        else:
            np.savetxt(filename+"_tamatrix", matrix, fmt='%1.5f')

    def save_tatime(self, filename):
        """Save the time axis. Saved file will be named as filename+"_tatime"

        Args:
            filename (str): directory to save the file. e.g. "C:/Users/xxx"
        """
        np.savetxt(filename+"_tatime", self.tatime, fmt='%1.5f')

    def save_tawavelength(self, filename):
        """Save the wavelength axis. Saved file will be named as filename+"_tawavelength"

        Args:
            filename (str): directory to save the file. e.g. "C:/Users/xxx"
        """
        np.savetxt(filename+"_tawavelength", self.tawavelength, fmt='%1.5f')

    def save_axes(self, filename):
        """Save the time and wavelength axes. Saved files will be named as filename+"_tatime" and filename+"_tawavelength"

        Args:
            filename (str): directory to save the files. e.g. "C:/Users/xxx"
        """
        np.savetxt(filename+"_tatime", self.tatime, fmt='%1.5f')
        np.savetxt(filename+"_tawavelength", self.tawavelength, fmt='%1.5f')

    def auto_bgcorr(self, points):
        """Background correction of the TA matrix using the negative time points. The number of time points taken as background should be given as input

        Args:
            points (int): The number of time points taken as background

        Returns:
            2darray: The background corrected TA matrix
        """
        npavg = 0
        self.bgcorr = self.tamatrix.copy()
        for i in range(points):
            npavg += self.tamatrix[:, i]

        print("The number of time points taken as background: "+str(i+1))
        npavg /= points
        # np.savetxt(self.filename+"_tamatrix_npavg", npavg, fmt='%1.5f')
        for x in range(self.tamatrix.shape[1]):
            self.bgcorr[:, x] = self.tamatrix[:, x] - npavg
        # np.savetxt(self.tamatrix+"_tamatrix_bgcorr", self.bgcorr, fmt='%1.5f')

        # Create a figure and axis for the contour plot
        # Create contour plot
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        plt.contourf(X, Y, self.bgcorr, [-0.005, -0.001, -0.0005,
                                         0, 0.0005, 0.001, 0.005], cmap='rainbow')
        plt.ylim(-1, 1)
        plt.colorbar()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Time (ps)")
        plt.show()

        return self.bgcorr

    def auto_tcorr(self, line_file):
        """This macro corrects a TA matrix (time versus wavelength or wavenumber) for the measured chirp in the continuum.
        To use it first plot the TA matrix as a contour plot and zoom in on the region around zerotime, say -1 ps to +1 ps.
        Then use the "correction_line.py" tool to draw in a contour that follows the curvature of the onset of the signal.
        Typically this curve would be lower, at more negative times, in the blue part of the spectrum and then extending up
        to more positive time delays in the red part of the spectrum.
        To get the "Draw Wave Monotonic" tool, first <Show Tools>, then right click + option button on the polygon tool. Select
        "Draw wave Monotonic" and then click along to place data points on the contour plot, double clicking when you reach the last one.
        This will make two waves, named something like W_Ypoly0 (the y points) and W_XPoly0 (the x points).
        "matrix" is the name of the TA matrix you are correcting, e.g. "TAmatrix0".
        "TAtime" is the name if the time axis, e.g. "TAtime0"
        "TAwavelength" is the name of the wavelength axis, e.g. "tawavelength"
        "zerotime_x" is the wave with the x values of the zerotime wave drawn above, e.g. "W_XPoly0"
        "zerotime_y" is the wave with the y values of the zerotime wave drawn above, e.g. "W_YPoly0"
        Note that you could also fit the kinetics at a lot of different wavelengths and thereby determine a series of zerotimes ("zerotime_y")
        at a series of wavelengths ("zerotime_x")
        So you'd call this macro with a command line like: Correct_zerotime("TAmatrix0","TAtime0", "tawavelength","W_XPoly0","W_YPoly0")

        Args:
            line_file (str): The filename of the line drawn for zero time correction

        Returns:
            2darray: The zero time corrected TA matrix
        """

        # import correction line from drawing script
        zerotime_x = np.loadtxt(line_file)[:, 0]
        zerotime_y = np.loadtxt(line_file)[:, 1]
        # generate contempory file and output matrix
        time_temp = self.tatime.copy()
        try:
            oldmatrix = self.bgcorr.copy()
        except:
            print('bgcorr matrix not found. zero time correction on original matrix')
            oldmatrix = self.tamatrix.copy()
        self.tcorr = np.zeros_like(self.tamatrix)

        for i in range(len(self.tawavelength)):
            # Tatime axis minus time offset from the drawn line
            time_temp = self.tatime + \
                np.interp(self.tawavelength[i], zerotime_x, zerotime_y)
            # extrapolate TAsignal to match corrected time axis
            self.tcorr[i, :] = np.interp(
                time_temp, self.tatime, oldmatrix[i, :])
            # Add the following smoothing line to clean up the spectra, with a slight loss in time resolution.
            # newmatrix[i, :] = np.convolve(kin_temp2[i, :], np.ones(3)/3, mode='same')
        # save tcorred matrix
        # np.savetxt(newmatrixname, newmatrix, fmt='%1.5f')

        # Create contour plot with plot_contour()
        # Create contour plot
        fig, ax = plt.subplots()
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        contour = ax.contour(
            X, Y, self.tcorr, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01])
        plt.ylim(-1, 1)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Time (ps)")
        plt.draw()

        return self.tcorr

    def auto_tcorr_fit(self):
        """
        This function uses fitted cross correlation curve to do zero-time correction. Need to run fit_correlation() to get good fit first.

        Returns:
            2darray: The zero time corrected TA matrix
        """

        # import correction line from drawing script
        zerotime_x = self.t0_list[0]
        zerotime_y = self.t0_list[2]
        # generate contempory file and output matrix
        time_temp = self.tatime.copy()
        try:
            oldmatrix = self.bgcorr.copy()
        except:
            print('bgcorr matrix not found. zero time correction on original matrix')
            oldmatrix = self.tamatrix.copy()
        self.tcorr = np.zeros_like(self.tamatrix)

        for i in range(len(self.tawavelength)):
            # Tatime axis minus time offset from the drawn line
            time_temp = self.tatime + \
                np.interp(self.tawavelength[i], zerotime_x, zerotime_y)
            # extrapolate TAsignal to match corrected time axis
            self.tcorr[i, :] = np.interp(
                time_temp, self.tatime, oldmatrix[i, :])
            # Add the following smoothing line to clean up the spectra, with a slight loss in time resolution.
            # newmatrix[i, :] = np.convolve(kin_temp2[i, :], np.ones(3)/3, mode='same')
        # save tcorred matrix
        # np.savetxt(newmatrixname, newmatrix, fmt='%1.5f')

        # Create contour plot with plot_contour()
        # Create contour plot
        fig, ax = plt.subplots()
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        contour = ax.contour(
            X, Y, self.tcorr, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01])
        plt.ylim(-1, 1)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Time (ps)")
        plt.draw()

        return self.tcorr

    def glotaran(self):
        """Export the background and zerotime corrected TA matrix to Glotaran input format (legacy JAVA version). Saved file will be named as filename+"glo.ascii"
        Don't need this array in pyglotaran.
        """
        output_matrix = glotaran(self.tcorr, self.tatime, self.tawavelength)

    def pyglotaran(self, mat=None):
        '''
        export tcorr matrix to pyglotaran xarray dataset
        e.g.
        dataset = tamatrix.pyglotaran()

        Args:
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        return:
            xarray dataset
        '''
        time_vals = self.tatime
        spectral_vals = self.tawavelength
        data_vals = self.mat_selector(mat).T

        # Define dimensions and coordinates
        dims = ('time', 'spectral')
        coords = {'time': time_vals, 'spectral': spectral_vals}

        # Create xarray dataset
        dataset = xr.Dataset(
            {'data': (dims, data_vals)},
            coords=coords,
            attrs={'source_path': 'dataset_1.nc'}
        )
        print(dataset)
        return dataset

    def mat_selector(self, mat=None):
        """Helper function to select the matrix to be used for the analysis. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.

        Args:
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.

        Returns:
            2darray: The selected matrix
        """
        if mat is None:
            try:
                matrix = self.tcorr.copy()
            except:
                try:
                    matrix = self.bgcorr.copy()
                    print('Background corrected matrix used')
                except:
                    matrix = self.tamatrix.copy()
                    print('Original matrix used')
        elif mat == 'original':
            matrix = self.tamatrix.copy()
        elif mat == 'bgcorr':
            matrix = self.bgcorr.copy()
        elif mat == 'tcorr':
            matrix = self.tcorr.copy()
        else:
            try:
                matrix = self.tcorr.copy()
                print('Invalid mat value. Use tcorrrected matrix')
            except:
                try:
                    matrix = self.bgcorr.copy()
                    print('Invalid mat value. Background corrected matrix used')
                except:
                    matrix = self.tamatrix.copy()
                    print('Invalid mat value. Original matrix used')
        return matrix

    def auto_taspectra(self, time_pts=None, mat=None):
        """Plot the TA spectra at selected time points

        Args:
            time_pts (list, optional): The time points to be plotted. Defaults to [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500].
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.

        Returns:
            1darray, 1darray: The spectra set, the index of the time points
        """

        if time_pts is None:
            time_pts = [-0.5, -0.2, 0, 0.1, 0.2, 0.5, 1, 2,
                        5, 10, 20, 50, 100, 200, 500, 1000, 1500]
        '''
        if mat is None:
            try:   
                matrix = self.tcorr.copy()
            except:
                try:
                    matrix = self.bgcorr.copy()
                    print('Background corrected matrix used')
                except:
                    matrix = self.tamatrix.copy()
                    print('Original matrix used')
        elif mat == 'original':
            matrix = self.tamatrix.copy()
        elif mat == 'bgcorr':
            matrix = self.bgcorr.copy()
        elif mat == 'tcorr':
            matrix = self.tcorr.copy()
        else:
            try:   
                matrix = self.tcorr.copy()
                print('Invalid mat value. Use tcorrrected matrix')
            except:
                try:
                    matrix = self.bgcorr.copy()
                    print('Invalid mat value. Background corrected matrix used')
                except:
                    matrix = self.tamatrix.copy()
                    print('Invalid mat value. Original matrix used')
        '''
        matrix = self.mat_selector(mat)
        # find closest time points
        time_index = find_closest_value(time_pts, self.tatime)
        colors = plt.cm.rainbow(np.linspace(1, 0, len(time_index)))
        cmap = ListedColormap(colors)
        self.spectra_set = self.tawavelength.copy()
        plt.figure(figsize=(7, 4))
        # plot spectra together
        for i in range(len(time_index)):
            spec = matrix[:, time_index[i]]
            self.spectra_set = np.c_[self.spectra_set, spec]
            plt.plot(self.tawavelength, spec, label='{:.2f}'.format(
                self.tatime[time_index[i]])+" ps", color=cmap(i), linewidth=0.5)
        # plt.ylim(-0.05,0.05)
        plt.title(self.filename)
        plt.rcParams.update({
            'font.size': 8,      # Default font size
            'axes.labelsize': 8,  # Label size for x and y axes
            'axes.titlesize': 8,  # Title size
            'xtick.labelsize': 8,  # Tick label size for x axis
            'ytick.labelsize': 8  # Tick label size for y axis
        })
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("ΔOD")
        plt.legend(loc='best')
        plt.show()
        return self.spectra_set, time_index

    def save_taspectra(self, name=None, time_pts=None, mat=None):
        """Save the TA spectra at selected time points. Saved file will be named as s_name

        Args:
            name (str, optional): The name of the file. Defaults to tamatrix_importer.filename.
            time_pts (list, optional): The time points to be plotted. Defaults to [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500].
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        """
        if name is None:
            name = self.filename
        self.spectra_set, time_index = self.auto_taspectra(time_pts, mat)
        header_str = 'Wavelength\t'
        for time in time_index:
            header_str = header_str+"s_"+name+"_" + \
                '{:.2f}'.format(self.tatime[time])+" ps\t"
        np.savetxt("s_"+name, self.spectra_set,
                   header=header_str, fmt='%1.5f', delimiter='\t')
        print("File s_"+name+" has been saved\n")

    def get_spectrum(self, name, time_pt, mat=None,):
        matrix = self.mat_selector(mat)
        diff = self.tatime.copy() - time_pt
        index = np.argmin(np.abs(diff))
        np.savetxt(
            "s_"+name+"_"+'{:.2f}'.format(self.tatime[index])+" ps", matrix[:, index], fmt='%1.5f')
        print("File s_"+name+"_" +
              '{:.2f}'.format(self.tatime[index])+" ps has been saved\n")
        plt.plot(self.tawavelength, matrix[:, index], label='{:.2f}'.format(
            self.tatime[index])+" ps")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return matrix[:, index]

    def return_spectrum(self, tamatrix, time_pt):
        diff = self.tatime.copy() - time_pt
        index = np.argmin(np.abs(diff))
        plt.plot(self.tawavelength, tamatrix[:, index], label='{:.2f}'.format(
            self.tatime[index])+"ps")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return tamatrix[:, index]

    def auto_takinetics(self, wavelength_pts, mat=None):
        # sample time_pts = [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500]
        # find closest time points
        wavelength_index = find_closest_value(
            wavelength_pts, self.tawavelength)
        matrix = self.mat_selector(mat)
        # plot spectra together
        plt.figure(figsize=(7, 4))
        for i in range(len(wavelength_index)):
            spec = matrix[wavelength_index[i], :].T
            plt.plot(self.tatime, spec, label='{:.2f}'.format(
                self.tawavelength[wavelength_index[i]])+" nm", linewidth=1)
        plt.title(self.filename)
        plt.rcParams.update({
            'font.size': 8,      # Default font size
            'axes.labelsize': 8,  # Label size for x and y axes
            'axes.titlesize': 8,  # Title size
            'xtick.labelsize': 8,  # Tick label size for x axis
            'ytick.labelsize': 8  # Tick label size for y axis
        })
        plt.xlabel("Time (ps)")
        plt.ylabel("ΔOD")
        plt.xlim(-1, 100)
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
        plt.legend(loc='best')
        plt.show()

    def save_takinetics(self, name, wavelength_pts, mat):
        # sample time_pts = [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500]
        # find closest time points
        wavelength_index = find_closest_value(
            wavelength_pts, self.tawavelength)
        self.kinetics_set = self.tatime.copy()
        header_str = 'Time(ps)\t'
        for wavelength in wavelength_index:
            header_str = header_str+"k_"+name+"_" + \
                '{:.2f}'.format(self.tawavelength[wavelength])+"nm\t"
        if mat == 'original':
            matrix = self.tamatrix.copy()
        elif mat == 'bgcorr':
            matrix = self.bgcorr.copy()
        else:
            matrix = self.tcorr.copy()
            print("output zero time corrected kinetics")
        # plot spectra together
        for i in range(len(wavelength_index)):
            spec = matrix[wavelength_index[i], :].T
            self.kinetics_set = np.c_[self.kinetics_set, spec]
            plt.plot(self.tatime, spec, label='{:.2f}'.format(
                self.tawavelength[wavelength_index[i]])+" nm")
        np.savetxt("k_"+name, self.kinetics_set,
                   header=header_str, fmt='%1.5f', delimiter='\t')
        plt.xlabel("Time (ps)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()

        return self.kinetics_set

    def get_kinetic(self, name, wavelength_pt, mat=None):
        matrix = self.mat_selector(mat)
        # plot spectra together
        diff = np.abs(self.tawavelength - wavelength_pt)
        wavelength_index = np.argmin(np.abs(diff))
        plt.plot(self.tatime, matrix[wavelength_index, :], label='{:.2f}'.format(
            self.tawavelength[wavelength_index])+" nm")
        np.savetxt("k_"+name+"_"+'{:.2f}'.format(
            self.tawavelength[wavelength_index])+"nm", matrix[wavelength_index, :].T, fmt='%1.5f', delimiter='\t')
        plt.xlabel("Time (ps)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return matrix[wavelength_index, :]

    def return_kinetic(self, tamatrix, wavelength_pt):
        # plot spectra together
        diff = np.abs(self.tawavelength - wavelength_pt)
        wavelength_index = np.argmin(np.abs(diff))
        plt.plot(self.tatime, tamatrix[wavelength_index, :], label='{:.2f}'.format(
            self.tawavelength[wavelength_index])+" nm")
        plt.xlabel("Time (ps)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return tamatrix[wavelength_index, :]

    def fit_kinetic(self, wavelength, num_of_exp=None, mat=None, params=None, time_split=None):
        if params is None:
            params = params_init(num_of_exp)
        matrix = self.mat_selector(mat)
        # plot spectra together
        diff = np.abs(self.tawavelength - wavelength)
        wavelength_index = np.argmin(np.abs(diff))
        y = matrix[wavelength_index, :]
        t = self.tatime
        lmodel = lmfit.Model(multiexp_func)
        result = lmodel.fit(y, params=params, t=t, max_nfev=100000,
                            ftol=1e-9, xtol=1e-9, nan_policy='omit')
        # print(result.fit_report())
        print('-------------------------------')
        print(
            f'{self.filename} kinetics fit at {self.tawavelength[wavelength_index]:.2f} nm')
        print('-------------------------------')
        print(f'chi-square: {result.chisqr:11.6f}')
        pearsonr = np.corrcoef(result.best_fit, y)[0, 1]
        print(f'Pearson\'s R: {pearsonr:11.6f}')
        print('-------------------------------')
        print('Parameter    Value       Stderr')
        for name, param in result.params.items():
            print(f'{name:7s} {param.value:11.6f} {param.stderr:11.6f}')
        print('-------------------------------')
        # result.plot_fit()
        if time_split is None:
            pt_split = find_closest_value([5], self.tatime)[0]
        else:
            pt_split = find_closest_value([time_split], self.tatime)[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                       gridspec_kw={'width_ratios': [2, 3]})
        fig.subplots_adjust(wspace=0.05)

        ax1.scatter(t[:pt_split], y[:pt_split], marker='o', color='black')
        ax1.plot(t[:pt_split], result.best_fit[:pt_split], color='red')
        ax1.set_xlim(t[0], t[pt_split-1])
        # ax1.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(right=False)

        ax2.scatter(t[pt_split:], y[pt_split:], marker='o', color='black',
                    label=f"{self.tawavelength[wavelength_index]:.2f} nm")
        ax2.plot(t[pt_split:], result.best_fit[pt_split:], color='red',
                 label=f"{self.tawavelength[wavelength_index]:.2f} nm fit")
        ax2.set_xscale('log')
        ax2.set_xlim(t[pt_split-1], t[-1])
        # ax2.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(left=False)

        # Creating a gap between the subplots to indicate the broken axis
        gap = 0.1
        ax1.spines['right'].set_position(('outward', gap))
        ax2.spines['left'].set_position(('outward', gap))
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        # Centered title above subplots
        fig.suptitle(self.filename, fontsize=10, ha='center')
        plt.legend(loc='best')
        fig.text(0.5, 0.04, 'Time (ps)', ha='center', fontsize=8)
        ax1.set_ylabel("ΔOD")
        plt.show()
        self.fit_results[str(wavelength)] = [y, result.best_fit, result]
        return result

    def plot_fit(self, time_split=None):
        colors = plt.cm.rainbow(np.linspace(1, 0, len(self.fit_results)))
        cmap = ListedColormap(colors)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                       gridspec_kw={'width_ratios': [2, 3]})
        fig.subplots_adjust(wspace=0.05)
        if time_split is None:
            pt_split = find_closest_value([5], self.tatime)[0]
        else:
            pt_split = find_closest_value([time_split], self.tatime)[0]
        for i, (key, value) in enumerate(self.fit_results.items()):
            # ax1.scatter(self.tatime,value[0], facecolor='none',marker = 'o',s = 50, edgecolor =cmap(i))
            # ax1.plot(self.tatime,value[1], color =cmap(i), label = f"{key} nm")
            ax1.scatter(self.tatime[:pt_split], value[0][:pt_split],
                        facecolor='none', marker='o', s=50, edgecolor=cmap(i))
            ax1.plot(self.tatime[:pt_split], value[1]
                     [:pt_split], color=cmap(i))

            # ax2.scatter(self.tatime,value[0], facecolor='none',marker = 'o',s = 50, edgecolor =cmap(i))
            # ax2.plot(self.tatime,value[1], color =cmap(i), label = f"{key} nm")
            ax2.scatter(self.tatime[pt_split:], value[0][pt_split:],
                        facecolor='none', marker='o', s=50, edgecolor=cmap(i))
            ax2.plot(self.tatime[pt_split:], value[1]
                     [pt_split:], color=cmap(i), label=f"{key} nm")
            ax2.set_xscale('log')

        ax2.set_xlim(self.tatime[pt_split-1], self.tatime[-1])
        # ax2.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(left=False)

        ax1.set_xlim(self.tatime[0], self.tatime[pt_split-1])
        # ax1.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(right=False)
        # Creating a gap between the subplots to indicate the broken axis
        gap = 0.1
        ax1.spines['right'].set_position(('outward', gap))
        ax2.spines['left'].set_position(('outward', gap))
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        # Centered title above subplots
        fig.suptitle(self.filename, fontsize=10, ha='center')
        plt.legend(loc='best')
        fig.text(0.5, 0.04, 'Time (ps)', ha='center', fontsize=8)
        ax1.set_ylabel("ΔOD")
        plt.show()

    def fit_correlation(self, num_of_exp):
        self.t0_list = np.empty((3, 0))
        t = self.tatime
        lmodel = lmfit.Model(multiexp_func)
        params = params_init(num_of_exp)
        t1 = find_closest_value([1], self.tatime)
        tamax = np.max(np.abs(self.bgcorr[:, t1]))
        wlmax = np.argmax(np.abs(self.bgcorr[:, t1]))
        y = self.bgcorr[wlmax, :]
        result = lmodel.fit(y, params=params, t=t, method='powell',
                            max_nfev=1000000, nan_policy='omit')
        params.update(result.params)
        for i in tqdm(range(self.bgcorr.shape[0]), desc='Fitting'):
            if np.abs(self.bgcorr[i, t1]) > 0.1*tamax and i % 20 == 0:
                y = self.bgcorr[i, :]
                result = lmodel.fit(
                    y, params=params, t=t, method='powell', max_nfev=1000000, nan_policy='omit')
                rms = result.chisqr
                if result.success and rms < 0.15:  # Check if the fit was successful
                    self.t0_list = np.append(self.t0_list, np.array(
                        [[self.tawavelength[i]], [result.params['w10'].value], [rms]]), axis=1)
                    params.update(result.params)
        fit = polyfit(self.t0_list[1], self.t0_list[0], self.t0_list[2])
        self.t0_list[2] = fit
        fig, ax = plt.subplots()
        ax.plot(self.t0_list[0], self.t0_list[1])
        ax.plot(self.t0_list[0], self.t0_list[2])
        plt.show()


def find_closest_value(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    closest = [0] * len(array1)  # Initialize closest list with zeros
    for i in range(len(array1)):
        difference = array2 - array1[i]
        # Use np.abs to get the absolute difference
        closest[i] = np.argmin(np.abs(difference))
    # Remove same elements
    closest_2 = []
    for x in closest:
        if x not in closest_2:
            closest_2.append(x)
    return closest_2


# Plot contour from files
def plot_contour_file(tatime_file, tawavelength_file, tamatrix_file, max_point):
    tatime = np.loadtxt(tatime_file)
    tawavelength = np.loadtxt(tawavelength_file)
    tamatrix = np.loadtxt(tamatrix_file)
    # Create contour plot
    Y, X = np.meshgrid(tatime, tawavelength)
    plt.contourf(
        X, Y, tamatrix, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01], cmap='rainbow')
    plt.colorbar()
    plt.ylim(-1, max_point)
    plt.show()


# Plot contour with numpy arrays
def plot_contour(tatime, tawavelength, tamatrix, max_point):
    # Create contour plot
    Y, X = np.meshgrid(tatime, tawavelength)
    plt.contourf(
        X, Y, tamatrix, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01], cmap='rainbow')
    plt.colorbar()
    plt.ylim(-1, max_point)
    plt.show()


def save_txt(array, file):
    np.savetxt(array, file, fmt='%1.5f')


def polynomial_func(x, a, b, c):
    return a/(1e-9+(x)**2) + c


def polyfit(y, x, weights):
    # Creating a Model object with the quadruple function
    poly_model = lmfit.Model(polynomial_func)

    # Creating Parameters and adding them to the model
    params = lmfit.Parameters()
    params.add('a', value=1.0)
    params.add('b', value=1.0)
    params.add('c', value=1.0)

    # Fitting the model to the data
    result = poly_model.fit(y, params, method='powell', x=x, weights=1/weights)
    fitted_curve = result.best_fit
    print(result.params)
    return fitted_curve


def multiexp_func(t, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12):
    w = [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12]
    sigma = np.sqrt(w[0]**2) / (2 * np.sqrt(2 * np.log(2)))
    result = np.zeros_like(t)  # initialize result

    if w[3] == 0:
        exp1 = np.zeros_like(t)
    else:
        k0 = 1/w[3]
        exp1 = w[2]*np.exp(-(t-w[10])*(k0))*norm.cdf(t-w[10], scale=sigma)

    if w[5] == 0:
        exp2 = np.zeros_like(t)
    else:
        k1 = 1/w[5]
        exp2 = w[4]*np.exp(-(t-w[10])*(k1))*norm.cdf(t-w[10], scale=sigma)

    if w[7] == 0:
        exp3 = np.zeros_like(t)
    else:
        k2 = 1/w[7]
        exp3 = w[6]*np.exp(-(t-w[10])*(k2))*norm.cdf(t-w[10], scale=sigma)

    if w[9] == 0:
        exp4 = np.zeros_like(t)
    else:
        k3 = 1/w[9]
        exp4 = w[8]*np.exp(-(t-w[10])*(k3))*norm.cdf(t-w[10], scale=sigma)

    result += exp1+exp2+exp3+exp4
    result += w[11]+w[12]*norm.cdf(t-w[10], scale=sigma)
    result *= 1
    # b=4*np.log1p(2)/(w[0]**2)
    return result


def params_init(num_of_exp):
    params = lmfit.Parameters()
    params.add('w0', value=0.1, min=0.05, max=0.2)
    params.add('w1', value=1.0, vary=False)
    params.add('w2', value=0, min=-0.2, max=0.2)
    params.add('w3', value=1, min=0.01, max=5000)
    if num_of_exp == 1:
        params.add('w4', value=0, min=-0.2, max=0.2, vary=False)
        params.add('w5', value=10, min=0.01, max=5000, vary=False)
        params.add('w6', value=0, min=-0.2, max=0.2, vary=False)
        params.add('w7', value=50, min=0.01, max=5000, vary=False)
        params.add('w8', value=0, min=-0.2, max=0.2, vary=False)
        params.add('w9', value=500, min=0.01, max=5000, vary=False)

    if num_of_exp == 2:
        params.add('w4', value=0, min=-0.2, max=0.2, vary=True)
        params.add('w5', value=10, min=0.01, max=5000, vary=True)
        params.add('w6', value=0, min=-0.2, max=0.2, vary=False)
        params.add('w7', value=50, min=0.01, max=5000, vary=False)
        params.add('w8', value=0, min=-0.2, max=0.2, vary=False)
        params.add('w9', value=500, min=0.01, max=5000, vary=False)

    if num_of_exp == 3:
        params.add('w4', value=0, min=-1.0, max=1.0, vary=True)
        params.add('w5', value=10, min=0.01, max=5000, vary=True)
        params.add('w6', value=0, min=-1.0, max=1.0, vary=True)
        params.add('w7', value=50, min=0.01, max=5000, vary=True)
        params.add('w8', value=0, min=-1.0, max=1.0, vary=False)
        params.add('w9', value=500, min=0.01, max=5000, vary=False)

    if num_of_exp == 4:
        params.add('w4', value=0, min=-1.0, max=1.0, vary=True)
        params.add('w5', value=10, min=0.01, max=5000, vary=True)
        params.add('w6', value=0, min=-1.0, max=1.0, vary=True)
        params.add('w7', value=50, min=0.01, max=5000, vary=True)
        params.add('w8', value=0, min=-1.0, max=1.0, vary=True)
        params.add('w9', value=500, min=0.01, max=5000, vary=True)

    params.add('w10', value=0.0, min=-0.5, max=0.5)
    params.add('w11', value=0.0, min=-0.1, max=0.1)
    params.add('w12', value=0.0, min=-0.5, max=0.5, vary=False)
    return params


def colorwaves(ax):
    """
    Change the colors of the lines in the given Axes object.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object containing the lines.
    colors (list of str): A list of colors to apply to the lines.
    """
    # Ensure the number of colors matches the number of lines
    lines = ax.get_lines()
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
              '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

    # Set the color for each line
    for i, line in enumerate(lines):
        line.set_color(colors[i])
    # ax.legend()
