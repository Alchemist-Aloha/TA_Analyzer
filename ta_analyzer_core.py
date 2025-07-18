from pathlib import Path
import lmfit
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import xarray as xr
import re
from scipy.stats import norm

# %matplotlib widget #uncomment for interactive plot
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from itertools import cycle  # Import cycle

"""

from glotaran.optimization.optimize import optimize
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.project.scheme import Scheme
"""
__docformat__ = "google"


def mat_avg(name: Path, select: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Average the TA matrice of multiple experiments.

    Args:
        name (Path): The path of the file to be loaded (e.g., "dir/expt_").
        select (list): A list of indices for the selected experiments to be loaded.
                       For example, [0, 2, 3, 5] will load files like "expt_1", "expt_3", "expt_4", "expt_6".

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The averaged matrix (2D numpy array).
            - The 3D matrix array with all experiments loaded.

    Notes:
        - The function attempts to load files using `Path` methods. If it fails, it falls back to string-based file paths.
        - The averaged matrix is saved as a new file with "averaged" appended to the base name.
    """
    try:
        first_array = np.loadtxt(name.with_name(name.stem + str(select[0] + 1)))
    except Exception as e:
        print(f"Error in loading file using Pathlib: {e}")
        first_array = np.loadtxt(str(name) + str(select[0] + 1))
    rows, columns = first_array.shape
    mat_array = np.zeros((rows, columns, len(select)))
    for i, x in enumerate(select):
        try:
            mat_array[:, :, i] = np.loadtxt(name.with_name(name.stem + str(x + 1)))
        except Exception as e:
            print(f"Error in loading file using Pathlib: {e}")
            mat_array[:, :, i] = np.loadtxt(str(name) + str(x + 1))
    sum_array = np.sum(mat_array, axis=2)
    avg_array = sum_array / len(select)
    try:
        np.savetxt(
            name.with_name(name.stem + "averaged"), avg_array, fmt="%f", delimiter="\t"
        )
    except Exception as e:
        print(f"Error in saving file using Pathlib: {e}")
        np.savetxt(str(name) + "averaged", avg_array, fmt="%f", delimiter="\t")
    return avg_array, mat_array


def load_tatime(mat: np.ndarray) -> np.ndarray:
    """Load the time axis TATime0 of the TA matrix

    Args:
        mat (np.ndarray): The TA matrix as a numpy array

    Returns:
        np.ndarray: The time axis of the TA matrix
    """
    tatime = mat[: mat.shape[1] - 2, 0]
    return tatime


def load_tawavelength(mat: np.ndarray) -> np.ndarray:
    """Load the wavelength axis TAWavelength0 of the TA matrix

    Args:
        mat (2darray): The TA matrix as a numpy array

    Returns:
        1darray: The wavelength axis of the TA matrix
    """
    tawavelength = mat[:, 1]
    return tawavelength


class load_single:
    """
    Load and process a single TA spectrum file.

    Args:
        file_name (str | Path): The name of the file to be loaded (e.g., "expt_1").

    Attributes:
        filename (Path): The full path of the file.
        filestem (str): The stem (name without extension) of the file.
        tawavelength (np.ndarray): The wavelength axis of the TA spectrum.
        spec_ta (np.ndarray): The TA spectrum data.
        spec_on (np.ndarray): The "ON" spectrum data.
        spec_off (np.ndarray): The "OFF" spectrum data.
        ax (list | None): Axes for plotting.

    Methods:
        plot(ylim: tuple[float, float] | None = None) -> None:
            Plot the TA spectrum along with the ON and OFF spectra.

    Notes:
        - The class automatically loads and processes the file upon initialization.
        - The `plot` method provides an option to set y-axis limits for the TA spectrum.
    """

    def __init__(self, file_name: str | Path) -> None:
        self.filename = Path(file_name)
        self.filestem = self.filename.stem
        data = np.loadtxt(self.filename)
        self.tawavelength = data[:, 0]
        self.spec_ta = data[:, 1]
        self.spec_on = data[:, 2]
        self.spec_off = data[:, 3]
        self.ax = None

    def plot(self, ylim: tuple[float, float] | None = None) -> None:
        """Plot the TA spectrum, ON and OFF spectrum

        Args:
            ylim (tuple, optional): y axis limit of TA spectrum. Defaults to None.
        """
        fig, self.ax = plt.subplots(nrows=2)
        self.ax[1].plot(self.tawavelength, self.spec_ta, label="TA")
        self.ax[0].plot(self.tawavelength, self.spec_on, label="ON")
        self.ax[0].plot(self.tawavelength, self.spec_off, label="OFF")
        self.ax[0].legend()
        self.ax[1].set_xlabel("Wavelength (nm)")
        self.ax[0].set_ylabel("ΔOD")
        self.ax[1].set_ylabel("ΔOD")
        self.ax[1].set_ylim(ylim)
        self.ax[1].set_title("TA spectrum")
        self.ax[0].set_title("On and Off spectrum")
        plt.show()


class load_spectra:
    """class to include single or multiple experiments (average) TA matrix

    Args:
        file_inp (str): The name of the file to be loaded. e.g. "expt_". if num_spec = 1, file_inp should use full name, e.g. "expt_3".
        num_spec (int, optional): num_spec is the number of experiments to be loaded. e.g. 5. Note this will load expt_1, expt_2, expt_3, expt_4, expt_5. Defaults to None.
        select (list, optional):  select is a list of the selected experiments to be loaded. e.g. [0,2,3,5]. Defaults to None.

    Notes:
        select [0,2,3,5] will load expt_1, expt_3, expt_4, expt_6. select CANNOT be a one element list.
        Use num_spec = 1 instead for single experiment.
    """

    def __init__(
        self, file_name: str, num_spec: int, select: list | None = None
    ) -> None:
        self.file_name = file_name
        self.file_inp = Path(self.file_name)
        self.file_inp_stem = self.file_inp.stem
        if select is None and (num_spec is None or num_spec == 1):
            self.num_spec = 1
            self.tamatrix_avg = np.loadtxt(self.file_inp)
            self.tatime = load_tatime(self.tamatrix_avg)
            self.tawavelength = load_tawavelength(self.tamatrix_avg)
        elif select is not None:
            self.select = select
            self.num_spec = len(self.select)
            self.tamatrix_avg, self.mat_array = mat_avg(self.file_inp, self.select)
            # load tatime and tawavelength axes
            self.tatime = load_tatime(self.tamatrix_avg)
            self.tawavelength = load_tawavelength(self.tamatrix_avg)
        elif num_spec >= 1:
            self.num_spec = num_spec
            # average the matrix
            self.select = list(range(self.num_spec))
            self.tamatrix_avg, self.mat_array = mat_avg(self.file_inp, self.select)
            # load tatime and tawavelength axes
            self.tatime = load_tatime(self.tamatrix_avg)
            self.tawavelength = load_tawavelength(self.tamatrix_avg)
        else:
            print("Invalid input. Please check the input parameters")
            return

    def mat_sub(self, obj_bg: "load_spectra", modifier: float | None = None) -> None:
        """Subtract background from the TA matrix

        Args:
            obj_bg (load_spectra): load_spectra object of the blank background TA matrix
            modifier (float, optional): modifier applied (multiplied) to the blank for subtraction. Defaults to None.
        """
        if modifier is None:
            modifier = 1
        self.tamatrix_avg = self.tamatrix_avg - obj_bg.tamatrix_avg * modifier
        self.mat_array = (
            self.mat_array - obj_bg.tamatrix_avg[:, :, np.newaxis] * modifier
        )

    def get_1ps(self) -> np.ndarray:
        """Get the 1ps spectrum and plot it

        Returns:
            1darray: 1ps spectrum
        """
        diff = np.abs(self.tatime - 1)
        pt = pt = np.argmin(diff)
        self.spec_1ps = self.tamatrix_avg[:, pt + 2]
        self.fig_s, self.ax_s = plt.subplots()
        self.ax_s.plot(self.tawavelength, self.spec_1ps)
        try:
            self.ax_s.set_title(self.file_inp.stem)
        except Exception as e:
            print(f"Error in loading file using Pathlib: {e}")
            self.ax_s.set_title(self.file_name)
        self.ax_s.set_xlabel("Wavelength (nm)")
        self.ax_s.set_ylabel("ΔOD")
        return self.spec_1ps

    def get_traces(self, wavelength: float, disable_plot: bool = False) -> np.ndarray:
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
            self.ax_k.plot(
                np.log(self.tatime), self.trace_avg, label=f"{wavelength} nm trace"
            )
        else:
            for i, x in enumerate(self.select):
                self.trace_array[:, i] = self.mat_array[pt, 2:, i]
                self.ax_k.plot(
                    np.log(self.tatime),
                    self.trace_array[:, i],
                    label=f"{wavelength} nm trace {x + 1}",
                )

            self.trace_avg = self.tamatrix_avg[pt, 2:]
            self.ax_k.plot(
                np.log(self.tatime),
                self.trace_avg,
                label=f"{wavelength} nm trace averaged",
            )
        self.ax_k.legend()
        self.ax_k.set_xlabel("Time (Log scale ps)")
        self.ax_k.set_ylabel("ΔOD")
        try:
            self.ax_k.set_title(self.file_inp.stem)
        except Exception as e:
            print(f"Error in loading file using Pathlib: {e}")
            self.ax_k.set_title(self.file_name)

        return self.trace_avg

    def fit_kinetic(
        self,
        wavelength: float,
        num_of_exp: int = 1,
        params: lmfit.Parameters | None = None,
        time_split: float = 5.0,
        w1_vary: bool = True,
        w12_vary: bool = True,
    ) -> lmfit.model.ModelResult:
        """Fit the kinetics at a specific wavelength

        Args:
            wavelength (num): The wavelength to be fitted
            num_of_exp (int, optional): Number of exponential to be fitted. Defaults to None.
            params (lmfit.Parameters, optional): The initial parameters for the fitting. Defaults to None.
            time_split (num, optional): The time point to split the plot. Defaults to None.
            w1_vary (bool, optional): Vary the w1 parameter. Defaults to None.
            w12_vary (bool, optional): Vary the w12 parameter. Defaults to None.

        Returns:
            lmfit.model.ModelResult: The result of the fitting
        """
        if params is None:
            params = params_init(num_of_exp, w1_vary=w1_vary, w12_vary=w12_vary)
        # plot spectra together
        diff = np.abs(self.tawavelength - wavelength)
        wavelength_index = np.argmin(np.abs(diff))
        y = self.trace_avg
        t = self.tatime
        lmodel = lmfit.Model(multiexp_func)
        result = lmodel.fit(
            y,
            params=params,
            t=t,
            max_nfev=100000,
            ftol=1e-9,
            xtol=1e-9,
            nan_policy="omit",
        )
        # print(result.fit_report())
        print("-------------------------------")
        print(
            f"{self.file_inp.stem} kinetics fit at {self.tawavelength[wavelength_index]:.2f} nm"
        )
        print("-------------------------------")
        print(f"chi-square: {result.chisqr:11.6f}")
        pearsonr = np.corrcoef(result.best_fit, y)[0, 1]
        print(f"Pearson's R: {pearsonr:11.6f}")
        print("-------------------------------")
        print("Parameter    Value       Stderr")
        for name, param in result.params.items():
            print(f"{name:7s} {param.value:11.6f} {param.stderr:11.6f}")
        print("-------------------------------")
        # result.plot_fit()

        pt_split = find_closest_value([time_split], self.tatime)[0]
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [2, 3]}
        )
        fig.subplots_adjust(wspace=0.05)

        ax1.scatter(t[:pt_split], y[:pt_split], marker="o", color="black")
        ax1.plot(t[:pt_split], result.best_fit[:pt_split], color="red")
        ax1.set_xlim(t[0], t[pt_split - 1])
        # ax1.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax1.spines["right"].set_visible(False)
        ax1.tick_params(right=False)

        ax2.scatter(
            t[pt_split:],
            y[pt_split:],
            marker="o",
            color="black",
            label=f"{self.tawavelength[wavelength_index]:.2f} nm",
        )
        ax2.plot(
            t[pt_split:],
            result.best_fit[pt_split:],
            color="red",
            label=f"{self.tawavelength[wavelength_index]:.2f} nm fit",
        )
        ax2.set_xscale("log")
        ax2.set_xlim(t[pt_split - 1], t[-1])
        # ax2.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(left=False)

        # Creating a gap between the subplots to indicate the broken axis
        gap = 0.1
        ax1.spines["right"].set_position(("outward", gap))
        ax2.spines["left"].set_position(("outward", gap))
        ax1.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
        # Centered title above subplots
        fig.suptitle(self.file_inp.stem, fontsize=10, ha="center")
        plt.legend(loc="best")
        fig.text(0.5, 0.04, "Time (ps)", ha="center", fontsize=8)
        ax1.set_ylabel("ΔOD")
        plt.show()
        return result

    def correct_burn(self, wavelength: float, disable_plot: bool = False) -> None:
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
            percent_per_point = (
                (
                    self.mat_array[pt, pt2 + 2, 0]
                    - self.mat_array[pt, pt2 + 2, len(self.select) - 1]
                )
                / self.mat_array[pt, pt2 + 2, 0]
                / (len(self.tatime) * (len(self.select) - 1))
            )
            burn_correction = 1 + percent_per_point * pts_time
            self.ax_b.plot(pts_time, burn_correction, label="Burn correction")
            self.ax_b.legend()
            self.ax_b.set_xlabel("time point")
            self.ax_b.set_ylabel("correction")
            self.tamatrix_avg_burncorr = self.tamatrix_avg.copy()
            self.tamatrix_avg_burncorr[:, 2:] *= burn_correction
            np.savetxt(
                self.file_name + "avg_burncorrected",
                self.tamatrix_avg_burncorr,
                fmt="%f",
                delimiter="\t",
            )


class compare_traces:
    """compare traces from load_spectra object

    Args:
        obj (load_spectra): first load_spectra object
        wavelength (num): wavelength to be compared
    """

    def __init__(self, obj: "load_spectra", wavelength: float) -> None:
        self.wavelength = wavelength
        self.tatime = obj.tatime
        trace = obj.get_traces(wavelength, disable_plot=True).reshape(1, -1)
        self.trace_array = np.empty((0, len(self.tatime)))
        print(self.trace_array.size)
        print(trace.size)
        self.trace_array = np.append(self.trace_array, trace, axis=0)
        self.wavelength_list = [self.wavelength]
        self.name_list = [obj.file_inp]

    def add_trace(self, obj: "load_spectra", wavelength: float | None = None) -> None:
        """add traces from another load_spectra object

        Args:
            obj (load_spectra): load_spectra object
            wavelength (num, optional): wavelength to be added if want to compare traces at diff wavelength. Defaults to None will use the wavelength from first object.
        """
        self.name_list.append(obj.file_inp)
        if wavelength is None:
            trace_toadd = obj.get_traces(self.wavelength, disable_plot=True).reshape(
                1, -1
            )
            self.wavelength_list.append(self.wavelength)
        else:
            try:
                trace_toadd = obj.get_traces(wavelength, disable_plot=True).reshape(
                    1, -1
                )
                self.wavelength_list.append(wavelength)
            except AttributeError:
                print("Invalid wavelength")
                return
        self.trace_array = np.append(self.trace_array, trace_toadd, axis=0)

    def plot(self) -> None:
        """plot the loaded traces"""
        self.fig, self.ax = plt.subplots()
        for i in range(len(self.trace_array)):
            self.ax.plot(
                np.log(self.tatime),
                self.trace_array[i, :] / np.max(np.abs(self.trace_array[i, :])),
                label=f"{self.name_list[i]} @ {self.wavelength_list[i]} nm",
            )
        self.ax.legend()
        self.ax.set_title("Normalized traces with logarithmic time axis")
        self.ax.set_xlabel("Time (Log scale ps)")
        self.ax.set_ylabel("ΔOD")


class glotaran:
    """Class to export the IGOR generated TAmatrix to Glotaran input format. Initialize the class with the TA matrix (Output from IGOR macro auto_tcorr. Without time and wavelength axis.
    NOT original TAMatrix like file), time axis and wavelength axis. Use SaveMatrix()macro in IGOR to get those inputs.
    Output file will be named as matrix_corr+"glo.ascii"

    Args:
        matrix_corr (str): The filename of the TA matrix file to be loaded.
        tatime (str): The filename of the time axis
        tawavelength (str): The filename of the wavelength axis
    """

    def __init__(self, matrix_corr: str | Path, tatime: str, tawavelength: str) -> None:
        self.filename = Path(matrix_corr)
        self.filestem = self.filename.stem
        self.tatime = np.loadtxt(tatime)
        self.tawavelength = np.loadtxt(tawavelength)
        # np.genfromtext will read nan as nan, avoid size mismatch with np.loadtxt
        self.output_matrix = np.genfromtxt(
            matrix_corr, delimiter="\t", filling_values=np.nan
        )
        self.output_matrix = np.append(
            self.tatime.reshape(1, -1), self.output_matrix, axis=0
        )
        self.output_matrix = np.append(
            np.append("", self.tawavelength).reshape(1, -1).T,
            self.output_matrix,
            axis=1,
        )
        self.header = (
            self.filestem + "\n\nTime explicit\nintervalnr " + str(len(self.tatime))
        )
        np.savetxt(
            self.filename.with_suffix(".ascii"),
            self.output_matrix,
            header=self.header,
            fmt="%s",
            comments="",
            delimiter="\t",
        )


class merge_glotaran:
    """Class to merge the Glotaran input files from visible and IR region
    The output will be saved as filename+"_ir_merged.ascii"
    Maybe write this as a function instead
    Args:
        glotaran_vis (glotaran): The load_glotaran object of the visible region
        glotaran_ir (glotaran): The load_glotaran object of the IR region
        vis_max (num): The maximum wavelength of the visible region
        ir_min (num): The minimum wavelength of the IR region
    """

    def __init__(
        self,
        glotaran_vis: "glotaran",
        glotaran_ir: "glotaran",
        vis_max: float,
        ir_min: float,
    ) -> None:
        self.glotaran_vis = glotaran_vis
        self.glotaran_ir = glotaran_ir
        if np.array_equal(self.glotaran_vis.tatime, self.glotaran_ir.tatime):
            self.tatime = self.glotaran_vis.tatime
        else:
            print("Time axis mismatch")
        self.vis_max_pt = np.argmin(np.abs(self.glotaran_vis.tawavelength - vis_max))
        self.ir_min_pt = np.argmin(np.abs(self.glotaran_ir.tawavelength - ir_min))
        self.output_matrix = np.vstack(
            (
                self.glotaran_vis.output_matrix[0 : self.vis_max_pt + 1, :],
                self.glotaran_ir.output_matrix[self.ir_min_pt :, :],
            )
        )
        self.header = self.glotaran_vis.header
        try:
            # May need further work to save the file correctly
            np.savetxt(
                self.glotaran_vis.filename.with_name(
                    self.glotaran_vis.filestem + "_ir_merged"
                ).with_suffix(".ascii"),
                self.output_matrix,
                header=self.header,
                fmt="%s",
                comments="",
                delimiter="\t",
            )
        except Exception as e:
            print(f"Error in merging using Pathlib: {e}")
            np.savetxt(
                self.glotaran_vis.filename.with_name(
                    self.glotaran_vis.filestem + "_ir_merged"
                ).with_suffix(".ascii"),
                self.output_matrix,
                header=self.header,
                fmt="%s",
                comments="",
                delimiter="\t",
            )
            print("Load with filename")


class load_glotaran:
    """Class to load the Glotaran input file. Output will be the time axis, wavelength axis and the TA matrix without time and wavelength axis

    Args:
        dir (str): The filename of the Glotaran input file to be loaded.
    """

    def __init__(self, dir):
        self.filename = Path(dir)
        try:
            self.filestem = self.filename.stem
        except Exception as e:
            print(f"Error in loading Glotaran file using Pathlib: {e}")
            self.filestem = dir.split(".")[-2]
            print("Load with filename")
        matrix = np.loadtxt(dir, skiprows=4, delimiter="\t", dtype=str)
        matrix[matrix == ""] = np.nan
        matrix = matrix.astype(np.float64)
        self.tatime = matrix[0, 1:]
        self.tawavelength = matrix[1:, 0]
        self.tamatrix = matrix[1:, 1:]


def batch_load_glotaran(
    dir=".",
    time_pts=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    xlim=None,
    ylim=None,
    figsize=(8,4),
):
    """Batch load all the Glotaran input files in the directory and process them.
    This function scans the specified directory for .ascii files, loads each file using
    the load_glotaran function, processes them with tamatrix_importer, and automatically
    extracts time-resolved spectra at specific time points.
    Args:
        dir (str, optional): The directory path where Glotaran input files (.ascii) are stored.
                            Defaults to the current directory (".").
        time_pts (list, optional): List of time points (in ps) for extracting spectra.
                            Defaults to [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].
        xlim (tuple, optional): x-axis limits for the plot. Defaults to None.
        ylim (tuple, optional): y-axis limits for the plot. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (8, 4).
        
    Returns:
        tuple: A tuple containing three elements:
            - ascii_files_list (list): List of Path objects for all found .ascii files.
            - glotaran_instance_list (list): List of processed Glotaran objects.
            - glotaran_instance_dict (dict): Dictionary mapping filenames (without extension)
              to their corresponding Glotaran objects.
    Raises:
        Exception: If there is an error loading the directory with Pathlib.
    Notes:
        - The function automatically applies auto_taspectra to each loaded file with
          predefined time points [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].
        - If the directory doesn't exist, the function prints "Invalid directory" and returns None.
    """
    try:
        current_dir = Path(dir)
    except Exception as e:
        print(f"Error in loading directory using Pathlib: {e}")
        return
    ascii_files_list = list(current_dir.glob("*.ascii"))
    print(ascii_files_list)
    glotaran_instance_list: list["tamatrix_importer"] = []
    glotaran_instance_dict = {}
    for i, ascii_file in enumerate(ascii_files_list):
        print(i, ascii_file)
        glotaran_instance_list.append(
            tamatrix_importer(load_glotaran=load_glotaran(ascii_file))
        )
        glotaran_instance_list[i].auto_taspectra(
            mat="tcorr",
            time_pts=time_pts,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
        )
        glotaran_instance_dict[ascii_files_list[i].stem] = glotaran_instance_list[i]
    return ascii_files_list, glotaran_instance_list, glotaran_instance_dict


class glotaran_output:
    """Class to plot Glotaran output files, including traces and DAS (Decay Associated Spectra).

    This class provides methods to visualize and analyze Glotaran output files:
    - Traces: "filename_traces.ascii"
    - DAS: "filename_DAS.ascii"
    - Summary: "filename_summary.txt"

    Usage:
        - Save the DAS as "filename_DAS.ascii", traces as "filename_traces.ascii", and summary as "filename_summary.txt".
        - For the plot_trace_fit method, ensure the Glotaran input file "filename.ascii" is present.

    Args:
        dir (str): Path to the file (without extension) for the Glotaran output set.
        low_threshold (float, optional): Lower threshold (in ps) to filter out ultrafast DAS. Default is 0.07.
    Raises:
        Exception: If there is an error loading the directory or files.
    """

    def __init__(self, dir: str):
        self.rate_list = []
        self.error_list = []
        self.filename = dir
        self.rate_list = []
        self.error_list = []
        try:
            with open(dir + "_summary.txt", "r") as file:
                find_rate = False
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith(
                        "Estimated Kinetic parameters: Dataset1:"
                    ):
                        # Split the line by spaces or commas and convert to float
                        self.rate_list = [
                            value for value in stripped_line.replace(",", " ").split()
                        ]
                        find_rate = True
                    if find_rate is True and stripped_line.startswith(
                        "Standard errors:"
                    ):
                        self.error_list = [
                            value for value in stripped_line.replace(",", " ").split()
                        ]
                        find_rate = False
                    if stripped_line.startswith("Estimated Irf parameters: Dataset1:"):
                        self.irf_paramters = [
                            value for value in stripped_line.replace(",", " ").split()
                        ]

            # Convert the list of rate and error to a NumPy array
            self.irf_parameters_array = np.array(self.irf_paramters[4:]).astype(float)
            self.irf_offset = self.irf_parameters_array[0]
            self.irf_width = self.irf_parameters_array[1]
            self.rate_array = np.array(self.rate_list[4:]).astype(float)
            self.error_array = np.array(self.error_list[2:]).astype(float)
        except Exception as e:
            print(f"Error in loading file: {e}")

    def plot_das(
        self,
        low_threshold: float = 0.07,
        save: bool = False,
        figsize: tuple[int, int] = (6, 3),
        time_split: int = 1,
    ) -> None:
        # Load the DAS and traces data
        self.das = np.loadtxt(self.filename + "_DAS.ascii", skiprows=1)
        self.fig_das, self.ax_das = plt.subplots(figsize=figsize)
        self.fig_das.subplots_adjust(left=0.2)
        self.fig_das.suptitle(self.filename.replace("_", " "), fontsize=10, ha="center")
        if self.das.shape[1] != 2 * self.rate_array.shape[0]:
            print("das and rate array size mismatch")
        for i in range(int(self.das.shape[1] / 2)):
            if 1 / self.rate_array[i] < low_threshold:
                continue
            else:
                self.ax_das.plot(
                    self.das[:, 2 * i],
                    self.das[:, 2 * i + 1],
                    label=(
                        "Long-term"
                        if 1 / self.rate_array[i] > 10000.0
                        else f"{1 / self.rate_array[i]:.2f} ps"
                    ),
                )
                colorwaves(self.ax_das)
                self.ax_das.legend()
                self.ax_das.set_xlabel("Wavelength (nm)")
                self.ax_das.set_ylabel("DAS")
                # print(self.das[:,i], self.das[:,i+1])
        self.ax_das.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        if save:
            self.fig_das.savefig(self.filename + "_DAS.png", dpi=300)

        # Load and plot the das trace data
        try:
            self.traces = np.loadtxt(self.filename + "_traces.ascii", skiprows=1)
            self.fig_traces, (self.ax_traces, self.ax_traces_2) = new_split_axes(
                figsize=figsize
            )
            self.fig_traces.suptitle(
                self.filename.replace("_", " "), fontsize=10, ha="center"
            )
            for i in range(int(self.traces.shape[1] / 2)):
                # Fix bug in Glotaran trace saving.
                # Only negative irf zerotime offset is saved in the traces file.
                # Manually add positive zerotime irf offset here
                if self.irf_offset > 0:
                    self.traces[:, 2 * i] += self.irf_offset

                if 1 / self.rate_array[i] < low_threshold:
                    continue
                else:
                    self.ax_traces.plot(
                        self.traces[:, 2 * i],
                        self.traces[:, 2 * i + 1],
                        label=(
                            "Long-term"
                            if 1 / self.rate_array[i] > 10000.0
                            else f"{1 / self.rate_array[i]:.2f} ps"
                        ),
                    )
                    self.ax_traces_2.plot(
                        self.traces[:, 2 * i],
                        self.traces[:, 2 * i + 1],
                        label=(
                            "Long-term"
                            if 1 / self.rate_array[i] > 10000.0
                            else f"{1 / self.rate_array[i]:.2f} ps"
                        ),
                    )
                    self.ax_traces.set_xlim(-0.5, time_split)
                    self.ax_traces_2.set_xlim(time_split, self.traces[-1, 2 * i])
                    self.ax_traces.spines["right"].set_visible(False)
                    self.ax_traces_2.spines["left"].set_visible(False)
                    self.ax_traces.yaxis.tick_left()
                    self.ax_traces.tick_params(labelright=False)
                    self.ax_traces_2.tick_params(axis="y", labelleft=False)
                    self.ax_traces_2.yaxis.tick_right()
                    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
                    kwargs = dict(
                        marker=[(-1, -d), (1, d)],
                        markersize=12,
                        linestyle="none",
                        color="k",
                        mec="k",
                        mew=1,
                        clip_on=False,
                    )
                    self.ax_traces.plot(
                        [1, 1],
                        [1, 0],
                        transform=self.ax_traces.transAxes,
                        **kwargs,  # type: ignore
                    )
                    self.ax_traces_2.plot(
                        [0, 0],
                        [0, 1],
                        transform=self.ax_traces_2.transAxes,
                        **kwargs,  # type: ignore
                    )
                    colorwaves(self.ax_traces)
                    colorwaves(self.ax_traces_2)
                    self.ax_traces_2.legend(loc="center right")
                    self.ax_traces_2.set_xscale("log")
                    self.ax_traces_2.set_xlabel("Time (ps)")
                    self.ax_traces_2.xaxis.set_label_coords(0.2, -0.1)
                    self.ax_traces.set_ylabel("Amplitude")
                    if save:
                        self.fig_traces.savefig(
                            self.filename + "_DAStraces.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
        except Exception as e:
            print(f"No trace data found or error in loading trace data: {e}")

    def plot_trace_fit(
        self,
        wavelength_select: list[float],
        tmax: int = 1000,
        figsize: tuple[int, int] = (8, 4),
        save: bool = False,
        time_split: int = 1,
    ) -> None:
        """Plot the traces with the fitted curve

        Args:
            wavelength_select (list[float]): The wavelength to be fitted..
            tmax (int, optional): The maximum time for the plot. Defaults to 1000.
            figsize (tuple[int, int], optional): The size of the figure. Defaults to (8, 3).
        """

        def get_base_path(filepath: str | Path) -> Path:
            """
            Extract the base path without the variable suffix (like '_5exp')

            Args:
                filepath (str or Path): The input filepath like 'dir/xxx_yyy_5exp'

            Returns:
                Path: The base path without the variable suffix, like 'dir/xxx_yyy'
            """
            path = Path(filepath)
            # Match pattern that ends with _Nexp where N is any number
            base_stem = re.sub(r"_\d+exp$", "", path.stem)
            return path.parent / base_stem

        self.wavelength_select = wavelength_select
        self.glotaran_matrix_dir = get_base_path(self.filename)
        try:
            self.glotaran_matrix = tamatrix_importer(
                load_glotaran=load_glotaran(
                    self.glotaran_matrix_dir.with_suffix(".ascii")
                )
            )
        except FileNotFoundError:
            print(f"Glotaran matrix file not found: {self.glotaran_matrix_dir}")
            return
        kinetics_set = self.glotaran_matrix.auto_takinetics(
            self.wavelength_select, tmax=tmax, plot=False
        )
        pts_select_fit = find_closest_value(wavelength_select, self.das[:, 0])
        self.kinect_fit_set = np.array([])
        self.cdf = norm.cdf(self.traces[:, 0], loc=0, scale=self.irf_width)
        self.fig_kin_fit, (self.ax_kin_fit1, self.ax_kin_fit2) = new_split_axes(
            figsize=figsize
        )
        for i in range(len(kinetics_set)):
            kinetic_fit = np.zeros_like(self.traces[:, 0])
            for j in range(int(self.das.shape[1] / 2)):
                kinetic_fit += (
                    self.das[pts_select_fit[i], 2 * j + 1] * self.traces[:, 2 * j + 1]
                )
            self.kinect_fit_set = np.append(self.kinect_fit_set, kinetic_fit)
            self.fig_kin_fit, (self.ax_kin_fit1, self.ax_kin_fit2) = plot_split_axes(
                x=self.glotaran_matrix.tatime,
                y=kinetics_set[i],
                fit_x=self.traces[:, 0],
                fit_y=kinetic_fit,
                fig=self.fig_kin_fit,
                axs_tuple=(self.ax_kin_fit1, self.ax_kin_fit2),
                time_split=time_split,
                title=f"{self.glotaran_matrix_dir.stem.replace('_', ' ')} - Kinetic & Fits",
                xlabel="Time (ps)",
                ylabel="ΔOD",
                label=f"{self.wavelength_select[i]} nm kinetics",
                fit_label=f"{self.wavelength_select[i]} nm fit",
                color_sequence=i,
            )
        # self.fig_kin_fit.suptitle(
        #     f"{self.glotaran_matrix_dir.stem.replace('_', ' ')} - Kinetic & Fits",
        #     fontsize=10,
        #     ha="center",
        # )
        if save:
            self.fig_kin_fit.savefig(
                Path(self.filename + "_globalfit_trace").with_suffix(".png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.plot()


class tamatrix_importer:
    """Class to import the TA matrix from the file.
    The input may vary from filename to Load_spectra object or load_glotaran objectInitialize the class with the filename, start and end wavelength to be loaded.
    If no filename is given, the object can be loaded from Load_spectra object or load_glotaran object

    Args:
        filename (Pathlib.Path, optional): The filename of the TA matrix file to be loaded.
        startnm (num, optional): The start wavelength to be loaded. Defaults to 0.
        endnm (num, optional): The end wavelength to be loaded. Defaults to 1200 nm.
        load_spectra (load_spectra, optional): The load_spectra object to be loaded.
        load_glotaran (load_glotaran, optional): The load_glotaran object to be loaded.
        tamatrix (str, optional): The filename of the TA matrix file to be loaded. NOT WORKING.
        tatime (str, optional): The filename of the time axis file to be loaded. NOT WORKING.
        tawavelength (str, optional): The filename of the wavelength axis file to be loaded. NOT WORKING.
        name (str, optional): The name of the object. Defaults to filename.
    """

    def __init__(
        self,
        filename=None,
        startnm=0,
        endnm=1200,
        load_spectra=None,
        load_glotaran=None,
        tamatrix_dir=None,
        tatime_dir=None,
        tawavelength_dir=None,
        name=None,
    ):
        self.startnm = startnm
        self.endnm = endnm
        # load from file if no object is given
        if filename is not None:
            # Load firstcol wave and find startrow and endrow
            # filename = input("Enter the filename for firstcol wave: ")
            # filename is the full directory while filestem is the name without extension

            try:
                self.filename = Path(filename)
                self.filestem = self.filename.stem
            except Exception as e:
                print(f"Error in loading file using Pathlib: {e}")
                self.filename = filename
                self.filestem = filename.split(".")[-2]
                print("Load with filename")
            firstcol = np.loadtxt(self.filename)[:, 1]
            self.startrow = 0
            self.endrow = len(firstcol)
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
                self.filename,
                skiprows=int(self.startrow),
                max_rows=int(self.endrow) - int(self.startrow),
            )[:, 1]
            # np.savetxt(self.filename+"_tawavelength",tawavelength,fmt='%1.5f')

            # Trim TAtime wave
            self.tatime = np.loadtxt(self.filename)[:, 0]
            idx = np.loadtxt(self.filename).shape[1] - 2
            self.tatime = self.tatime[:idx]
            # np.savetxt(self.filename+"_tatime",tatime,fmt='%1.5f')

            # Load TAmatrix waves
            self.tamatrix = np.loadtxt(
                self.filename,
                skiprows=int(self.startrow),
                max_rows=int(self.endrow) - int(self.startrow),
                usecols=np.arange(2, idx + 2).tolist(),
            )
            # np.savetxt(self.filename+"_tamatrix",self.tamatrix,fmt='%1.5f'

        elif load_spectra is not None:
            self.tawavelength = load_spectra.tawavelength
            self.tatime = load_spectra.tatime
            self.tamatrix = load_spectra.tamatrix_avg[:, 2:]
            self.filename = load_spectra.file_inp
            try:
                self.filestem = load_spectra.file_inp.stem
            except Exception as e:
                print(f"Error in loading file using Pathlib: {e}")
                self.filestem = load_spectra.file_inp.split(".")[-2]
                print("Load with filename")

        elif load_glotaran is not None:
            self.tawavelength = load_glotaran.tawavelength
            self.tatime = load_glotaran.tatime
            self.tcorr = load_glotaran.tamatrix
            self.filename = load_glotaran.filename
            try:
                self.filestem = Path(load_glotaran.filename).stem
            except Exception as e:
                print(f"Error in loading file using Pathlib: {e}")
                self.filestem = load_glotaran.filename.split(".")[-2]
                print("Load with filename")
        elif tamatrix_dir is not None and tatime_dir is not None and tawavelength_dir is not None:
            self.tawavelength = np.loadtxt(tawavelength_dir)
            self.tatime = np.loadtxt(tatime_dir)
            self.tamatrix = np.loadtxt(tamatrix_dir)
            self.filename = tamatrix_dir.split("/")[-1]
            try:
                self.filestem = Path(tamatrix_dir).stem
            except Exception as e:
                print(f"Error in loading file using Pathlib: {e}")
                self.filestem = tamatrix_dir.split(".")[-2]
                print("Load with filename")

        if name is not None:
            self.filestem = name

        self.fit_results = {}

    def contour(self, time_min, time_max):
        """Create a contour plot

        Args:
            time_min (num): lower limit of time axis
            time_max (num): upper limit of time axis
        """
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        plt.contourf(
            X,
            Y,
            self.tamatrix,
            [-0.005, -0.001, -0.0005, 0, 0.0005, 0.001, 0.005],
            cmap="rainbow",
        )
        plt.ylim(time_min, time_max)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Time (ps)")
        plt.colorbar()
        plt.show()

    def save_all(self, filename=None, mat=None):
        """Save the time axis, wavelength axis and TA matrix. Saved files will be named as filename+"_tatime", filename+"_tawavelength", filename+"_<matrix type>"

        Args:
            filename (str): directory to save the files. e.g. "C:/Users/xxx"
            mat (str): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        """
        # if mat is None:
        #     matrix = self.tcorr.copy()
        #     print("Background and Zero time corrected matrix is selected\n")
        # if mat == "original":
        #     matrix = self.tamatrix.copy()
        #     print("Original matrix is selected\n")
        # elif mat == "bgcorr":
        #     matrix = self.bgcorr.copy()
        #     print("Background corrected matrix is selected\n")
        # else:
        #     matrix = self.tcorr.copy()
        #     print("Background and Zero time corrected matrix is selected\n")
        matrix, mat = self.mat_selector(mat)
        if filename is None:
            filename = self.filestem
        try:
            np.savetxt(
                self.filename.with_name(filename + "_tawavelength"),
                self.tawavelength,
                fmt="%1.5f",
                delimiter="\t",
            )
            print(filename + "_tawavelength has been saved\n")
        except Exception as e:
            print(f"Error in saving tawavelength with Pathlib: {e}")
            np.savetxt(
                filename + "_tawavelength",
                self.tawavelength,
                fmt="%1.5f",
                delimiter="\t",
            )
            print(filename + "_tawavelength has been saved without Pathilb\n")
        try:
            np.savetxt(
                self.filename.with_name(filename + "_tatime"),
                self.tatime,
                fmt="%1.5f",
                delimiter="\t",
            )
            print(filename + "_tatime has been saved\n")
        except Exception as e:
            print(f"Error in saving tatime with Pathlib: {e}")
            np.savetxt(filename + "_tatime", self.tatime, fmt="%1.5f", delimiter="\t")
            print(filename + "_tatime has been saved without Pathlib\n")
        try:
            np.savetxt(
                self.filename.with_name(filename + "_" + mat),
                matrix,
                delimiter="\t",
                fmt="%1.5f",
            )
            print(filename + "_tamatrix has been saved\n")
        except Exception as e:
            print(f"Error in saving tamatrix with Pathlib: {e}")
            np.savetxt(filename + "_" + mat, matrix, fmt="%1.5f", delimiter="\t")
            print(filename + "_" + mat + "_tamatrix has been saved without Pathlib\n")

    def save_tamatrix(self, mat, filename=None):
        """Save the TA matrix. Saved file will be named as filename+"_tamatrix"

        Args:
            filename (str): directory to save the file. e.g. "C:/Users/xxx"
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to tcorr.
        """
        matrix, mat = self.mat_selector(mat)
        if filename is None:
            filename = self.filestem
        try:
            np.savetxt(
                self.filename.with_name(filename + "_" + mat), matrix, fmt="%1.5f"
            )
            print(filename + "_tamatrix has been saved\n")
        except Exception as e:
            print(f"Error in saving tamatrix with Pathlib: {e}")
            np.savetxt(filename + "_" + mat, matrix, fmt="%1.5f")
            print(filename + "_" + mat + "_tamatrix has been saved without Pathlib\n")

    def save_tatime(self, filename=None):
        """Save the time axis. Saved file will be named as filename+"_tatime"

        Args:
            filename (str): directory to save the file. e.g. "C:/Users/xxx"
        """
        if filename is None:
            filename = self.filestem
        try:
            np.savetxt(
                self.filename.with_name(filename + "_tatime"), self.tatime, fmt="%1.5f"
            )
            print(filename + "_tatime has been saved\n")
        except Exception as e:
            print(f"Error in saving tatime with Pathlib: {e}")
            np.savetxt(filename + "_tatime", self.tatime, fmt="%1.5f")
            print(filename + "_tatime has been saved without Pathlib\n")

    def save_tawavelength(self, filename=None):
        """Save the wavelength axis. Saved file will be named as filename+"_tawavelength"

        Args:
            filename (str): directory to save the file. e.g. "C:/Users/xxx"
        """
        if filename is None:
            filename = self.filestem
        try:
            np.savetxt(
                self.filename.with_name(filename + "_tawavelength"),
                self.tawavelength,
                fmt="%1.5f",
            )
            print(filename + "_tawavelength has been saved\n")
        except Exception as e:
            print(f"Error in saving tawavelength with Pathlib: {e}")
            np.savetxt(filename + "_tawavelength", self.tawavelength, fmt="%1.5f")
            print(filename + "_tawavelength has been saved without Pathilb\n")

    def save_axes(self, filename=None):
        """Save the time and wavelength axes. Saved files will be named as filename+"_tatime" and filename+"_tawavelength"

        Args:
            filename (str): directory to save the files. e.g. "C:/Users/xxx"
        """
        if filename is None:
            filename = self.filestem
        try:
            np.savetxt(
                self.filename.with_name(filename + "_tawavelength"),
                self.tawavelength,
                fmt="%1.5f",
            )
            print(filename + "_tawavelength has been saved\n")
        except Exception as e:
            print(f"Error in saving tawavelength with Pathlib: {e}")
            np.savetxt(filename + "_tawavelength", self.tawavelength, fmt="%1.5f")
            print(filename + "_tawavelength has been saved without Pathilb\n")
        try:
            np.savetxt(
                self.filename.with_name(filename + "_tatime"), self.tatime, fmt="%1.5f"
            )
            print(filename + "_tatime has been saved\n")
        except Exception as e:
            print(f"Error in saving tatime with Pathlib: {e}")
            np.savetxt(filename + "_tatime", self.tatime, fmt="%1.5f")
            print(filename + "_tatime has been saved without Pathlib\n")

    def auto_bgcorr(self, points: int) -> np.ndarray:
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

        print("The number of time points taken as background: " + str(points))
        npavg /= points
        # np.savetxt(self.filename+"_tamatrix_npavg", npavg, fmt='%1.5f')
        for x in range(self.tamatrix.shape[1]):
            self.bgcorr[:, x] = self.tamatrix[:, x] - npavg
        # np.savetxt(self.tamatrix+"_tamatrix_bgcorr", self.bgcorr, fmt='%1.5f')

        # Create a figure and axis for the contour plot
        # Create contour plot
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        plt.contourf(
            X,
            Y,
            self.bgcorr,
            [-0.005, -0.001, -0.0005, 0, 0.0005, 0.001, 0.005],
            cmap="rainbow",
        )
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

        Raises:
            Exception: If the bgcorr matrix is not found
        """

        # import correction line from drawing script
        self.zerotime_x = np.loadtxt(line_file)[:, 0]
        self.zerotime_y = np.loadtxt(line_file)[:, 1]
        # generate contempory file and output matrix
        time_temp = self.tatime.copy()
        try:
            oldmatrix = self.bgcorr.copy()
        except AttributeError:
            print("bgcorr matrix not found. zero time correction on original matrix")
            oldmatrix = self.tamatrix.copy()
        self.tcorr = np.zeros_like(self.tamatrix)

        for i in range(len(self.tawavelength)):
            # Tatime axis minus time offset from the drawn line
            time_temp = self.tatime + np.interp(
                self.tawavelength[i], self.zerotime_x, self.zerotime_y
            )
            # extrapolate TAsignal to match corrected time axis
            self.tcorr[i, :] = np.interp(time_temp, self.tatime, oldmatrix[i, :])
            # Add the following smoothing line to clean up the spectra, with a slight loss in time resolution.
            # newmatrix[i, :] = np.convolve(kin_temp2[i, :], np.ones(3)/3, mode='same')

        # Create contour plot with plot_contour()
        # Create contour plot
        fig, ax = plt.subplots()
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        ax.contour(X, Y, self.tcorr, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01])
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
        except Exception:
            print("bgcorr matrix not found. zero time correction on original matrix")
            oldmatrix = self.tamatrix.copy()
        self.tcorr = np.zeros_like(self.tamatrix)

        for i in range(len(self.tawavelength)):
            # Tatime axis minus time offset from the drawn line
            time_temp = self.tatime + np.interp(
                self.tawavelength[i], zerotime_x, zerotime_y
            )
            # extrapolate TAsignal to match corrected time axis
            self.tcorr[i, :] = np.interp(time_temp, self.tatime, oldmatrix[i, :])
            # Add the following smoothing line to clean up the spectra, with a slight loss in time resolution.
            # newmatrix[i, :] = np.convolve(kin_temp2[i, :], np.ones(3)/3, mode='same')
        # save tcorred matrix
        # np.savetxt(newmatrixname, newmatrix, fmt='%1.5f')

        # Create contour plot with plot_contour()
        # Create contour plot
        fig, ax = plt.subplots()
        Y, X = np.meshgrid(self.tatime, self.tawavelength)
        ax.contour(X, Y, self.tcorr, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01])
        plt.ylim(-1, 1)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Time (ps)")
        plt.draw()

        return self.tcorr

    def glotaran(self, mat=None, name=None):
        """Export the background and zerotime corrected TA matrix to Glotaran input format (legacy JAVA version). Saved file will be named as filename+"glo.ascii"
        Don't need this array in pyglotaran.
        Args:
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        """
        output_matrix, mat = self.mat_selector(mat)
        output_matrix = np.append(self.tatime.reshape(1, -1), output_matrix, axis=0)
        output_matrix = np.append(
            np.append("", self.tawavelength).reshape(1, -1).T, output_matrix, axis=1
        )
        header = (
            self.filestem + "\n\nTime explicit\nintervalnr " + str(len(self.tatime))
        )
        if name is None:
            name = f"{str(self.filename)}_{mat}.ascii"
        np.savetxt(
            name,
            output_matrix,
            header=header,
            fmt="%s",
            comments="",
            delimiter="\t",
        )
        print(f"{name} has been saved\n")

    def pyglotaran(self, mat=None):
        """
        export tcorr matrix to pyglotaran xarray dataset
        e.g.
        dataset = tamatrix.pyglotaran()

        Args:
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        return:
            xarray dataset
        """
        time_vals = self.tatime
        spectral_vals = self.tawavelength
        matrix, mat = self.mat_selector(mat)
        data_vals = matrix.T

        # Define dimensions and coordinates
        dims = ("time", "spectral")
        coords = {"time": time_vals, "spectral": spectral_vals}

        # Create xarray dataset
        dataset = xr.Dataset(
            {"data": (dims, data_vals)},
            coords=coords,
            attrs={"source_path": "dataset_1.nc"},
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
                mat = "tcorr"
            except Exception:
                try:
                    matrix = self.bgcorr.copy()
                    mat = "bgcorr"
                    print("Background corrected matrix used")
                except Exception:
                    matrix = self.tamatrix.copy()
                    mat = "original"
                    print("Original matrix used")
        elif mat == "original":
            matrix = self.tamatrix.copy()
        elif mat == "bgcorr":
            matrix = self.bgcorr.copy()
        elif mat == "tcorr":
            matrix = self.tcorr.copy()
        else:
            try:
                matrix = self.tcorr.copy()
                print("Invalid mat value. Use tcorrrected matrix")
            except Exception:
                try:
                    matrix = self.bgcorr.copy()
                    print("Invalid mat value. Background corrected matrix used")
                except Exception:
                    matrix = self.tamatrix.copy()
                    print("Invalid mat value. Original matrix used")
        return matrix, mat

    def auto_taspectra(
        self, time_pts=None, mat=None, xlim=None, ylim=None, figsize=(6, 3)
    ):
        """Plot the TA spectra at selected time points

        Args:
            time_pts (list, optional): The time points to be plotted. Defaults to [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500].
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
            xlim (tuple, optional): The x-axis limits. Defaults to None.
            ylim (tuple, optional): The y-axis limits. Defaults to None.
            figsize (tuple, optional): The size of the figure. Defaults to (6, 3).

        Returns:
            1darray, 1darray: The spectra set, the index of the time points
        """

        if time_pts is None:
            time_pts = [
                -0.5,
                -0.2,
                0,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                50,
                100,
                200,
                500,
                1000,
                1500,
            ]

        matrix, mat = self.mat_selector(mat)
        # find closest time points
        time_index = find_closest_value(time_pts, self.tatime)
        rainbow = colormaps["rainbow"]
        colors = rainbow(np.linspace(1, 0, len(time_index)))
        cmap = ListedColormap(colors)
        self.spectra_set = self.tawavelength.copy()
        fig, ax = plt.subplots(figsize=figsize)
        # plot spectra together
        for i in range(len(time_index)):
            spec = matrix[:, time_index[i]]
            self.spectra_set = np.c_[self.spectra_set, spec]
            ax.plot(
                self.tawavelength,
                spec,
                label="{:.2f}".format(self.tatime[time_index[i]]) + " ps",
                color=cmap(i),
                linewidth=0.5,
            )
        # plt.ylim(-0.05,0.05)
        ax.set_title(self.filestem.replace("_", " "))
        plt.rcParams.update(
            {
                "font.size": 8,  # Default font size
                "axes.labelsize": 8,  # Label size for x and y axes
                "axes.titlesize": 8,  # Title size
                "xtick.labelsize": 8,  # Tick label size for x axis
                "ytick.labelsize": 8,  # Tick label size for y axis
            }
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("ΔOD")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc="best", ncol=2)
        fig.show()
        return self.spectra_set, time_index

    def save_taspectra(self, name=None, time_pts=None, mat=None):
        """Save the TA spectra at selected time points. Saved file will be named as s_name

        Args:
            name (str, optional): The name of the file. Defaults to tamatrix_importer.filename.
            time_pts (list, optional): The time points to be plotted. Defaults to [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500].
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        """
        if name is None:
            name = self.filestem
        self.spectra_set, time_index = self.auto_taspectra(time_pts, mat)
        header_str = "Wavelength\t"
        for time in time_index:
            header_str = (
                header_str
                + "s_"
                + name
                + "_"
                + "{:.2f}".format(self.tatime[time])
                + "ps\t"
            )
        try:
            np.savetxt(
                self.filename.with_name("s_" + name),
                self.spectra_set,
                header=header_str,
                fmt="%1.5f",
                delimiter="\t",
            )
            print("File s_" + name + " has been saved\n")
        except Exception as e:
            print(f"Error in saving spectra with Pathlib: {e}")
            np.savetxt(
                "s_" + name,
                self.spectra_set,
                header=header_str,
                fmt="%1.5f",
                delimiter="\t",
            )
            print("File s_" + name + " has been saved without Pathlib\n")

    def get_spectrum(
        self,
        name,
        time_pt,
        mat=None,
    ):
        matrix, mat = self.mat_selector(mat)
        diff = self.tatime.copy() - time_pt
        index = np.argmin(np.abs(diff))
        try:
            np.savetxt(
                self.filename.with_name(
                    "s_" + name + "_" + "{:.2f}".format(self.tatime[index]) + " ps"
                ),
                matrix[:, index],
                fmt="%1.5f",
            )
        except Exception as e:
            print(f"Error in saving spectra with Pathlib: {e}")
            np.savetxt(
                "s_" + name + "_" + "{:.2f}".format(self.tatime[index]) + " ps",
                matrix[:, index],
                fmt="%1.5f",
            )
            print(
                "File s_"
                + name
                + "_"
                + "{:.2f}".format(self.tatime[index])
                + " ps has been saved without Pathlib\n"
            )
        plt.plot(
            self.tawavelength,
            matrix[:, index],
            label="{:.2f}".format(self.tatime[index]) + " ps",
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return matrix[:, index]

    def return_spectrum(self, time_pt, mat=None):
        tamatrix, mat = self.mat_selector(mat)
        diff = self.tatime.copy() - time_pt
        index = np.argmin(np.abs(diff))
        plt.plot(
            self.tawavelength,
            tamatrix[:, index],
            label="{:.2f}".format(self.tatime[index]) + "ps",
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return tamatrix[:, index]

    def auto_takinetics(self, wavelength_pts, mat=None, tmax=1000, plot=True):
        """Plot the TA kinetics at selected wavelengths

        Args:
            wavelength_pts (list): The wavelengths to be plotted.
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
            tmax (num, optional): The maximum time to be plotted. Defaults to 1000.

        Returns:
            2darray: The kinetics set
        """
        self.kinetics_set = []
        # sample time_pts = [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500]
        # find closest time points
        wavelength_index = find_closest_value(wavelength_pts, self.tawavelength)
        matrix, mat = self.mat_selector(mat)
        # plot spectra together
        # plot spectra together
        if plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            for i in range(len(wavelength_index)):
                spec = matrix[wavelength_index[i], :].T
                self.kinetics_set.append(spec)

                ax.plot(
                    self.tatime,
                    spec,
                    label="{:.2f}".format(self.tawavelength[wavelength_index[i]])
                    + " nm",
                    linewidth=1,
                )
            ax.set_title(self.filestem.replace("_", " "))
            plt.rcParams.update(
                {
                    "font.size": 8,  # Default font size
                    "axes.labelsize": 8,  # Label size for x and y axes
                    "axes.titlesize": 8,  # Title size
                    "xtick.labelsize": 8,  # Tick label size for x axis
                    "ytick.labelsize": 8,  # Tick label size for y axis
                }
            )
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("ΔOD")
            ax.set_xlim(-1, tmax)
            ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax.legend(loc="best")
            fig.show()
        else:
            for i in range(len(wavelength_index)):
                spec = matrix[wavelength_index[i], :].T
                self.kinetics_set.append(spec)
        return self.kinetics_set

    def save_takinetics(
        self, wavelength_pts, tmax=1000, name=None, mat=None, plot=True
    ):
        """Plot and Save the TA kinetics at selected wavelengths. Saved file will be named as k_name

        Args:
            name (str): The name of the file.
            wavelength_pts (list): The wavelengths to be plotted.
            mat (str): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'.

        Returns:
            2darray: The kinetics set
        """
        # sample time_pts = [-0.5,-0.2, 0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500]
        # find closest time points
        if name is None:
            name = self.filestem
        self.kinetics_set = self.auto_takinetics(
            wavelength_pts=wavelength_pts, mat=mat, tmax=tmax, plot=plot
        )
        header_str = "Time(ps)\t"
        try:
            np.savetxt(
                self.filename.with_name("k_" + name),
                self.kinetics_set,
                header=header_str,
                fmt="%1.5f",
                delimiter="\t",
            )
            print("File k_" + name + " has been saved\n")
        except Exception as e:
            print(f"Error in saving kinetics with Pathlib: {e}")
            np.savetxt(
                "k_" + name,
                self.kinetics_set,
                header=header_str,
                fmt="%1.5f",
                delimiter="\t",
            )
            print("File k_" + name + " has been saved without Pathlib\n")
        return self.kinetics_set

    def get_kinetic(self, name, wavelength_pt, mat=None):
        """Get the kinetics at a specified wavelength
        Args:
            name (str): The name of the file.
            wavelength_pt (float): The wavelength at which to get the kinetics.
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.
        Returns:
            1darray: The kinetics at the specified wavelength
        """
        matrix, mat = self.mat_selector(mat)
        # plot spectra together
        diff = np.abs(self.tawavelength - wavelength_pt)
        wavelength_index = np.argmin(np.abs(diff))
        plt.plot(
            self.tatime,
            matrix[wavelength_index, :],
            label="{:.2f}".format(self.tawavelength[wavelength_index]) + " nm",
        )
        try:
            np.savetxt(
                self.filename.with_name(
                    "k_"
                    + name
                    + "_"
                    + "{:.2f}".format(self.tawavelength[wavelength_index])
                    + "nm"
                ),
                matrix[wavelength_index, :].T,
                fmt="%1.5f",
                delimiter="\t",
            )
            print(
                "File k_"
                + name
                + "_"
                + "{:.2f}".format(self.tawavelength[wavelength_index])
                + "nm has been saved\n"
            )
        except Exception as e:
            print(f"Error in saving kinetics with Pathlib: {e}")
            np.savetxt(
                "k_"
                + name
                + "_"
                + "{:.2f}".format(self.tawavelength[wavelength_index])
                + "nm",
                matrix[wavelength_index, :].T,
                fmt="%1.5f",
                delimiter="\t",
            )
            print(
                "File k_"
                + name
                + "_"
                + "{:.2f}".format(self.tawavelength[wavelength_index])
                + "nm has been saved without Pathlib\n"
            )
        plt.xlabel("Time (ps)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return matrix[wavelength_index, :]

    def return_kinetic(self, wavelength_pt, mat=None):
        """Return the kinetics at a specified wavelength

        Args:
            wavelength_pt (float): The wavelength at which to return the kinetics.
            mat (str, optional): The matrix to be saved. Options are 'original', 'bgcorr', 'tcorr'. Defaults to 'tcorr'.

        Returns:
            1darray: The kinetics at the specified wavelength
        """
        # plot spectra together
        tamatrix, mat = self.mat_selector(mat)
        diff = np.abs(self.tawavelength - wavelength_pt)
        wavelength_index = np.argmin(np.abs(diff))
        plt.plot(
            self.tatime,
            tamatrix[wavelength_index, :],
            label="{:.2f}".format(self.tawavelength[wavelength_index]) + " nm",
        )
        plt.xlabel("Time (ps)")
        plt.ylabel("ΔOD")
        plt.legend()
        plt.show()
        return tamatrix[wavelength_index, :]

    def fit_kinetic(
        self,
        wavelength,
        num_of_exp=1,
        mat=None,
        params=None,
        time_split=None,
        fitstart=None,
        ignore=None,
        avg_pts=1,
    ):
        """
        Fits kinetic data at a specified wavelength using a multi-exponential model.

        Args:
            wavelength (float): The wavelength at which to fit the kinetic data.
            num_of_exp (int, optional): The number of exponentials to use in the fitting model.
            mat (array-like, optional): The matrix containing the data to be fitted.
                If None, the default matrix is used. Defaults to None.
            params (lmfit.Parameters, optional): Initial parameters for the fitting model.
                If None, default parameters are initialized. Defaults to None.
            time_split (float, optional): The time point at which to split the plot
                into linear and logarithmic scales. Defaults to None.
            fitstart (float, optional): The time (ps) at which to start the fitting.
                Data points before this time are given lower weights. Defaults to None.
            ignore (list of tuples, optional): List of time regions (ps, ps) to ignore during fitting.
                Each tuple contains the start and end times of the region. Defaults to None.
            avg_pts (int, optional): The number of points to average around the specified wavelength.
                Defaults to 1 (no averaging).

        Returns:
            lmfit.model.ModelResult: The result of the fitting process,
                containing the best-fit parameters and statistics.

        Notes:
            - The function plots the fitted data and the original data for visual inspection.
            - The fitting results are stored in the `fit_results` attribute of the object.
        """
        if params is None:
            params = params_init(num_of_exp)

        matrix, mat = self.mat_selector(mat)

        # plot spectra together
        # diff = np.abs(self.tawavelength - wavelength)
        # wavelength_index = np.argmin(np.abs(diff))
        wavelength_index = np.searchsorted(self.tawavelength, wavelength)

        if avg_pts == 1:
            y = matrix[wavelength_index, :]
        else:
            y = np.mean(
                matrix[
                    wavelength_index - int(avg_pts / 2) : wavelength_index
                    + int(avg_pts / 2),
                    :,
                ],
                axis=0,
            )
        t = self.tatime
        # Determine the start index for fitting

        if fitstart is None:
            weights = np.ones_like(t)
        else:
            weights = np.ones_like(t)
            fitstart_idx = np.searchsorted(t, fitstart)
            weights[:fitstart_idx] = 0.001

        if ignore is not None:
            for region in ignore:
                weights[
                    np.searchsorted(t, region[0]) : np.searchsorted(t, region[1])
                ] = 0.001

        lmodel = lmfit.Model(multiexp_func)
        result = lmodel.fit(
            y,
            params=params,
            t=t,
            max_nfev=100000,
            ftol=1e-9,
            xtol=1e-9,
            nan_policy="omit",
            weights=weights,
        )

        print("-------------------------------")
        print(
            f"{self.filestem} kinetics fit at {self.tawavelength[wavelength_index]:.2f} nm"
        )
        print("-------------------------------")
        print(f"chi-square: {result.chisqr:11.6f}")

        pearsonr = np.corrcoef(result.best_fit, y)[0, 1]
        print(f"Pearson's R: {pearsonr:11.6f}")
        print("-------------------------------")
        print("Parameter    Value       Stderr")

        for name, param in result.params.items():
            stderr_value = param.stderr if param.stderr is not None else float("nan")
            print(f"{name:7s} {param.value:11.6f} {stderr_value:11.6f}")
        print("-------------------------------")

        if time_split is None:
            pt_split = find_closest_value([5], self.tatime)[0]
        else:
            pt_split = find_closest_value([time_split], self.tatime)[0]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [2, 3]}
        )
        fig.subplots_adjust(wspace=0.05)

        ax1.scatter(t[:pt_split], y[:pt_split], marker="o", color="black")
        ax1.plot(t[:pt_split], result.best_fit[:pt_split], color="red")
        ax1.set_xlim(t[0], t[pt_split - 1])
        ax1.spines["right"].set_visible(False)
        ax1.tick_params(right=False)

        ax2.scatter(
            t[pt_split:],
            y[pt_split:],
            marker="o",
            color="black",
            label=f"{self.tawavelength[wavelength_index]:.2f} nm",
        )
        ax2.plot(
            t[pt_split:],
            result.best_fit[pt_split:],
            color="red",
            label=f"{self.tawavelength[wavelength_index]:.2f} nm fit",
        )
        ax2.set_xscale("log")
        ax2.set_xlim(t[pt_split - 1], t[-1])
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(left=False)

        # Creating a gap between the subplots to indicate the broken axis
        gap = 0.1
        ax1.spines["right"].set_position(("outward", gap))
        ax2.spines["left"].set_position(("outward", gap))
        ax1.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)

        # Centered title above subplots
        fig.suptitle(self.filestem, fontsize=10, ha="center")
        plt.legend(loc="best")
        fig.text(0.5, 0.04, "Time (ps)", ha="center", fontsize=8)
        ax1.set_ylabel("ΔOD")
        plt.show()

        self.fit_results[str(wavelength)] = [y, result.best_fit, result]
        return result

    def plot_fit(self, time_split=None):
        """Plot the fitted kinetics data.
        Args:
            time_split (float, optional): The time point at which to split the plot
                into linear and logarithmic scales. Defaults to None.

        """
        rainbow = colormaps["rainbow"]
        colors = rainbow(np.linspace(1, 0, len(self.fit_results)))
        cmap = ListedColormap(colors)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [2, 3]}
        )
        fig.subplots_adjust(wspace=0.05)
        if time_split is None:
            pt_split = find_closest_value([5], self.tatime)[0]
        else:
            pt_split = find_closest_value([time_split], self.tatime)[0]
        for i, (key, value) in enumerate(self.fit_results.items()):
            # ax1.scatter(self.tatime,value[0], facecolor='none',marker = 'o',s = 50, edgecolor =cmap(i))
            # ax1.plot(self.tatime,value[1], color =cmap(i), label = f"{key} nm")
            ax1.scatter(
                self.tatime[:pt_split],
                value[0][:pt_split],
                facecolor="none",
                marker="o",
                s=50,
                edgecolor=cmap(i),
            )
            ax1.plot(self.tatime[:pt_split], value[1][:pt_split], color=cmap(i))

            # ax2.scatter(self.tatime,value[0], facecolor='none',marker = 'o',s = 50, edgecolor =cmap(i))
            # ax2.plot(self.tatime,value[1], color =cmap(i), label = f"{key} nm")
            ax2.scatter(
                self.tatime[pt_split:],
                value[0][pt_split:],
                facecolor="none",
                marker="o",
                s=50,
                edgecolor=cmap(i),
            )
            ax2.plot(
                self.tatime[pt_split:],
                value[1][pt_split:],
                color=cmap(i),
                label=f"{key} nm",
            )
            ax2.set_xscale("log")

        ax2.set_xlim(self.tatime[pt_split - 1], self.tatime[-1])
        # ax2.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(left=False)

        ax1.set_xlim(self.tatime[0], self.tatime[pt_split - 1])
        # ax1.set_ylim(min(result.best_fit), max(result.best_fit)*1.1)
        ax1.spines["right"].set_visible(False)
        ax1.tick_params(right=False)
        # Creating a gap between the subplots to indicate the broken axis
        gap = 0.1
        ax1.spines["right"].set_position(("outward", gap))
        ax2.spines["left"].set_position(("outward", gap))
        ax1.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
        # Centered title above subplots
        fig.suptitle(self.filestem, fontsize=10, ha="center")
        plt.legend(loc="best")
        fig.text(0.5, 0.04, "Time (ps)", ha="center", fontsize=8)
        ax1.set_ylabel("ΔOD")
        plt.show()

    def fit_correlation(self, num_of_exp: int) -> None:
        """Fit the cross-correlation curve to determine the zero time.

        Args:
            num_of_exp (int): The number of exponentials to use in the fitting model.
        """
        self.t0_list = np.empty((3, 0))
        t = self.tatime
        lmodel = lmfit.Model(multiexp_func)
        params = params_init(num_of_exp)
        t1 = find_closest_value([1], self.tatime)
        tamax = np.max(np.abs(self.bgcorr[:, t1]))
        wlmax = np.argmax(np.abs(self.bgcorr[:, t1]))
        y = self.bgcorr[wlmax, :]
        result = lmodel.fit(
            y, params=params, t=t, method="powell", max_nfev=1000000, nan_policy="omit"
        )
        params.update(result.params)
        for i in tqdm(range(self.bgcorr.shape[0]), desc="Fitting"):
            if np.abs(self.bgcorr[i, t1]) > 0.1 * tamax and i % 20 == 0:
                y = self.bgcorr[i, :]
                result = lmodel.fit(
                    y,
                    params=params,
                    t=t,
                    method="powell",
                    max_nfev=1000000,
                    nan_policy="omit",
                )
                rms = result.chisqr
                if (
                    result.success and rms is not None and rms < 0.15
                ):  # Check if the fit was successful
                    self.t0_list = np.append(
                        self.t0_list,
                        np.array(
                            [
                                [self.tawavelength[i]],
                                [result.params["w10"].value],
                                [rms],
                            ]
                        ),
                        axis=1,
                    )
                    params.update(result.params)
        fit = polyfit(self.t0_list[1], self.t0_list[0], self.t0_list[2])
        self.t0_list[2] = fit
        fig, ax = plt.subplots()
        ax.plot(self.t0_list[0], self.t0_list[1], label="correction line")
        ax.plot(self.t0_list[0], self.t0_list[2], label="Fitted correction line")
        ax.legend()
        plt.show()


def find_closest_value(
    list1: list[int | float] | np.ndarray, list2: list[int | float] | np.ndarray
) -> list[int]:
    """Find the closest value in list2 for each element in list1.
    Similar to np.searchsorted but doesn't require sorted array.

    Args:
        list1 (list): The list of values to find the closest value for.
        list2 (list): The list of values to search for the closest value.

    Returns:
        list: The list of indices of the closest values in list2 for each element in list1.
    """
    array1 = np.array(list1)
    array2 = np.array(list2)
    closest = [0] * len(array1)  # Initialize closest list with zeros
    for i in range(len(array1)):
        difference = array2 - array1[i]
        # Use np.abs to get the absolute difference
        closest[i] = int(np.argmin(np.abs(difference)))
    # Remove same elements
    closest_2 = []
    for x in closest:
        if x not in closest_2:
            closest_2.append(x)
    return closest_2


# Plot contour from files
def plot_contour_file(
    tatime_file: str, tawavelength_file: str, tamatrix_file: str, max_point: int
) -> None:
    """Plot a contour plot of the TA matrix from files.

    Args:
        tatime_file (str): The file containing the time points of the TA data.
        tawavelength_file (str): The file containing the wavelength points of the TA data.
        tamatrix_file (str): The file containing the TA matrix.
        max_point (int): The maximum time (ps) to plot.
    """
    tatime = np.loadtxt(tatime_file)
    tawavelength = np.loadtxt(tawavelength_file)
    tamatrix = np.loadtxt(tamatrix_file)
    # Create contour plot
    Y, X = np.meshgrid(tatime, tawavelength)
    plt.contourf(
        X, Y, tamatrix, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01], cmap="rainbow"
    )
    plt.colorbar()
    plt.ylim(-1, max_point)
    plt.show()


# Plot contour with numpy arrays
def plot_contour(
    tatime: np.ndarray,
    tawavelength: np.ndarray,
    tamatrix: np.ndarray,
    max_point: int = 1500,
) -> None:
    """Plot a contour plot of the TA matrix.

    Args:
        tatime (array-like): The time points of the TA data.
        tawavelength (array-like): The wavelength points of the TA data.
        tamatrix (array-like): The TA matrix.
        max_point (int): The maximum time (ps) to plot.
    """
    # Create contour plot
    Y, X = np.meshgrid(tatime, tawavelength)
    plt.contourf(
        X, Y, tamatrix, [-0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01], cmap="rainbow"
    )
    plt.colorbar()
    plt.ylim(-1, max_point)
    plt.show()


def polynomial_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Polynomial function for fitting data.

    Args:
        x (array-like): The x-values of the data.
        a (float): The coefficient of the polynomial.
        b (float): The coefficient of the polynomial.
        c (float): The coefficient of the polynomial.

    Returns:
        array-like: The fitted curve
    """
    return a / (1e-9 + (x) ** 2) + c


def polyfit(y, x, weights):
    """Fits a polynomial function to the data.

    Args:
        y (array-like): The y-values of the data.
        x (array-like): The x-values of the data.
        weights (array-like): The weights of the data.

    Returns:
        array-like: The fitted curve.
    """
    # Creating a Model object with the quadruple function
    poly_model = lmfit.Model(polynomial_func)

    # Creating Parameters and adding them to the model
    params = lmfit.Parameters()
    params.add("a", value=1.0)
    params.add("b", value=1.0)
    params.add("c", value=1.0)

    # Fitting the model to the data
    result = poly_model.fit(y, params, method="powell", x=x, weights=1 / weights)
    fitted_curve = result.best_fit
    print(result.params)
    return fitted_curve


def multiexp_func(
    t: np.ndarray,
    w0: float,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    w5: float,
    w6: float,
    w7: float,
    w8: float,
    w9: float,
    w10: float,
    w11: float,
    w12: float,
) -> np.ndarray:
    """Multi-exponential function for fitting TA data.

    Args:
        t (array-like): Time points.
        w0 (float): Gaussian distribution width for IRF fitting. Use gaussian integral to fit the IRF.
        w1 (float): General amplitude of the fitting. Default to 1.
        w2 (float): Amplitude of the first exponential.
        w3 (float): Lifetime of the first exponential.
        w4 (float): Amplitude of the second exponential.
        w5 (float): Lifetime of the second exponential.
        w6 (float): Amplitude of the third exponential.
        w7 (float): Lifetime of the third exponential.
        w8 (float): Amplitude of the fourth exponential.
        w9 (float): Lifetime of the fourth exponential.
        w10 (float): Zero time.
        w11 (float): Pre-zero offset.
        w12 (float): Long-terme offset.

    Returns:
        array-like: The fitted data.
    """
    w = [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12]
    sigma = np.sqrt(w[0] ** 2) / (2 * np.sqrt(2 * np.log(2)))
    result = np.zeros_like(t)  # initialize result

    if w[3] == 0:
        exp1 = np.zeros_like(t)
    else:
        k0 = 1 / w[3]
        exp1 = w[2] * np.exp(-(t - w[10]) * (k0)) * norm.cdf(t - w[10], scale=sigma)

    if w[5] == 0:
        exp2 = np.zeros_like(t)
    else:
        k1 = 1 / w[5]
        exp2 = w[4] * np.exp(-(t - w[10]) * (k1)) * norm.cdf(t - w[10], scale=sigma)

    if w[7] == 0:
        exp3 = np.zeros_like(t)
    else:
        k2 = 1 / w[7]
        exp3 = w[6] * np.exp(-(t - w[10]) * (k2)) * norm.cdf(t - w[10], scale=sigma)

    if w[9] == 0:
        exp4 = np.zeros_like(t)
    else:
        k3 = 1 / w[9]
        exp4 = w[8] * np.exp(-(t - w[10]) * (k3)) * norm.cdf(t - w[10], scale=sigma)

    result += exp1 + exp2 + exp3 + exp4
    result += w[11] + w[12] * norm.cdf(t - w[10], scale=sigma)
    result *= 1
    # b=4*np.log1p(2)/(w[0]**2)
    return result


def params_init(
    num_of_exp: int,
    w0_value: float = 0.1,
    w0_min: float = 0.05,
    w0_max: float = 0.2,
    w0_vary: bool = True,
    w1_value: float = 1.0,
    w1_vary: bool = False,
    c1_value: float = 0,
    c1_min: float = -0.5,
    c1_max: float = 0.5,
    c1_vary: bool = True,
    t1_value: float = 1,
    t1_min: float = 0.01,
    t1_max: float = 5000,
    t1_vary: bool = True,
    c2_value: float = 0,
    c2_min: float = -0.5,
    c2_max: float = 0.5,
    c2_vary: bool = True,
    t2_value: float = 10,
    t2_min: float = 0.01,
    t2_max: float = 5000,
    t2_vary: bool = True,
    c3_value: float = 0,
    c3_min: float = -0.5,
    c3_max: float = 0.5,
    c3_vary: bool = True,
    t3_value: float = 50,
    t3_min: float = 0.01,
    t3_max: float = 5000,
    t3_vary: bool = True,
    c4_value: float = 0,
    c4_min: float = -0.5,
    c4_max: float = 0.5,
    c4_vary: bool = True,
    t4_value: float = 500,
    t4_min: float = 0.01,
    t4_max: float = 5000,
    t4_vary: bool = True,
    w10_value: float = 0.0,
    w10_min: float = -0.5,
    w10_max: float = 0.5,
    w10_vary: bool = True,
    w11_value: float = 0.0,
    w11_min: float = -0.1,
    w11_max: float = 0.1,
    w11_vary: bool = True,
    w12_value: float = 0.0,
    w12_min: float = -0.5,
    w12_max: float = 0.5,
    w12_vary: bool = False,
) -> lmfit.Parameters:
    """Initialize parameters for the TA Analyzer.

    Args:
        num_of_exp (int): Number of experiments.
        w0_value (float, optional): Initial value for w0. Defaults to 0.1.
        w0_min (float, optional): Minimum value for w0. Defaults to 0.05.
        w0_max (float, optional): Maximum value for w0. Defaults to 0.2.
        w0_vary (bool, optional): Whether w0 varies. Defaults to None.
        w1_value (float, optional): Initial value for w1. Defaults to 1.0.
        w1_vary (bool, optional): Whether w1 varies. Defaults to False.
        c1_value (float, optional): Initial value for c1. Defaults to 0.
        c1_min (float, optional): Minimum value for c1. Defaults to -0.5.
        c1_max (float, optional): Maximum value for c1. Defaults to 0.5.
        c1_vary (bool, optional): Whether c1 varies. Defaults to True.
        t1_value (float, optional): Initial value for t1. Defaults to 1.
        t1_min (float, optional): Minimum value for t1. Defaults to 0.01.
        t1_max (float, optional): Maximum value for t1. Defaults to 5000.
        t1_vary (bool, optional): Whether t1 varies. Defaults to True.
        c2_value (float, optional): Initial value for c2. Defaults to 0.
        c2_min (float, optional): Minimum value for c2. Defaults to -0.5.
        c2_max (float, optional): Maximum value for c2. Defaults to 0.5.
        c2_vary (bool, optional): Whether c2 varies. Defaults to None.
        t2_value (float, optional): Initial value for t2. Defaults to 10.
        t2_min (float, optional): Minimum value for t2. Defaults to 0.01.
        t2_max (float, optional): Maximum value for t2. Defaults to 5000.
        t2_vary (bool, optional): Whether t2 varies. Defaults to None.
        c3_value (float, optional): Initial value for c3. Defaults to 0.
        c3_min (float, optional): Minimum value for c3. Defaults to -0.5.
        c3_max (float, optional): Maximum value for c3. Defaults to 0.5.
        c3_vary (bool, optional): Whether c3 varies. Defaults to None.
        t3_value (float, optional): Initial value for t3. Defaults to 50.
        t3_min (float, optional): Minimum value for t3. Defaults to 0.01.
        t3_max (float, optional): Maximum value for t3. Defaults to 5000.
        t3_vary (bool, optional): Whether t3 varies. Defaults to None.
        c4_value (float, optional): Initial value for c4. Defaults to 0.
        c4_min (float, optional): Minimum value for c4. Defaults to -0.5.
        c4_max (float, optional): Maximum value for c4. Defaults to 0.5.
        c4_vary (bool, optional): Whether c4 varies. Defaults to None.
        t4_value (float, optional): Initial value for t4. Defaults to 500.
        t4_min (float, optional): Minimum value for t4. Defaults to 0.01.
        t4_max (float, optional): Maximum value for t4. Defaults to 5000.
        t4_vary (bool, optional): Whether t4 varies. Defaults to None.
        w10_value (float, optional): Initial value for w10. Defaults to 0.0.
        w10_min (float, optional): Minimum value for w10. Defaults to -0.5.
        w10_max (float, optional): Maximum value for w10. Defaults to 0.5.
        w10_vary (bool, optional): Whether w10 varies. Defaults to True.
        w11_value (float, optional): Initial value for w11. Defaults to 0.0.
        w11_min (float, optional): Minimum value for w11. Defaults to -0.1.
        w11_max (float, optional): Maximum value for w11. Defaults to 0.1.
        w11_vary (bool, optional): Whether w11 varies. Defaults to True.
        w12_value (float, optional): Initial value for w12. Defaults to 0.0.
        w12_min (float, optional): Minimum value for w12. Defaults to -0.5.
        w12_max (float, optional): Maximum value for w12. Defaults to 0.5.
        w12_vary (bool, optional): Whether w12 varies. Defaults to None.

    Returns:
        lmfit.Parameters: Initialized parameters for the TA Analyzer.

    Note:
        - The vary flags for parameters are automatically set based on num_of_exp to enable
          only the parameters needed for the specified number of components.

    """
    params = lmfit.Parameters()
    params.add("w0", value=w0_value, min=w0_min, max=w0_max, vary=w0_vary)
    params.add("w1", value=w1_value, vary=w1_vary)
    params.add("w2", value=c1_value, min=c1_min, max=c1_max, vary=c1_vary)
    params.add("w3", value=t1_value, min=t1_min, max=t1_max, vary=t1_vary)
    if num_of_exp == 1:
        params.add("w4", value=c2_value, min=c2_min, max=c2_max, vary=False)
        params.add("w5", value=t2_value, min=t2_min, max=t2_max, vary=False)
        params.add("w6", value=c3_value, min=c3_min, max=c3_max, vary=False)
        params.add("w7", value=t3_value, min=t3_min, max=t3_max, vary=False)
        params.add("w8", value=c4_value, min=c4_min, max=c4_max, vary=False)
        params.add("w9", value=t4_value, min=t4_min, max=t4_max, vary=False)

    if num_of_exp == 2:
        params.add("w4", value=c2_value, min=c2_min, max=c2_max, vary=c2_vary)
        params.add("w5", value=t2_value, min=t2_min, max=t2_max, vary=t2_vary)
        params.add("w6", value=c3_value, min=c3_min, max=c3_max, vary=False)
        params.add("w7", value=t3_value, min=t3_min, max=t3_max, vary=False)
        params.add("w8", value=c4_value, min=c4_min, max=c4_max, vary=False)
        params.add("w9", value=t4_value, min=t4_min, max=t4_max, vary=False)

    if num_of_exp == 3:
        params.add("w4", value=c2_value, min=c2_min, max=c2_max, vary=c2_vary)
        params.add("w5", value=t2_value, min=t2_min, max=t2_max, vary=t2_vary)
        params.add("w6", value=c3_value, min=c3_min, max=c3_max, vary=c3_vary)
        params.add("w7", value=t3_value, min=t3_min, max=t3_max, vary=c3_vary)
        params.add("w8", value=c4_value, min=c4_min, max=c4_max, vary=False)
        params.add("w9", value=t4_value, min=t4_min, max=t4_max, vary=False)

    if num_of_exp == 4:
        params.add("w4", value=c2_value, min=c2_min, max=c2_max, vary=c2_vary)
        params.add("w5", value=t2_value, min=t2_min, max=t2_max, vary=t2_vary)
        params.add("w6", value=c3_value, min=c3_min, max=c3_max, vary=c3_vary)
        params.add("w7", value=t3_value, min=t3_min, max=t3_max, vary=t3_vary)
        params.add("w8", value=c4_value, min=c4_min, max=c4_max, vary=c4_vary)
        params.add("w9", value=t4_value, min=t4_min, max=t4_max, vary=t4_vary)

    params.add("w10", value=w10_value, min=w10_min, max=w10_max, vary=w10_vary)
    params.add("w11", value=w11_value, min=w11_min, max=w11_max, vary=w11_vary)
    params.add("w12", value=w12_value, min=w12_min, max=w12_max, vary=w12_vary)
    return params


def colorwaves(ax):
    """
    Change the colors of the lines in the given Axes object using a predefined color cycle.
    This function applies a custom color scheme to lines in a matplotlib plot,
    skipping lines that have labels starting with "_child".

    Args:
        ax (matplotlib.axes.Axes): The Axes object containing the lines to colorize.
    Notes:
        - Uses a predefined list of 10 colors in a cyclical pattern.
        - Only changes colors of lines that don't have labels starting with "_child".
        - Does not automatically display the legend.
    """

    # Ensure the number of colors matches the number of lines
    lines = ax.get_lines()
    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]
    color_cycle = cycle(colors)
    # Set the color for each line
    # for i, line in enumerate(lines):
    #     line.set_color(colors[i])
    for line in lines:
        if not (
            line.get_label().startswith("_child")
        ):  # Check if line has a label (and is not default)
            line.set_color(next(color_cycle))  # Assign color and move to next color
    # ax.legend()


def new_split_axes(
    figsize: tuple[int, int] = (8, 3),
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Create figure with two subplots sharing the y-axis
    and a specified width ratio.

    Args:
        figsize (tuple, optional): Figure size (width, height). Defaults to (8, 3).

    Returns:
        tuple: Figure and axes objects (fig, (ax1, ax2))
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, width_ratios=[0.3, 0.7], facecolor="w", figsize=figsize
    )
    fig.subplots_adjust(wspace=0.1)
    return fig, (ax1, ax2)


def plot_split_axes(
    fig: matplotlib.figure.Figure,
    axs_tuple: tuple[matplotlib.axes.Axes, matplotlib.axes.Axes],
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    fit_x: np.ndarray | None = None,
    fit_y: np.ndarray | None = None,
    time_split: float = 5.0,
    title: str | None = None,
    xlabel: str = "Time (ps)",
    ylabel: str = "ΔOD",
    label: str | None = None,
    fit_label: str | None = None,
    color_sequence: int = 0,
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """
    Create a plot with split axes - linear scale for early times and log scale for later times.

    Args:
        x (array-like): x-axis data (typically time values)
        y (array-like): y-axis data (typically signal amplitude)
        fit_curve (array-like, optional): Fitted curve data if available
        time_split (float, optional): Time point where to split between linear and log scale. Defaults to 5 ps.
        title (str, optional): Plot title
        xlabel (str, optional): x-axis label. Defaults to "Time (ps)".
        ylabel (str, optional): y-axis label. Defaults to "ΔOD".
        label (str, optional): Label for the data points
        fit_label (str, optional): Label for the fit curve
        figsize (tuple, optional): Figure size (width, height). Defaults to (8, 3).
        color_sequence (int, optional): Index for color selection from a predefined 10-color palette.

    Returns:
        tuple: Figure and axes objects (fig, (ax1, ax2))
    """
    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]
    color = colors[color_sequence % len(colors)]
    # Find the split point index in the data
    if x is not None:
        pt_split = np.searchsorted(x, time_split)
    else:
        pt_split = 0  # Provide a default value for pt_split
    if fit_x is not None:
        pt_split_fit = np.searchsorted(fit_x, time_split)
    else:
        pt_split_fit = 0  # Provide a default value for pt_split_fit
    ax1 = axs_tuple[0]
    ax2 = axs_tuple[1]

    if x is not None and y is not None:
        # Plot data points
        ax1.scatter(
            x[:pt_split],
            y[:pt_split],
            marker="o",
            s=50,
            facecolor="none",
            edgecolor=color,
            label=label if label else None,
        )
        ax2.scatter(
            x[pt_split:],
            y[pt_split:],
            marker="o",
            s=50,
            facecolor="none",
            color=color,
            label=label if label else None,
        )
        # Configure the axes
        ax1.set_xlim(-0.5, time_split)
        ax2.set_xlim(time_split, x[-1])

    # Plot fit curve if provided
    if fit_x is not None and fit_y is not None:
        ax1.plot(
            fit_x[:pt_split_fit],
            fit_y[:pt_split_fit],
            color=color,
            label=fit_label if fit_label else "Fit",
        )
        ax2.plot(fit_x[pt_split_fit:], fit_y[pt_split_fit:], color=color)
        ax1.set_xlim(-0.5, time_split)
        ax2.set_xlim(time_split, fit_x[-1])

    ax2.set_xscale("log")

    # Hide the right spine of the first subplot and the left spine of the second subplot
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Configure tick parameters
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright=False)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.yaxis.tick_right()

    # Add diagonal break lines to show the discontinuity
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)  # type: ignore
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)  # type: ignore

    # Add horizontal line at y=0
    ax1.axhline(0, color="black", linewidth=0.5, zorder=0)
    ax2.axhline(0, color="black", linewidth=0.5, zorder=0)

    # Add labels and title
    if title:
        fig.suptitle(title, fontsize=10, ha="center")
    # fig.text(0.5, 0.04, xlabel, ha="center", fontsize=8)
    ax2.set_xlabel(xlabel)
    ax2.xaxis.set_label_coords(0.2, -0.1)
    ax1.set_ylabel(ylabel)

    # Add legend if labels were provided
    if label or fit_label:
        ax2.legend(loc="best", ncols=1, fontsize=6)

    return fig, (ax1, ax2)

def load_streak_camera_data(file_path):
    """ Load streak camera data from a file.

    Parameters:
    file_path (str): Path to the streak camera data file.   
    Returns:
    tuple: A tuple containing:
        - time (numpy.ndarray): Time data from the streak camera.
        - wavelength (numpy.ndarray): Wavelength data from the streak camera.   
        - matrix (numpy.ndarray): Matrix data from the streak camera.
    """
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=0)
    time = data[1:, 0]
    matrix = data[1:, 1:].T # Transpose to have time as columns and wavelengths as rows
    wavelength = data[0, 1:]
    np.savetxt(file_path.replace('.dac', '_time.txt'), time)
    np.savetxt(file_path.replace('.dac', '_wavelength.txt'), wavelength)
    np.savetxt(file_path.replace('.dac', '_matrix.txt'), matrix)
    return time, wavelength, matrix

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def fit_and_plot_gaussian(time_data, kinetic_data, plot=True):
    """
    Perform Gaussian fit on kinetic trace data and optionally plot the results.
    
    Parameters:
    -----------
    time_data : array-like
        Time values
    kinetic_data : array-like
        Kinetic trace data to fit
    wavelength : int or float
        Wavelength of the kinetic trace
    plot : bool, default=True
        Whether to plot the results
        
    Returns:
    --------
    dict
        Dictionary containing fit results and parameters
    """
    # Create Gaussian model
    gmodel = lmfit.Model(gaussian)
    params = gmodel.make_params(amp=np.max(kinetic_data), 
                                cen=time_data[np.argmax(kinetic_data)], 
                                wid=1.0)

    # Perform the fit
    result = gmodel.fit(kinetic_data, params, x=time_data)

    # Get parameters
    amp = result.params['amp'].value
    cen = result.params['cen'].value
    wid = result.params['wid'].value
    
    # Calculate CDF
    fit_cdf = norm.cdf(time_data, loc=cen, scale=wid) * amp
    
    if plot:
        # Plot data and fit
        plt.figure(figsize=(10, 6))
        plt.plot(time_data, kinetic_data, 'o', label='Data')
        plt.plot(time_data, result.best_fit, '-', label='Gaussian fit')
        plt.plot(time_data, fit_cdf, 'g--', label='Gaussian CDF')
        plt.legend()
        plt.xlabel('Time (ps)')
        plt.ylabel('ΔA')
        plt.title('Gaussian Fit of Kinetic Trace')
    
    # Print fit report
    print(result.fit_report())
    
    return {
        'result': result,
        'amp': amp,
        'cen': cen,
        'wid': wid,
        'fit_cdf': fit_cdf
    }