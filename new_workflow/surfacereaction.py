from adsorbate import Adsorbate, Adsorbates
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit

class SurfaceReaction:
    """
    This class is designed to take in 2 Adsorbate objects
    which are taken to be the initial state and the first-order saddle-point.
    Methods intended to be used are get_arrhenius_parameters, plot_arrhenius,
    write_RMG_library_entry, write_RMG_dictionary_entry.

    We may want to think further about how we instantiate these objects in the
    future, using an "Adsorbate" for the initial state probably doesn't make
    sense. We should later on make a StationaryPoint class which handles thermo
    generally and think about how to package input more cleanly.
    """
    def __init__(self,
                 initial_states: list[Adsorbate],
                 first_order_saddle_point: Adsorbate,
                 vacant_sites_required: int,
                 ):

        self.initial_states = initial_states
        self.first_order_saddle_point = first_order_saddle_point
        self.vacant_sites_required = vacant_sites_required
        self.reacant_number = len(initial_states) + self.vacant_sites_required
        self._get_delta_energy()

    def _get_delta_energy(self):
        E_init = 0
        for IS in self.initial_states:
            E_init += IS.dft_energy[0] + IS.zpe[0]
        nslabs = len(self.initial_states) - 1
        slabE = self.first_order_saddle_point.reference_energies['slab']
        E_fosp = self.first_order_saddle_point.dft_energy[0] \
            + self.first_order_saddle_point.zpe[0]
        dE = E_fosp + slabE * nslabs - E_init
        self.delta_energy = dE
        return

    def get_arrhenius_parameters(self):
        temps = self.first_order_saddle_point.temperatures.copy()
        q_is = 1
        for IS in self.initial_states:
            qtmp, _, _, _ = IS.get_thermo()
            q_is *= qtmp
        q_ts, _, _, _ = self.first_order_saddle_point.get_thermo()
        dE = self.delta_energy
        A, b, Ea = self._parameterize_arrhenius(temps, q_is, q_ts, dE)
        site_area = self.first_order_saddle_point.unit_cell_area
        N_A = 6.022e23
        site_dens = 1 / site_area / N_A
        gamma = 1 / site_dens ** site_dens
        A *= gamma
        return A, b, Ea

    def _parameterize_arrhenius(self,
                                temps: list[float],  # K
                                q_is: list[float],
                                q_ts: list[float],
                                dE: float,  # eV
                                ) -> tuple[float, float, float]:  # [m^2/mol/sec], [], [kJ/mol]
        A, b, Ea = 0, 0, 0
        """
        ToDo for Jully
        temps = self.initial_state.temperatures
        q_is, _, _, _ = self.initial_state.get_thermo()
        q_ts, _, _, _ = self.first_order_saddle_point.get_thermo()
        q_ratio = q_ts / q_is
        kfwd=f(q(T)
        please take this q ratio, and the temperatures and
        fit the arrhenious constants accourding to:
            https://cantera.org/3.1/python/kinetics.html#arrheniusrate
        """
        # k(T) = (k_B T / h) * (q_ts / q_is) * exp(-dE / (k_B T))
        # k(T) = A * T^b * exp(-Ea / (k_B T))
        # Ea would be eV

        # Boltzmann constant in eV/K
        kB_eV = constants.physical_constants["Boltzmann constant in eV/K"][0]

        # TST forward rate constant
        q_ratio = q_ts / q_is
        #kfwd = (constants.k * temps / constants.h) * q_ratio * np.exp(-dE / (kB_eV * temps))
        site_area = self.first_order_saddle_point.unit_cell_area
        N_A = constants.Avogadro

        kfwd = ((constants.k * temps / constants.h) * q_ratio * np.exp(-dE / (kB_eV * temps))) * site_area * N_A

        # Fit ln(k) = ln(A) + b ln(T) - Ea/(kB*T)
        # b = 0 
        y = np.log(kfwd)
        x = 1.0 / temps

        slope, intercept = np.polyfit(x, y, 1)
        A = np.exp(intercept)
        b = 0.0
        Ea_eV = -slope * kB_eV   # eV
        Ea = Ea_eV * constants.eV * constants.Avogadro / 1000
        return float(A), float(b), float(Ea)

    def plot_arrhenius(self, filename=None):
        """
        ToDo for Jully
        please save a png of an arrhenious plot to filename
        please use a log y axis instead of log(k), and please
        use 1000/T for the x axis.
        """
        #x-axis: 1000 / T
        #y-axis: k on a log scale
        
        temps = self.first_order_saddle_point.temperatures.copy()

        q_is = 1
        for IS in self.initial_states:
            qtmp, _, _, _ = IS.get_thermo()
            q_is *= qtmp

        q_ts, _, _, _ = self.first_order_saddle_point.get_thermo()
        dE = self.delta_energy

        temps = np.asarray(temps, dtype=float)
        q_is = np.asarray(q_is, dtype=float)
        q_ts = np.asarray(q_ts, dtype=float)

        kB_eV = constants.physical_constants["Boltzmann constant in eV/K"][0]

        # raw TST-like rates
        q_ratio = q_ts / q_is
        k_raw = (constants.k * temps / constants.h) * q_ratio * np.exp(-dE / (kB_eV * temps))

        # fitted Arrhenius rates
        A, b, Ea = self._parameterize_arrhenius(temps, q_is, q_ts, dE)
        k_fit = A * (temps ** b) * np.exp(-Ea / (kB_eV * temps))

        x = 1000.0 / temps

        plt.figure()
        plt.semilogy(x, k_raw, "o", label="TST")
        plt.semilogy(x, k_fit, "-", label=f"Fit: A={A:.3e}, b={b:.3f}, Ea={Ea:.3f} eV")
        plt.xlabel("1000 / T (K$^{-1}$)")
        plt.ylabel("k")
        plt.legend()
        plt.tight_layout()
        plt.subplot()

        return

    def write_RMG_library_entry(self):
        """
        leave for kirk to write later
        """
        pass

    def write_RMG_dictionary_entry(self):
        """
        leave for kirk to write later
        """
        pass


class SurfaceReactions:
    """
    This class is a wrapper around the Reaction class, designed to handle
    multiple reactions simultaneously.
    """
    def __init__(self,
                 initial_states: Adsorbates,
                 first_order_saddle_points: Adsorbates
                 ):
        self.initial_states = initial_states
        self.first_order_saddle_points = first_order_saddle_points
        self.reaction_list = []

        for i in range(len(initial_states)):
            reaction = SurfaceReaction(initial_states[i],
                                       first_order_saddle_points[i])
            self.reaction_list.append(reaction)

    def get_arrhenius_parameters(self):
        parameters = []
        for rxn in self.reaction_list:
            parameters.append(rxn.get_arrhenius_parameters())
        return parameters

    def plot_arrhenius_fits(self):
        for i, rxn in enumerate(self.reaction_list):
            filename = f"arrhenius_fit_{i}.png"
            rxn.plot_arrhenius(filename=filename)
        return

    def write_RMG_library_entries(self):
        """
        Leave for kirk to write later
        """
        pass

    def write_RMG_dictionary_entries(self):
        """
        Leave for kirk to write later
        """
        pass
