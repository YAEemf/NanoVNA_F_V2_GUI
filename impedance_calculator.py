"""
Impedance Calculator using Shunt-Through Method
Calculates impedance from S-parameters measured by VNA
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImpedanceData:
    """Data class for impedance measurement results"""
    frequencies: np.ndarray  # Frequencies in Hz
    impedances: np.ndarray   # Complex impedances in Ohms
    magnitudes: np.ndarray   # Impedance magnitudes in Ohms
    phases: np.ndarray       # Impedance phases in degrees
    s11: np.ndarray          # S11 complex data
    s21: np.ndarray          # S21 complex data


class ImpedanceCalculator:
    """
    Calculate impedance from S-parameters using shunt-through method

    The shunt-through method measures a DUT (Device Under Test) connected
    between Port1 and Port2 of a VNA in shunt configuration.
    """

    def __init__(self, z0: float = 50.0):
        """
        Initialize impedance calculator

        Args:
            z0: Characteristic impedance in Ohms (default: 50Ω)
        """
        self.z0 = z0

    def calculate_from_s21_shunt(self, frequencies: np.ndarray, s21: np.ndarray) -> ImpedanceData:
        """
        Calculate impedance from S21 using shunt-through method

        For shunt configuration (DUT connected in parallel between Port1 and Port2):
        Z_shunt = 2 * Z0 / ((1/S21) - 1)

        Args:
            frequencies: Array of frequencies in Hz
            s21: Array of complex S21 values

        Returns:
            ImpedanceData object containing calculated impedance data
        """
        # Calculate impedance using shunt-through formula
        # Z = 2 * Z0 / ((1/S21) - 1)
        # Avoid division by zero
        s21_safe = np.where(np.abs(s21) < 1e-10, 1e-10 + 0j, s21)

        impedances = 2.0 * self.z0 / ((1.0 / s21_safe) - 1.0)

        # Calculate magnitude and phase
        magnitudes = np.abs(impedances)
        phases = np.angle(impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=np.zeros_like(s21),  # Not used in this calculation
            s21=s21
        )

    def calculate_from_s11_reflection(self, frequencies: np.ndarray, s11: np.ndarray) -> ImpedanceData:
        """
        Calculate impedance from S11 using reflection method

        For reflection measurement:
        Z = Z0 * (1 + S11) / (1 - S11)

        Args:
            frequencies: Array of frequencies in Hz
            s11: Array of complex S11 values

        Returns:
            ImpedanceData object containing calculated impedance data
        """
        # Calculate impedance using reflection formula
        # Z = Z0 * (1 + S11) / (1 - S11)
        # Avoid division by zero
        denominator = 1.0 - s11
        denominator_safe = np.where(np.abs(denominator) < 1e-10, 1e-10 + 0j, denominator)

        impedances = self.z0 * (1.0 + s11) / denominator_safe

        # Calculate magnitude and phase
        magnitudes = np.abs(impedances)
        phases = np.angle(impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=s11,
            s21=np.zeros_like(s11)  # Not used in this calculation
        )

    def calculate_from_s21_series(self, frequencies: np.ndarray, s21: np.ndarray) -> ImpedanceData:
        """
        Calculate impedance from S21 for series configuration

        For series configuration (DUT connected in series):
        Z_series = Z0 * (1 - S21) / (2 * S21)

        Args:
            frequencies: Array of frequencies in Hz
            s21: Array of complex S21 values

        Returns:
            ImpedanceData object containing calculated impedance data
        """
        # Calculate impedance using series formula
        # Z = Z0 * (1 - S21) / (2 * S21)
        # Avoid division by zero
        s21_safe = np.where(np.abs(s21) < 1e-10, 1e-10 + 0j, s21)

        impedances = self.z0 * (1.0 - s21_safe) / (2.0 * s21_safe)

        # Calculate magnitude and phase
        magnitudes = np.abs(impedances)
        phases = np.angle(impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=np.zeros_like(s21),
            s21=s21
        )

    def calculate_average(self, impedance_data_list: List[ImpedanceData]) -> ImpedanceData:
        """
        Calculate average of multiple impedance measurements

        If measurements have different number of points, they will be trimmed
        to the minimum length to ensure compatibility.

        Args:
            impedance_data_list: List of ImpedanceData objects to average

        Returns:
            Averaged ImpedanceData object
        """
        if not impedance_data_list:
            raise ValueError("Empty impedance data list")

        if len(impedance_data_list) == 1:
            # No averaging needed for single measurement
            return impedance_data_list[0]

        # Find minimum length across all measurements
        lengths = [len(data.frequencies) for data in impedance_data_list]
        min_length = min(lengths)
        max_length = max(lengths)

        # Warn if measurements have different lengths
        if min_length != max_length:
            print(f"Warning: Measurements have different lengths ({min_length} to {max_length} points)")
            print(f"Trimming all measurements to {min_length} points for averaging")

        # Trim all data to minimum length
        trimmed_data = []
        for data in impedance_data_list:
            trimmed = ImpedanceData(
                frequencies=data.frequencies[:min_length],
                impedances=data.impedances[:min_length],
                magnitudes=data.magnitudes[:min_length],
                phases=data.phases[:min_length],
                s11=data.s11[:min_length],
                s21=data.s21[:min_length]
            )
            trimmed_data.append(trimmed)

        # Use first measurement's frequencies as reference
        frequencies = trimmed_data[0].frequencies

        # Average complex impedances
        impedances_array = np.array([data.impedances for data in trimmed_data])
        avg_impedances = np.mean(impedances_array, axis=0)

        # Average S-parameters
        s11_array = np.array([data.s11 for data in trimmed_data])
        s21_array = np.array([data.s21 for data in trimmed_data])
        avg_s11 = np.mean(s11_array, axis=0)
        avg_s21 = np.mean(s21_array, axis=0)

        # Recalculate magnitude and phase from averaged complex impedance
        magnitudes = np.abs(avg_impedances)
        phases = np.angle(avg_impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=avg_impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=avg_s11,
            s21=avg_s21
        )

    def calculate_with_reference(
        self,
        frequencies: np.ndarray,
        s21_dut: np.ndarray,
        s21_ref: np.ndarray
    ) -> ImpedanceData:
        """
        Calculate impedance with reference (through) measurement

        This method uses a reference measurement (direct connection) to
        normalize the DUT measurement.

        Args:
            frequencies: Array of frequencies in Hz
            s21_dut: S21 measured with DUT
            s21_ref: S21 reference measurement (through connection)

        Returns:
            ImpedanceData object containing calculated impedance data
        """
        # Normalize S21 by reference
        s21_ref_safe = np.where(np.abs(s21_ref) < 1e-10, 1e-10 + 0j, s21_ref)
        s21_normalized = s21_dut / s21_ref_safe

        # Calculate impedance from normalized S21
        return self.calculate_from_s21_shunt(frequencies, s21_normalized)

    def calculate_parallel_components(
        self,
        frequencies: np.ndarray,
        impedances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate parallel R and C components from complex impedance

        Assumes parallel RC model: Z = R || (1/jωC)

        Args:
            frequencies: Array of frequencies in Hz
            impedances: Array of complex impedances

        Returns:
            Tuple of (R_parallel, C_parallel) arrays
        """
        # Y = 1/Z = 1/R + jωC
        admittances = 1.0 / impedances

        conductances = np.real(admittances)  # G = 1/R
        susceptances = np.imag(admittances)  # B = ωC

        # R = 1/G
        r_parallel = np.where(np.abs(conductances) > 1e-10, 1.0 / conductances, np.inf)

        # C = B / ω
        omega = 2.0 * np.pi * frequencies
        c_parallel = susceptances / omega

        return r_parallel, c_parallel

    def calculate_series_components(
        self,
        frequencies: np.ndarray,
        impedances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate series R and C components from complex impedance

        Assumes series RC model: Z = R + 1/jωC

        Args:
            frequencies: Array of frequencies in Hz
            impedances: Array of complex impedances

        Returns:
            Tuple of (R_series, C_series) arrays
        """
        r_series = np.real(impedances)
        reactances = np.imag(impedances)

        # For capacitive reactance: X = -1/(ωC), so C = -1/(ωX)
        omega = 2.0 * np.pi * frequencies
        c_series = np.where(
            np.abs(reactances) > 1e-10,
            -1.0 / (omega * reactances),
            np.inf
        )

        return r_series, c_series


if __name__ == "__main__":
    # Test code
    print("Impedance Calculator Test")
    print("-" * 50)

    # Create test data
    frequencies = np.linspace(100e6, 200e6, 51)  # 100-200 MHz, 51 points

    # Simulate S21 data for a shunt impedance of 100Ω
    z_test = 100.0  # 100Ω
    z0 = 50.0

    # From shunt formula: S21 = 2*Z0 / (2*Z0 + Z_shunt)
    s21_test = 2 * z0 / (2 * z0 + z_test)
    s21_array = np.full(len(frequencies), s21_test, dtype=complex)

    # Calculate impedance
    calc = ImpedanceCalculator(z0=z0)
    impedance_data = calc.calculate_from_s21_shunt(frequencies, s21_array)

    print(f"\nTest: Input impedance = {z_test} Ω")
    print(f"Calculated impedance (first point) = {impedance_data.magnitudes[0]:.2f} Ω")
    print(f"Calculated impedance (mean) = {np.mean(impedance_data.magnitudes):.2f} Ω")
    print(f"Phase (mean) = {np.mean(impedance_data.phases):.2f}°")

    # Test with S11
    print("\n" + "-" * 50)
    print("Test with S11 (reflection method)")

    # Simulate S11 for 75Ω impedance
    z_test_s11 = 75.0
    s11_test = (z_test_s11 - z0) / (z_test_s11 + z0)
    s11_array = np.full(len(frequencies), s11_test, dtype=complex)

    impedance_data_s11 = calc.calculate_from_s11_reflection(frequencies, s11_array)

    print(f"\nTest: Input impedance = {z_test_s11} Ω")
    print(f"Calculated impedance (first point) = {impedance_data_s11.magnitudes[0]:.2f} Ω")
    print(f"Calculated impedance (mean) = {np.mean(impedance_data_s11.magnitudes):.2f} Ω")
