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
        S21 = 2*Z0 / (Zdut + 2*Z0)
        Therefore: Zdut = 2*Z0 * ((1/S21) - 1) = 2*Z0 * (1 - S21) / S21

        Args:
            frequencies: Array of frequencies in Hz
            s21: Array of complex S21 values

        Returns:
            ImpedanceData object containing calculated impedance data
        """
        # Calculate impedance using shunt-through formula
        # Z = 2*Z0 * ((1/S21) - 1) = 2*Z0 * (1 - S21) / S21
        # Avoid division by zero
        s21_safe = np.where(np.abs(s21) < 1e-10, 1e-10 + 0j, s21)

        # Correct formula: Z = 2*Z0 * ((1/S21) - 1)
        impedances = 2.0 * self.z0 * ((1.0 / s21_safe) - 1.0)

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

        For series configuration (DUT connected in series between Port1 and Port2):
        S21 = Z / (Z + 2*Z0)
        Therefore: Z = 2*Z0*S21 / (1 - S21)

        Args:
            frequencies: Array of frequencies in Hz
            s21: Array of complex S21 values

        Returns:
            ImpedanceData object containing calculated impedance data
        """
        # Calculate impedance using series formula
        # Z = 2*Z0*S21 / (1 - S21)
        # Avoid division by zero
        s21_safe = np.where(np.abs(s21) < 1e-10, 1e-10 + 0j, s21)
        denominator = 1.0 - s21_safe
        denominator_safe = np.where(np.abs(denominator) < 1e-10, 1e-10 + 0j, denominator)

        # Correct formula: Z = 2*Z0*S21 / (1 - S21)
        impedances = 2.0 * self.z0 * s21_safe / denominator_safe

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

    def calculate_average(
        self,
        impedance_data_list: List[ImpedanceData],
        mode: str = "mean"
    ) -> ImpedanceData:
        """
        Calculate average of multiple impedance measurements using various methods

        If measurements have different number of points, they will be trimmed
        to the minimum length to ensure compatibility.

        Args:
            impedance_data_list: List of ImpedanceData objects to average
            mode: Averaging method
                - "mean": Simple arithmetic mean (default)
                - "median": Median (robust to outliers)
                - "trimmed": Trimmed mean (removes top/bottom 10%)
                - "robust": Robust mean with MAD-based outlier rejection

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

        # Prepare data arrays
        impedances_array = np.array([data.impedances for data in trimmed_data])
        s11_array = np.array([data.s11 for data in trimmed_data])
        s21_array = np.array([data.s21 for data in trimmed_data])

        # Apply averaging method
        if mode == "mean":
            avg_impedances = self._mean_average(impedances_array)
            avg_s11 = self._mean_average(s11_array)
            avg_s21 = self._mean_average(s21_array)
        elif mode == "median":
            avg_impedances = self._median_average(impedances_array)
            avg_s11 = self._median_average(s11_array)
            avg_s21 = self._median_average(s21_array)
        elif mode == "trimmed":
            avg_impedances = self._trimmed_mean_average(impedances_array)
            avg_s11 = self._trimmed_mean_average(s11_array)
            avg_s21 = self._trimmed_mean_average(s21_array)
        elif mode == "robust":
            avg_impedances = self._robust_average(impedances_array)
            avg_s11 = self._robust_average(s11_array)
            avg_s21 = self._robust_average(s21_array)
        else:
            raise ValueError(f"Unknown averaging mode: {mode}")

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

    def _mean_average(self, data_array: np.ndarray) -> np.ndarray:
        """
        Simple arithmetic mean

        Args:
            data_array: Array of shape (n_measurements, n_points)

        Returns:
            Averaged array of shape (n_points,)
        """
        return np.mean(data_array, axis=0)

    def _median_average(self, data_array: np.ndarray) -> np.ndarray:
        """
        Median (most robust to outliers)

        For complex numbers, compute median of real and imaginary parts separately.

        Args:
            data_array: Array of shape (n_measurements, n_points)

        Returns:
            Median array of shape (n_points,)
        """
        if np.iscomplexobj(data_array):
            # For complex data, compute median of real and imaginary parts separately
            real_median = np.median(data_array.real, axis=0)
            imag_median = np.median(data_array.imag, axis=0)
            return real_median + 1j * imag_median
        else:
            return np.median(data_array, axis=0)

    def _trimmed_mean_average(self, data_array: np.ndarray, trim_percent: float = 0.1) -> np.ndarray:
        """
        Trimmed mean (removes extreme values before averaging)

        Removes the top and bottom trim_percent of values before computing mean.
        For complex numbers, trimming is based on magnitude.

        Args:
            data_array: Array of shape (n_measurements, n_points)
            trim_percent: Percentage to trim from each end (default: 0.1 = 10%)

        Returns:
            Trimmed mean array of shape (n_points,)
        """
        from scipy import stats

        if np.iscomplexobj(data_array):
            # For complex data, trim based on magnitude
            magnitudes = np.abs(data_array)

            # Compute trimmed mean for each frequency point
            result = np.zeros(data_array.shape[1], dtype=complex)

            for i in range(data_array.shape[1]):
                # Get data for this frequency point
                point_data = data_array[:, i]
                point_mags = magnitudes[:, i]

                # Sort by magnitude
                sorted_indices = np.argsort(point_mags)

                # Calculate trim count
                n = len(point_data)
                trim_count = int(n * trim_percent)

                # Remove extreme values based on magnitude
                if trim_count > 0 and n > 2 * trim_count:
                    trimmed_indices = sorted_indices[trim_count:-trim_count]
                    result[i] = np.mean(point_data[trimmed_indices])
                else:
                    # Not enough data to trim, use mean
                    result[i] = np.mean(point_data)

            return result
        else:
            # For real data, use scipy's trimmed mean
            return stats.trim_mean(data_array, trim_percent, axis=0)

    def _robust_average(self, data_array: np.ndarray, mad_threshold: float = 3.0) -> np.ndarray:
        """
        Robust average using MAD (Median Absolute Deviation) for outlier rejection

        This method:
        1. Computes the median for each point
        2. Computes MAD (Median Absolute Deviation)
        3. Rejects outliers beyond mad_threshold * MAD
        4. Averages remaining values

        For complex numbers, outlier detection is based on magnitude deviation.

        Args:
            data_array: Array of shape (n_measurements, n_points)
            mad_threshold: Threshold for outlier rejection (default: 3.0)

        Returns:
            Robust averaged array of shape (n_points,)
        """
        if np.iscomplexobj(data_array):
            # For complex data, use magnitude-based outlier detection
            magnitudes = np.abs(data_array)

            # Compute robust average for each frequency point
            result = np.zeros(data_array.shape[1], dtype=complex)

            for i in range(data_array.shape[1]):
                # Get data for this frequency point
                point_data = data_array[:, i]
                point_mags = magnitudes[:, i]

                # Compute median and MAD for magnitudes
                median_mag = np.median(point_mags)
                mad = np.median(np.abs(point_mags - median_mag))

                # Avoid division by zero
                if mad < 1e-10:
                    # All values are very similar, use mean
                    result[i] = np.mean(point_data)
                else:
                    # Identify inliers (non-outliers)
                    # MAD-based threshold: |magnitude - median_magnitude| < threshold * MAD
                    deviation = np.abs(point_mags - median_mag)
                    inlier_mask = deviation < (mad_threshold * mad)

                    # Use mean of inliers
                    if np.sum(inlier_mask) > 0:
                        result[i] = np.mean(point_data[inlier_mask])
                    else:
                        # All points rejected (shouldn't happen), use median
                        real_median = np.median(point_data.real)
                        imag_median = np.median(point_data.imag)
                        result[i] = real_median + 1j * imag_median

            return result
        else:
            # For real data
            result = np.zeros(data_array.shape[1])

            for i in range(data_array.shape[1]):
                point_data = data_array[:, i]

                # Compute median and MAD
                median_val = np.median(point_data)
                mad = np.median(np.abs(point_data - median_val))

                if mad < 1e-10:
                    result[i] = np.mean(point_data)
                else:
                    # Identify inliers
                    deviation = np.abs(point_data - median_val)
                    inlier_mask = deviation < (mad_threshold * mad)

                    if np.sum(inlier_mask) > 0:
                        result[i] = np.mean(point_data[inlier_mask])
                    else:
                        result[i] = median_val

            return result

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
    print("=" * 70)

    z0 = 50.0
    calc = ImpedanceCalculator(z0=z0)
    frequencies = np.linspace(100e6, 200e6, 51)  # 100-200 MHz, 51 points

    # ========== Test 1: Shunt method ==========
    print("\n[Test 1] Shunt (S21) method")
    print("-" * 70)

    z_test_shunt = 100.0  # 100 Ohm
    # From shunt formula: S21 = 2*Z0 / (Zdut + 2*Z0)
    s21_test_shunt = 2 * z0 / (z_test_shunt + 2 * z0)
    s21_array_shunt = np.full(len(frequencies), s21_test_shunt, dtype=complex)

    print(f"Input impedance:     {z_test_shunt} Ohm")
    print(f"Calculated S21:      {s21_test_shunt:.6f}")
    print(f"Expected S21:        {2*50/(100+100):.6f} = 0.5")

    impedance_data_shunt = calc.calculate_from_s21_shunt(frequencies, s21_array_shunt)

    print(f"\nReconstructed impedance: {impedance_data_shunt.magnitudes[0]:.2f} Ohm")
    print(f"Error:                   {abs(impedance_data_shunt.magnitudes[0] - z_test_shunt):.6f} Ohm")
    assert abs(impedance_data_shunt.magnitudes[0] - z_test_shunt) < 0.01, "Shunt calculation error!"
    print("[OK] Shunt method verified")

    # ========== Test 2: Reflection (S11) method ==========
    print("\n[Test 2] Reflection (S11) method")
    print("-" * 70)

    z_test_s11 = 75.0  # 75 Ohm
    # From reflection formula: S11 = (Z - Z0) / (Z + Z0)
    s11_test = (z_test_s11 - z0) / (z_test_s11 + z0)
    s11_array = np.full(len(frequencies), s11_test, dtype=complex)

    print(f"Input impedance:     {z_test_s11} Ohm")
    print(f"Calculated S11:      {s11_test:.6f}")
    print(f"Expected S11:        {(75-50)/(75+50):.6f} = 0.2")

    impedance_data_s11 = calc.calculate_from_s11_reflection(frequencies, s11_array)

    print(f"\nReconstructed impedance: {impedance_data_s11.magnitudes[0]:.2f} Ohm")
    print(f"Error:                   {abs(impedance_data_s11.magnitudes[0] - z_test_s11):.6f} Ohm")
    assert abs(impedance_data_s11.magnitudes[0] - z_test_s11) < 0.01, "S11 calculation error!"
    print("[OK] S11 method verified")

    # ========== Test 3: Series (S21) method ==========
    print("\n[Test 3] Series (S21) method")
    print("-" * 70)

    z_test_series = 50.0  # 50 Ohm
    # From series formula: S21 = Z / (Z + 2*Z0)
    s21_test_series = z_test_series / (z_test_series + 2 * z0)
    s21_array_series = np.full(len(frequencies), s21_test_series, dtype=complex)

    print(f"Input impedance:     {z_test_series} Ohm")
    print(f"Calculated S21:      {s21_test_series:.6f}")
    print(f"Expected S21:        {50/(50+100):.6f} = 0.333...")

    impedance_data_series = calc.calculate_from_s21_series(frequencies, s21_array_series)

    print(f"\nReconstructed impedance: {impedance_data_series.magnitudes[0]:.2f} Ohm")
    print(f"Error:                   {abs(impedance_data_series.magnitudes[0] - z_test_series):.6f} Ohm")
    assert abs(impedance_data_series.magnitudes[0] - z_test_series) < 0.01, "Series calculation error!"
    print("[OK] Series method verified")

    # ========== Test 4: Consistency check ==========
    print("\n[Test 4] Same DUT measured with S11 and Shunt should give similar results")
    print("-" * 70)

    z_dut = 100.0  # 100 Ohm DUT

    # S11 measurement
    s11_dut = (z_dut - z0) / (z_dut + z0)
    s11_array_dut = np.full(len(frequencies), s11_dut, dtype=complex)
    z_from_s11 = calc.calculate_from_s11_reflection(frequencies, s11_array_dut)

    # Shunt measurement
    s21_dut = 2 * z0 / (z_dut + 2 * z0)
    s21_array_dut = np.full(len(frequencies), s21_dut, dtype=complex)
    z_from_shunt = calc.calculate_from_s21_shunt(frequencies, s21_array_dut)

    print(f"DUT impedance:           {z_dut} Ohm")
    print(f"From S11 (Reflection):   {z_from_s11.magnitudes[0]:.2f} Ohm")
    print(f"From S21 (Shunt):        {z_from_shunt.magnitudes[0]:.2f} Ohm")
    print(f"Difference:              {abs(z_from_s11.magnitudes[0] - z_from_shunt.magnitudes[0]):.6f} Ohm")
    assert abs(z_from_s11.magnitudes[0] - z_from_shunt.magnitudes[0]) < 0.01, "S11 and Shunt should give same result!"
    print("[OK] Consistency verified - S11 and Shunt give identical results")

    print("\n" + "=" * 70)
    print("All tests passed! [OK]")
    print("=" * 70)
