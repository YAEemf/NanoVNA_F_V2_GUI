"""
NanoVNA-F v2 Serial Communication Controller
Handles serial communication with NanoVNA-F v2 device
"""

import serial
import serial.tools.list_ports
import time
import re
from typing import Optional, List, Tuple
import numpy as np


class NanoVNAController:
    """Controller for NanoVNA-F v2 device communication"""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 5.0, debug: bool = False):
        """
        Initialize NanoVNA controller

        Args:
            port: COM port name (e.g., 'COM3'). If None, auto-detect
            baudrate: Baud rate (default: 115200)
            timeout: Serial timeout in seconds
            debug: Enable debug output
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug = debug
        self.serial: Optional[serial.Serial] = None

    def auto_detect_port(self) -> Optional[str]:
        """
        Auto-detect NanoVNA COM port on Windows

        Returns:
            Port name if found, None otherwise
        """
        ports = serial.tools.list_ports.comports()

        # Try to find NanoVNA device by VID/PID or description
        for port in ports:
            # NanoVNA-F v2 typically appears as USB Serial Device
            if self.debug:
                print(f"Found port: {port.device} - {port.description} - VID:PID={port.vid}:{port.pid}")

            # Common VID:PID for NanoVNA-F v2: 0483:5740 (STM32 Virtual COM Port)
            if port.vid == 0x0483 and port.pid == 0x5740:
                if self.debug:
                    print(f"NanoVNA-F v2 detected on {port.device}")
                return port.device

            # Fallback: check description
            if "NanoVNA" in port.description or "STM32 Virtual ComPort" in port.description:
                if self.debug:
                    print(f"Possible NanoVNA device on {port.device}")
                return port.device

        # If no specific device found, return first available COM port
        if ports:
            if self.debug:
                print(f"No NanoVNA detected, using first available port: {ports[0].device}")
            return ports[0].device

        return None

    def connect(self) -> bool:
        """
        Connect to NanoVNA device

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.port is None:
                self.port = self.auto_detect_port()

            if self.port is None:
                print("Error: No COM port available")
                return False

            if self.debug:
                print(f"Connecting to {self.port}...")

            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # Wait for device to be ready
            time.sleep(0.5)

            # Clear input buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            if self.debug:
                print(f"Connected to {self.port}")

            return True

        except Exception as e:
            print(f"Error connecting to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from NanoVNA device"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            if self.debug:
                print("Disconnected from NanoVNA")

    def send_command(self, command: str) -> str:
        """
        Send command to NanoVNA and get response

        Args:
            command: Command string (without newline)

        Returns:
            Response string
        """
        if not self.serial or not self.serial.is_open:
            raise RuntimeError("Not connected to NanoVNA")

        # Clear buffers
        self.serial.reset_input_buffer()

        # Send command with newline
        cmd_bytes = (command + '\n').encode('ascii')
        self.serial.write(cmd_bytes)

        if self.debug:
            print(f"TX: {command}")

        # Read response
        response_lines = []
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                break

            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('ascii', errors='ignore').strip()
                if line:
                    response_lines.append(line)
                    if self.debug:
                        print(f"RX: {line}")

                    # Check if response is complete (ends with 'ch>' prompt)
                    if line.endswith('ch>'):
                        break
            else:
                time.sleep(0.01)

        return '\n'.join(response_lines)

    def get_version(self) -> str:
        """Get NanoVNA firmware version"""
        return self.send_command("version")

    def set_sweep_parameters(self, start_freq: int, stop_freq: int, points: int):
        """
        Set sweep parameters

        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            points: Number of sweep points (11-301)
        """
        if points < 11 or points > 301:
            raise ValueError("Points must be between 11 and 301")

        command = f"sweep {start_freq} {stop_freq} {points}"
        response = self.send_command(command)
        return response

    def scan(self, start_freq: int, stop_freq: int, points: int, outmask: int = 7) -> List[Tuple[float, complex, complex]]:
        """
        Perform scan and get measurement data

        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            points: Number of sweep points (11-301)
            outmask: Output format mask
                0: No output
                1: Frequency only
                2: S11 data only
                3: Frequency + S11 data
                4: S21 data only
                5: Frequency + S21 data
                6: S11 data + S21 data
                7: Frequency + S11 data + S21 data (recommended)

        Returns:
            List of tuples: (frequency, S11_complex, S21_complex)
        """
        if points < 11 or points > 301:
            raise ValueError("Points must be between 11 and 301")

        command = f"scan {start_freq} {stop_freq} {points} {outmask}"

        if self.debug:
            print(f"Scanning: {start_freq/1e6:.1f} MHz to {stop_freq/1e6:.1f} MHz, {points} points")

        # Calculate dynamic timeout based on points (approximately 0.1s per point + buffer)
        scan_timeout = max(self.timeout, points * 0.15 + 5)

        # Clear buffers
        self.serial.reset_input_buffer()

        # Send command
        cmd_bytes = (command + '\n').encode('ascii')
        self.serial.write(cmd_bytes)

        if self.debug:
            print(f"TX: {command}")
            print(f"Timeout set to {scan_timeout:.1f}s for {points} points")

        # Read response
        data = []
        start_time = time.time()
        line_count = 0
        last_data_time = time.time()
        idle_timeout = 3.0  # Timeout if no data received for 3 seconds

        while True:
            elapsed = time.time() - start_time
            idle_time = time.time() - last_data_time

            # Check total timeout
            if elapsed > scan_timeout:
                if self.debug:
                    print(f"Total timeout reached: received {line_count}/{points} lines in {elapsed:.1f}s")
                break

            # Check idle timeout (no data received recently)
            if idle_time > idle_timeout and line_count > 0:
                if self.debug:
                    print(f"Idle timeout: no data for {idle_time:.1f}s, received {line_count}/{points} lines")
                break

            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('ascii', errors='ignore').strip()

                if not line:
                    continue

                last_data_time = time.time()  # Reset idle timer

                if self.debug:
                    print(f"RX: {line}")

                # Skip echo of command
                if line.startswith('scan'):
                    continue

                # Check for prompt (end of data)
                if 'ch>' in line:
                    if self.debug:
                        print(f"Prompt detected, scan complete")
                    break

                # Parse data line
                # Format: <frequency> <s11_real> <s11_imag> <s21_real> <s21_imag>
                try:
                    parts = line.split()
                    if len(parts) >= 5:
                        freq = float(parts[0])
                        s11_real = float(parts[1])
                        s11_imag = float(parts[2])
                        s21_real = float(parts[3])
                        s21_imag = float(parts[4])

                        s11 = complex(s11_real, s11_imag)
                        s21 = complex(s21_real, s21_imag)

                        data.append((freq, s11, s21))
                        line_count += 1

                        # Check if we have all expected points
                        if line_count >= points:
                            if self.debug:
                                print(f"Received all {points} expected points")
                            break
                except (ValueError, IndexError) as e:
                    if self.debug:
                        print(f"Error parsing line: {line} - {e}")
                    continue
            else:
                time.sleep(0.01)

        if self.debug:
            print(f"Scan completed: received {len(data)}/{points} data points in {time.time() - start_time:.2f}s")

        # Warn if we didn't get all expected points
        if len(data) < points:
            print(f"Warning: Expected {points} points but received {len(data)} points")

        return data

    def scan_logarithmic(self, start_freq: int, stop_freq: int, points: int, outmask: int = 7) -> List[Tuple[float, complex, complex]]:
        """
        Perform logarithmic sweep scan

        NanoVNA-F v2 scan command only supports linear sweep, so this method
        performs multiple linear scans across logarithmically-spaced frequency
        bands and combines the results.

        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            points: Total number of sweep points (11-301)
            outmask: Output format mask (same as scan method)

        Returns:
            List of tuples: (frequency, S11_complex, S21_complex)
        """
        if points < 11 or points > 301:
            raise ValueError("Points must be between 11 and 301")

        if self.debug:
            print(f"Logarithmic scan: {start_freq/1e6:.1f} MHz to {stop_freq/1e6:.1f} MHz, {points} points")

        # Generate logarithmically-spaced frequency points
        log_start = np.log10(start_freq)
        log_stop = np.log10(stop_freq)
        log_freqs = np.logspace(log_start, log_stop, points)

        # Divide frequency range into bands for scanning
        # Use fewer bands for better performance while maintaining log distribution
        num_bands = min(3, max(3, points // 4))  # 3-10 bands depending on point count

        band_edges = np.logspace(log_start, log_stop, num_bands + 1)

        if self.debug:
            print(f"Using {num_bands} frequency bands for logarithmic scan")

        all_data = []

        for i in range(num_bands):
            band_start = int(band_edges[i])
            band_stop = int(band_edges[i + 1])

            # Calculate points for this band (proportional to log density)
            # Count how many target frequencies fall in this band
            band_points = np.sum((log_freqs >= band_start) & (log_freqs <= band_stop))
            band_points = max(11, min(band_points + 5, 101))  # At least 11, max 101 per band

            if self.debug:
                print(f"Band {i+1}/{num_bands}: {band_start/1e6:.3f}-{band_stop/1e6:.3f} MHz, {band_points} points")

            # Scan this band
            try:
                band_data = self.scan(band_start, band_stop, band_points, outmask)
                all_data.extend(band_data)
            except Exception as e:
                if self.debug:
                    print(f"Error scanning band {i+1}: {e}")
                continue

            # Small delay between bands
            time.sleep(0.05)

        if not all_data:
            if self.debug:
                print("No data collected from logarithmic scan")
            return []

        # Sort by frequency
        all_data.sort(key=lambda x: x[0])

        # Remove duplicates (keep first occurrence)
        unique_data = []
        last_freq = -1
        freq_tolerance = 0.01  # 1% tolerance for considering frequencies as duplicate

        for freq, s11, s21 in all_data:
            if last_freq < 0 or abs(freq - last_freq) / last_freq > freq_tolerance:
                unique_data.append((freq, s11, s21))
                last_freq = freq

        if self.debug:
            print(f"Logarithmic scan collected {len(all_data)} raw points, {len(unique_data)} unique points")

        # Interpolate to exact logarithmic frequencies if needed
        # This ensures the output matches the requested log-spaced points
        if len(unique_data) > points:
            # Downsample to requested number of points
            indices = np.linspace(0, len(unique_data) - 1, points, dtype=int)
            final_data = [unique_data[i] for i in indices]
        else:
            # Use all collected data
            final_data = unique_data

        if self.debug:
            print(f"Final logarithmic scan: {len(final_data)} points")

        return final_data

    def scan_multi_band(
        self,
        bands: List[Tuple[int, int, int]],
        outmask: int = 7,
        sweep_mode: str = "linear"
    ) -> List[Tuple[float, complex, complex]]:
        """
        Perform multi-band scan with different frequency ranges

        This allows scanning different frequency bands with specific point counts,
        useful for detailed analysis across wide frequency ranges.

        Args:
            bands: List of (start_freq, stop_freq, points) tuples for each band
                   Example: [(100e3, 1e6, 100), (1e6, 10e6, 100), ...]
            outmask: Output format mask (same as scan method)
            sweep_mode: "linear" or "logarithmic" sweep for each band

        Returns:
            Combined list of tuples: (frequency, S11_complex, S21_complex)
        """
        if not bands:
            raise ValueError("At least one band must be specified")

        if self.debug:
            print(f"Multi-band scan: {len(bands)} bands, {sweep_mode} sweep per band")
            total_points = sum(b[2] for b in bands)
            print(f"Total points: {total_points}")

        all_data = []
        scan_function = self.scan_logarithmic if sweep_mode == "logarithmic" else self.scan

        for i, (start_freq, stop_freq, points) in enumerate(bands):
            if self.debug:
                print(f"\nBand {i+1}/{len(bands)}: {start_freq/1e6:.3f} - {stop_freq/1e6:.3f} MHz, {points} points")

            try:
                # Scan this band
                band_data = scan_function(int(start_freq), int(stop_freq), points, outmask)

                if band_data:
                    all_data.extend(band_data)
                    if self.debug:
                        print(f"  Collected {len(band_data)} points from band {i+1}")
                else:
                    print(f"Warning: No data collected from band {i+1}")

            except Exception as e:
                print(f"Error scanning band {i+1}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue

            # Small delay between bands
            if i < len(bands) - 1:
                time.sleep(0.1)

        if not all_data:
            if self.debug:
                print("No data collected from multi-band scan")
            return []

        # Sort by frequency to ensure proper order
        all_data.sort(key=lambda x: x[0])

        # Remove duplicates at band boundaries (keep first occurrence)
        unique_data = []
        last_freq = -1
        freq_tolerance = 0.001  # 0.1% tolerance

        for freq, s11, s21 in all_data:
            if last_freq < 0 or abs(freq - last_freq) / max(last_freq, freq) > freq_tolerance:
                unique_data.append((freq, s11, s21))
                last_freq = freq

        if self.debug:
            print(f"\nMulti-band scan completed:")
            print(f"  Total raw points: {len(all_data)}")
            print(f"  Unique points: {len(unique_data)}")
            print(f"  Frequency range: {unique_data[0][0]/1e6:.3f} - {unique_data[-1][0]/1e6:.3f} MHz")

        return unique_data

    def get_frequencies(self) -> List[float]:
        """
        Get current sweep frequency list

        Returns:
            List of frequencies in Hz
        """
        response = self.send_command("frequencies")

        frequencies = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.endswith('ch>') and not line.startswith('frequencies'):
                try:
                    freq = float(line)
                    frequencies.append(freq)
                except ValueError:
                    continue

        return frequencies

    def calibration_load(self):
        """Start calibration: LOAD"""
        return self.send_command("cal load")

    def calibration_open(self):
        """Start calibration: OPEN"""
        return self.send_command("cal open")

    def calibration_short(self):
        """Start calibration: SHORT"""
        return self.send_command("cal short")

    def calibration_thru(self):
        """Start calibration: THRU"""
        return self.send_command("cal thru")

    def calibration_done(self):
        """Finish calibration"""
        return self.send_command("cal done")

    def calibration_on(self):
        """Enable calibration"""
        return self.send_command("cal on")

    def calibration_off(self):
        """Disable calibration"""
        return self.send_command("cal off")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


if __name__ == "__main__":
    # Test code
    print("NanoVNA-F v2 Controller Test")
    print("-" * 50)

    with NanoVNAController(debug=True) as vna:
        print("\nGetting version...")
        version = vna.get_version()
        print(f"Version: {version}")

        print("\nPerforming test scan...")
        start = 100_000_000  # 100 MHz
        stop = 200_000_000   # 200 MHz
        points = 51

        data = vna.scan(start, stop, points, outmask=7)

        print(f"\nReceived {len(data)} data points")
        if data:
            print("\nFirst 5 data points:")
            for i, (freq, s11, s21) in enumerate(data[:5]):
                print(f"  {i+1}. Freq: {freq/1e6:.2f} MHz, S11: {s11:.6f}, S21: {s21:.6f}")
