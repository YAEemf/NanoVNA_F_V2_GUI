"""
NanoVNA-F v2 Impedance Measurement Application
Streamlit-based GUI for impedance measurement using NanoVNA-F v2
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import serial.tools.list_ports
from typing import Optional
import time

from nanovna_controller import NanoVNAController
from impedance_calculator import ImpedanceCalculator, ImpedanceData


# Page configuration
st.set_page_config(
    page_title="NanoVNA-F v2 Impedance Measurement",
    page_icon=":bar_chart:",
    layout="wide"
)

# Title
st.title("NanoVNA-F v2 Impedance Measurement System")
st.markdown("### Shunt-Through Method Impedance Analyzer")

# Initialize session state
if 'measurement_data' not in st.session_state:
    st.session_state.measurement_data = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'vna_controller' not in st.session_state:
    st.session_state.vna_controller = None
if 'sweep_type' not in st.session_state:
    st.session_state.sweep_type = "Linear"
if 'measurement_method' not in st.session_state:
    st.session_state.measurement_method = "Shunt (S21)"
if 'use_multi_band' not in st.session_state:
    st.session_state.use_multi_band = False
if 'band_configs' not in st.session_state:
    st.session_state.band_configs = []
if 'calibration_ids' not in st.session_state:
    st.session_state.calibration_ids = None
if 'use_calibration' not in st.session_state:
    st.session_state.use_calibration = False


# Sidebar - Configuration
st.sidebar.header("Configuration")

# COM Port Settings
st.sidebar.subheader("COM Port")

# List available COM ports
available_ports = [port.device for port in serial.tools.list_ports.comports()]
if not available_ports:
    available_ports = ["No ports found"]

port_option = st.sidebar.selectbox(
    "Select COM Port",
    ["Auto Detect"] + available_ports,
    help="Select COM port or use Auto Detect"
)

# Frequency Settings
st.sidebar.subheader("Frequency Settings")

freq_unit = st.sidebar.radio(
    "Frequency Unit",
    ["Hz", "kHz", "MHz", "GHz"],
    index=2,  # Default: MHz
    horizontal=True
)

# Unit conversion factors
unit_factors = {
    "Hz": 1,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9
}

freq_factor = unit_factors[freq_unit]

# Frequency range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_freq_input = st.number_input(
        f"Start ({freq_unit})",
        min_value=0.05,
        max_value=10000.0,
        value=0.1,
        step=10.0,
        format="%.2f"
    )
with col2:
    stop_freq_input = st.number_input(
        f"Stop ({freq_unit})",
        min_value=0.05,
        max_value=10000.0,
        value=1000.0,
        step=10.0,
        format="%.2f"
    )

# Convert to Hz
start_freq = int(start_freq_input * freq_factor)
stop_freq = int(stop_freq_input * freq_factor)

# Sweep points
sweep_points = st.sidebar.slider(
    "Sweep Points",
    min_value=101,
    max_value=301,
    value=301,
    step=1,
    help="Number of measurement points (101-301)"
)

# Sweep type
sweep_type = st.sidebar.radio(
    "Sweep Type",
    ["Linear", "Logarithmic"],
    help="Linear or logarithmic frequency sweep"
)

# Multi-Band Scan Settings
st.sidebar.subheader("Multi-Band Scan")

use_multi_band = st.sidebar.checkbox(
    "Enable Multi-Band Scan",
    value=True,
    help="Scan multiple frequency bands with specific point counts for each band"
)

if use_multi_band:
    st.sidebar.markdown("**Band Configuration**")

    # Number of bands
    num_bands = st.sidebar.number_input(
        "Number of Bands",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of frequency bands to scan"
    )

    # Points per band
    points_per_band = st.sidebar.number_input(
        "Points per Band",
        min_value=11,
        max_value=301,
        value=100,
        step=10,
        help="Number of measurement points for each band"
    )

    # Band sweep mode
    band_sweep_mode = st.sidebar.radio(
        "Band Sweep Mode",
        ["Linear", "Logarithmic"],
        help="Sweep mode for each individual band"
    )

    # Calibration settings
    use_calibration = st.sidebar.checkbox(
        "Apply Calibration per Band",
        value=True,
        help="Apply different calibration data for each frequency band"
    )

    # Preset configurations
    preset = st.sidebar.selectbox(
        "Preset Configuration",
        ["Custom", "100kHz-1GHz (4 bands)", "1MHz-1GHz (3 bands)", "10MHz-1GHz (2 bands)"],
        help="Select a preset band configuration"
    )

    # Band definitions based on preset
    if preset == "100kHz-1GHz (4 bands)":
        band_configs = [
            (0.1, 1, "100kHz - 1MHz"),
            (1, 10, "1MHz - 10MHz"),
            (10, 100, "10MHz - 100MHz"),
            (100, 1000, "100MHz - 1GHz")
        ]
        # Default calibration IDs for preset
        default_cal_ids = [0, 1, 2, 3]
    elif preset == "1MHz-1GHz (3 bands)":
        band_configs = [
            (1, 10, "1MHz - 10MHz"),
            (10, 100, "10MHz - 100MHz"),
            (100, 1000, "100MHz - 1GHz")
        ]
        default_cal_ids = [1, 2, 3]
    elif preset == "10MHz-1GHz (2 bands)":
        band_configs = [
            (10, 100, "10MHz - 100MHz"),
            (100, 1000, "100MHz - 1GHz")
        ]
        default_cal_ids = [2, 3]
    else:  # Custom
        band_configs = []
        default_cal_ids = []
        for i in range(num_bands):
            with st.sidebar.expander(f"Band {i+1}", expanded=(i == 0)):
                col1, col2 = st.columns(2)
                with col1:
                    band_start = st.number_input(
                        f"Start (MHz)",
                        min_value=0.05,
                        max_value=4500.0,
                        value=0.1 * (10 ** i) if i < 4 else 1000.0,
                        step=1.0,
                        format="%.2f",
                        key=f"band_{i}_start"
                    )
                with col2:
                    band_stop = st.number_input(
                        f"Stop (MHz)",
                        min_value=0.05,
                        max_value=4500.0,
                        value=0.1 * (10 ** (i+1)) if i < 3 else 1000.0,
                        step=1.0,
                        format="%.2f",
                        key=f"band_{i}_stop"
                    )

                # Calibration ID for this band
                if use_calibration:
                    cal_id = st.number_input(
                        "Calibration ID",
                        min_value=0,
                        max_value=6,
                        value=min(i, 6),
                        step=1,
                        key=f"band_{i}_cal_id",
                        help="Calibration slot (0-6) to apply for this band"
                    )
                    default_cal_ids.append(cal_id)
                else:
                    default_cal_ids.append(None)

                band_configs.append((band_start, band_stop, f"Band {i+1}"))

    # Get calibration IDs (only if calibration is enabled)
    if use_calibration:
        calibration_ids = default_cal_ids
    else:
        calibration_ids = None

    # Display band summary
    if preset != "Custom":
        st.sidebar.markdown("**Bands:**")
        for i, (start, stop, label) in enumerate(band_configs):
            if use_calibration and calibration_ids:
                st.sidebar.text(f"• {label}: {points_per_band} pts (Cal ID: {calibration_ids[i]})")
            else:
                st.sidebar.text(f"• {label}: {points_per_band} pts")
        total_points = len(band_configs) * points_per_band
        st.sidebar.info(f"Total: {total_points} points")

# Measurement Settings
st.sidebar.subheader("Measurement Settings")

# Measurement method
measurement_method = st.sidebar.selectbox(
    "Calculation Method",
    ["Reflection (S11)", "Shunt (S21)", "Series (S21)"],
    help="Method for impedance calculation"
)

# Characteristic impedance
z0 = st.sidebar.number_input(
    "Characteristic Impedance Z0 (Ω)",
    min_value=1.0,
    max_value=1000.0,
    value=50.0,
    step=1.0,
    format="%.1f",
    help="System characteristic impedance (typically 50Ω)"
)

# Averaging
average_count = st.sidebar.number_input(
    "Average Count (N)",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
    help="Number of measurements to average (1 = no averaging)"
)

# Averaging method (only shown if N > 1)
if average_count > 1:
    averaging_method = st.sidebar.selectbox(
        "Averaging Method",
        ["Mean", "Median", "Trimmed Mean", "Robust (MAD)"],
        index=3,  # Default: Robust
        help=(
            "Method for combining multiple measurements:\n\n"
            "• Mean: Simple arithmetic average (sensitive to outliers)\n\n"
            "• Median: Middle value (very robust, but may lose precision)\n\n"
            "• Trimmed Mean: Remove top/bottom 10% before averaging (balanced)\n\n"
            "• Robust (MAD): Remove outliers using MAD (Median Absolute Deviation), "
            "then average remaining values. **Recommended for low impedance (<1Ω) measurements**"
        )
    )

    # Show method description
    if averaging_method == "Mean":
        st.sidebar.caption("📊 Standard average - good for stable measurements")
    elif averaging_method == "Median":
        st.sidebar.caption("🛡️ Most robust - excellent for noisy data")
    elif averaging_method == "Trimmed Mean":
        st.sidebar.caption("⚖️ Balanced - removes 10% extreme values")
    else:  # Robust (MAD)
        st.sidebar.caption("🎯 Intelligent outlier rejection - best for low-Z measurements")
else:
    averaging_method = "Mean"  # Default when N=1

# Debug mode
debug_mode = st.sidebar.checkbox(
    "Debug Mode",
    value=False,
    help="Show raw data and debug information"
)

# Display Settings
st.sidebar.subheader("Display Settings")

show_phase = st.sidebar.checkbox(
    "Show Phase Plot",
    value=True,
    help="Display phase vs frequency plot"
)

show_smith = st.sidebar.checkbox(
    "Show Smith Chart",
    value=False,
    help="Display Smith chart (S11 only)"
)


# Main content area
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Control Panel")

    # Connection status
    if st.session_state.connected:
        st.success("Connected to NanoVNA")
    else:
        st.warning("Not connected")

    # Connect button
    if st.button("Connect to NanoVNA", width='stretch'):
        try:
            port = None if port_option == "Auto Detect" else port_option

            vna = NanoVNAController(port=port, debug=debug_mode)
            if vna.connect():
                st.session_state.vna_controller = vna
                st.session_state.connected = True

                # Get version
                version = vna.get_version()
                if debug_mode:
                    st.info(f"Device: {version}")

                st.success("Connected successfully!")
                st.rerun()
            else:
                st.error("Connection failed")
        except Exception as e:
            st.error(f"Error: {e}")

    # Disconnect button
    if st.button("Disconnect", width='stretch', disabled=not st.session_state.connected):
        if st.session_state.vna_controller:
            st.session_state.vna_controller.disconnect()
            st.session_state.vna_controller = None
            st.session_state.connected = False
            st.success("Disconnected")
            st.rerun()

    st.markdown("---")

    # Measurement button
    if st.button(
        "Start Measurement",
        width='stretch',
        disabled=not st.session_state.connected,
        type="primary"
    ):
        if st.session_state.vna_controller:
            try:
                vna = st.session_state.vna_controller
                calc = ImpedanceCalculator(z0=z0)

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Perform measurements with averaging
                all_measurements = []

                for i in range(average_count):
                    status_text.text(f"Measurement {i+1}/{average_count}...")
                    progress_bar.progress((i) / average_count)

                    # Scan based on mode
                    if use_multi_band:
                        # Multi-band scan
                        bands = []
                        for band_start_mhz, band_stop_mhz, _ in band_configs:
                            band_start_hz = int(band_start_mhz * 1e6)
                            band_stop_hz = int(band_stop_mhz * 1e6)
                            bands.append((band_start_hz, band_stop_hz, points_per_band))

                        sweep_mode = "logarithmic" if band_sweep_mode == "Logarithmic" else "linear"
                        data = vna.scan_multi_band(
                            bands,
                            outmask=7,
                            sweep_mode=sweep_mode,
                            calibration_ids=calibration_ids
                        )

                        if debug_mode:
                            cal_info = f", Calibration: {calibration_ids}" if calibration_ids else ""
                            st.info(f"Multi-band scan: {len(bands)} bands, {len(data)} total points{cal_info}")

                    else:
                        # Single sweep scan
                        if sweep_type == "Logarithmic":
                            data = vna.scan_logarithmic(start_freq, stop_freq, sweep_points, outmask=7)
                        else:  # Linear
                            data = vna.scan(start_freq, stop_freq, sweep_points, outmask=7)

                    if not data:
                        st.error("No data received from NanoVNA")
                        break

                    # Extract data
                    frequencies = np.array([d[0] for d in data])
                    s11 = np.array([d[1] for d in data])
                    s21 = np.array([d[2] for d in data])

                    # Calculate impedance based on selected method
                    if measurement_method == "Shunt (S21)":
                        impedance_data = calc.calculate_from_s21_shunt(frequencies, s21)
                    elif measurement_method == "Reflection (S11)":
                        impedance_data = calc.calculate_from_s11_reflection(frequencies, s11)
                    else:  # Series (S21)
                        impedance_data = calc.calculate_from_s21_series(frequencies, s21)

                    all_measurements.append(impedance_data)

                    # Small delay between measurements
                    if i < average_count - 1:
                        time.sleep(0.1)

                progress_bar.progress(1.0)
                status_text.text("Processing...")

                # Average measurements if needed
                if average_count > 1:
                    # Convert averaging method name to mode parameter
                    mode_map = {
                        "Mean": "mean",
                        "Median": "median",
                        "Trimmed Mean": "trimmed",
                        "Robust (MAD)": "robust"
                    }
                    mode = mode_map.get(averaging_method, "mean")

                    if debug_mode:
                        st.info(f"Averaging {average_count} measurements using {averaging_method} method")

                    final_data = calc.calculate_average(all_measurements, mode=mode)
                else:
                    final_data = all_measurements[0]

                st.session_state.measurement_data = final_data
                st.session_state.sweep_type = sweep_type
                st.session_state.measurement_method = measurement_method
                st.session_state.use_multi_band = use_multi_band
                if use_multi_band:
                    st.session_state.band_configs = band_configs
                    st.session_state.use_calibration = use_calibration
                    st.session_state.calibration_ids = calibration_ids

                status_text.text("Measurement complete!")
                progress_bar.empty()
                status_text.empty()

                if use_multi_band:
                    st.success(f"Measurement completed! ({len(final_data.frequencies)} points, Multi-band: {len(band_configs)} bands)")
                else:
                    st.success(f"Measurement completed! ({len(final_data.frequencies)} points, {sweep_type} sweep)")

            except Exception as e:
                st.error(f"Measurement error: {e}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())

    # Display measurement info
    if st.session_state.measurement_data:
        st.markdown("---")
        st.subheader("Measurement Info")

        data = st.session_state.measurement_data

        st.metric("Data Points", len(data.frequencies))
        st.metric("Frequency Range", f"{data.frequencies[0]/1e6:.2f} - {data.frequencies[-1]/1e6:.2f} MHz")
        st.metric("Avg Impedance", f"{np.mean(data.magnitudes):.2f} Ω")

        # Display sweep type and method
        if st.session_state.get('use_multi_band', False):
            st.text(f"Mode: Multi-Band Scan")
            if 'band_configs' in st.session_state and st.session_state.band_configs:
                st.text(f"Bands: {len(st.session_state.band_configs)}")

                # Show calibration status
                if st.session_state.get('use_calibration', False):
                    st.text(f"Calibration: Enabled")

                with st.expander("Band Details"):
                    for i, (start, stop, label) in enumerate(st.session_state.band_configs):
                        cal_info = ""
                        if (st.session_state.get('use_calibration', False) and
                            st.session_state.get('calibration_ids') and
                            i < len(st.session_state.calibration_ids)):
                            cal_id = st.session_state.calibration_ids[i]
                            cal_info = f" (Cal ID: {cal_id})" if cal_id is not None else ""
                        st.text(f"  {i+1}. {label}{cal_info}")
        else:
            if 'sweep_type' in st.session_state:
                st.text(f"Sweep: {st.session_state.sweep_type}")
        if 'measurement_method' in st.session_state:
            st.text(f"Method: {st.session_state.measurement_method}")


with col_right:
    st.subheader("Measurement Results")

    if st.session_state.measurement_data:
        data = st.session_state.measurement_data

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Frequency (Hz)': data.frequencies,
            'Impedance (Ω)': data.magnitudes,
            'Phase (deg)': data.phases
        })

        # Calculate decade tick values for frequency axis
        def get_decade_ticks(freq_min, freq_max):
            """
            Generate decade (10^n) tick values for logarithmic axis

            Args:
                freq_min: Minimum frequency in Hz
                freq_max: Maximum frequency in Hz

            Returns:
                List of decade values and their labels
            """
            import math

            # Calculate decade range
            min_decade = math.floor(math.log10(freq_min))
            max_decade = math.ceil(math.log10(freq_max))

            # Generate decade values (1, 10, 100, 1k, 10k, 100k, 1M, 10M, 100M, 1G, ...)
            decade_values = []
            decade_labels = []

            for decade in range(min_decade, max_decade + 1):
                value = 10 ** decade
                if freq_min <= value <= freq_max:
                    decade_values.append(value)

                    # Format label with SI prefix
                    if value >= 1e9:
                        label = f"{value/1e9:.0f}G"
                    elif value >= 1e6:
                        label = f"{value/1e6:.0f}M"
                    elif value >= 1e3:
                        label = f"{value/1e3:.0f}k"
                    else:
                        label = f"{value:.0f}"
                    decade_labels.append(label)

            return decade_values, decade_labels

        freq_min = data.frequencies.min()
        freq_max = data.frequencies.max()
        decade_values, decade_labels = get_decade_ticks(freq_min, freq_max)

        # Impedance vs Frequency plot (log-log)
        st.markdown("#### Impedance vs Frequency")

        impedance_chart = alt.Chart(df).mark_line(point=True, color='steelblue').encode(
            x=alt.X('Frequency (Hz):Q',
                    scale=alt.Scale(type='log'),
                    axis=alt.Axis(
                        title='Frequency',
                        values=decade_values,
                        labelExpr=f"datum.value >= 1e9 ? datum.value/1e9 + 'G' : datum.value >= 1e6 ? datum.value/1e6 + 'M' : datum.value >= 1e3 ? datum.value/1e3 + 'k' : datum.value",
                        labelAngle=-45,
                        tickCount=len(decade_values),
                        grid=True
                    )),
            y=alt.Y('Impedance (Ω):Q',
                    scale=alt.Scale(type='log'),
                    axis=alt.Axis(title='Impedance (Ω)', format='~s')),
            tooltip=[
                alt.Tooltip('Frequency (Hz):Q', format=',.0f'),
                alt.Tooltip('Impedance (Ω):Q', format=',.2f')
            ]
        ).properties(
            width=800,
            height=400
        ).interactive()

        st.altair_chart(impedance_chart, use_container_width=True)  # Note: Streamlit charts still use use_container_width

        # Phase vs Frequency plot
        if show_phase:
            st.markdown("#### Phase vs Frequency")

            phase_chart = alt.Chart(df).mark_line(point=True, color='coral').encode(
                x=alt.X('Frequency (Hz):Q',
                        scale=alt.Scale(type='log'),
                        axis=alt.Axis(
                            title='Frequency',
                            values=decade_values,
                            labelExpr=f"datum.value >= 1e9 ? datum.value/1e9 + 'G' : datum.value >= 1e6 ? datum.value/1e6 + 'M' : datum.value >= 1e3 ? datum.value/1e3 + 'k' : datum.value",
                            labelAngle=-45,
                            tickCount=len(decade_values),
                            grid=True
                        )),
                y=alt.Y('Phase (deg):Q',
                        axis=alt.Axis(title='Phase (degrees)')),
                tooltip=[
                    alt.Tooltip('Frequency (Hz):Q', format=',.0f'),
                    alt.Tooltip('Phase (deg):Q', format=',.2f')
                ]
            ).properties(
                width=800,
                height=300
            ).interactive()

            st.altair_chart(phase_chart, use_container_width=True)  # Note: Streamlit charts still use use_container_width

        # Smith Chart (for S11 only)
        if show_smith and measurement_method == "Reflection (S11)":
            st.markdown("#### Smith Chart")

            # Create Smith Chart data
            s11 = data.s11
            real_part = np.real(s11)
            imag_part = np.imag(s11)

            smith_df = pd.DataFrame({
                'Real': real_part,
                'Imag': imag_part,
                'Frequency (Hz)': data.frequencies
            })

            smith_chart = alt.Chart(smith_df).mark_point(filled=True, size=50).encode(
                x=alt.X('Real:Q', scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(title='Real(S11)')),
                y=alt.Y('Imag:Q', scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(title='Imag(S11)')),
                color=alt.Color('Frequency (Hz):Q', scale=alt.Scale(scheme='viridis')),
                tooltip=[
                    alt.Tooltip('Frequency (Hz):Q', format=',.0f'),
                    alt.Tooltip('Real:Q', format='.3f'),
                    alt.Tooltip('Imag:Q', format='.3f')
                ]
            ).properties(
                width=500,
                height=500
            ).interactive()

            # Add unit circle
            circle_data = pd.DataFrame({
                'angle': np.linspace(0, 2*np.pi, 100)
            })
            circle_data['x'] = np.cos(circle_data['angle'])
            circle_data['y'] = np.sin(circle_data['angle'])

            circle = alt.Chart(circle_data).mark_line(color='gray', strokeDash=[5, 5]).encode(
                x='x:Q',
                y='y:Q'
            )

            st.altair_chart(circle + smith_chart, use_container_width=True)  # Note: Streamlit charts still use use_container_width

        # Data table
        if debug_mode:
            st.markdown("#### Raw Data")
            st.dataframe(df, use_container_width=True)  # Note: Streamlit dataframe still uses use_container_width

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="impedance_data.csv",
                mime="text/csv"
            )

    else:
        st.info("No measurement data available. Connect to NanoVNA and start measurement.")


# Footer
st.markdown("---")
st.markdown(
    """
    **NanoVNA-F v2 Impedance Measurement System**
    - Shunt-Through Method
    - Log-Log Plot
    - Averaging Support
    """
)
