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

# Measurement Settings
st.sidebar.subheader("Measurement Settings")

# Measurement method
measurement_method = st.sidebar.selectbox(
    "Calculation Method",
    ["Shunt (S21)", "Reflection (S11)", "Series (S21)"],
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

                    # Scan based on sweep type
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
                    final_data = calc.calculate_average(all_measurements)
                else:
                    final_data = all_measurements[0]

                st.session_state.measurement_data = final_data
                st.session_state.sweep_type = sweep_type
                st.session_state.measurement_method = measurement_method

                status_text.text("Measurement complete!")
                progress_bar.empty()
                status_text.empty()

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

        # Impedance vs Frequency plot (log-log)
        st.markdown("#### Impedance vs Frequency")

        impedance_chart = alt.Chart(df).mark_line(point=True, color='steelblue').encode(
            x=alt.X('Frequency (Hz):Q',
                    scale=alt.Scale(type='log'),
                    axis=alt.Axis(title='Frequency (Hz)', format='~s', labelAngle=-45)),
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
                        axis=alt.Axis(title='Frequency (Hz)', format='~s', labelAngle=-45)),
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
