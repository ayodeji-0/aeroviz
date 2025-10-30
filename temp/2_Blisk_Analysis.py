import streamlit as st
from lpm.blisk import Blisk
from lpm.analysis import BliskAnalysis
from utils.blisks.constants import materials
from styles.stylesheet import apply_all_styles

# Page Config
st.set_page_config(page_title='AeroViz', layout='wide', page_icon="./icons/icon.ico", initial_sidebar_state='collapsed')
apply_all_styles()

# Session State Setup
if "blisk_obj" not in st.session_state:
    st.session_state.blisk_obj = None
if "show_preview" not in st.session_state:
    st.session_state.show_preview = True
if "blisk_params" not in st.session_state:
    st.session_state.blisk_params = None
if "sys_params" not in st.session_state:
    st.session_state.sys_params = None
if "ba" not in st.session_state:
    st.session_state.ba = None
if "plots" not in st.session_state:
    st.session_state.plots = {"Deformation": None}
if "show_theory" not in st.session_state:
    st.session_state.show_theory = True

st.title('Blisk Structural Analysis')

# Dynamic column layout based on theory visibility
if st.session_state.show_theory:
    col1, col2, col3 = st.columns([0.5, 0.75, 0.5])
    cont_height = 1000
else:
    col1, col2 = st.columns([0.33, 0.66])
    col3 = None
    cont_height = 1000

# Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header3">Configure Blisk Model & Properties</div>', unsafe_allow_html=True)
    cont1_3 = st.container(border=True, height=cont_height)
    with cont1_3:
        col1_tabs = st.tabs(["Design Blisk", "System Configuration", "Parametric Study", "Settings"])

        # Design Blisk Tab
        with col1_tabs[0]:
            # Define parameter options and their slider configs
            blisk_parameters = {
                'Material': (list(materials.keys()), 1, 'Material of the blisk'),
                'Number of Blades': (4, 40, 20, 1, 'Number of blades attached to the disk'),
                'Blade Length (m)': (0.05, 0.5, 0.1, 0.01, 'Length of each blade'),
                'Blade Width (m)': (0.005, 0.05, 0.0125, 0.001, 'Width of each blade'),
                'Blade Thickness (m)': (0.005, 0.05, 0.01, 0.001, 'Thickness of each blade'),
                'Disk Radius (m)': (0.1, 1.0, 0.5, 0.01, 'Radius of the disk'),
                'Disk Thickness (m)': (0.01, 0.2, 0.1, 0.005, 'Thickness of the disk'),
                'Blade Segments': (2, 20, 10, 1, 'Number of segments per blade'),
                'Disk Radial Segments': (2, 20, 10, 1, 'Number of radial segments in the disk'),
            }
            # Pills for parameter selection
            blisk_param = st.pills('Select Blisk Parameters', list(blisk_parameters.keys()), default=list(blisk_parameters.keys())[:3], selection_mode='multi', key='blisk_param_pills', help='Select the blisk parameters to configure. You can select multiple parameters to adjust their values simultaneously.')
            # Set defaults
            mat = blisk_parameters['Material'][0][blisk_parameters['Material'][1]]
            num_blades = blisk_parameters['Number of Blades'][2]
            blade_length = blisk_parameters['Blade Length (m)'][2]
            blade_width = blisk_parameters['Blade Width (m)'][2]
            blade_thickness = blisk_parameters['Blade Thickness (m)'][2]
            disk_radius = blisk_parameters['Disk Radius (m)'][2]
            disk_thickness = blisk_parameters['Disk Thickness (m)'][2]
            blade_segments = blisk_parameters['Blade Segments'][2]
            radial_segments = blisk_parameters['Disk Radial Segments'][2]
            # Show sliders for selected parameters
            if not blisk_param:
                st.info("Select parameter(s) to vary")
            else:
                for k in blisk_param:
                    if k == 'Material':
                        mat = st.selectbox('Material', blisk_parameters['Material'][0], index=blisk_parameters['Material'][1])
                    elif k == 'Number of Blades':
                        num_blades = st.slider('Number of Blades', *blisk_parameters['Number of Blades'][:4], help=blisk_parameters['Number of Blades'][4])
                    elif k == 'Blade Length (m)':
                        blade_length = st.slider('Blade Length (m)', *blisk_parameters['Blade Length (m)'][:4], help=blisk_parameters['Blade Length (m)'][4])
                    elif k == 'Blade Width (m)':
                        blade_width = st.slider('Blade Width (m)', *blisk_parameters['Blade Width (m)'][:4], help=blisk_parameters['Blade Width (m)'][4])
                    elif k == 'Blade Thickness (m)':
                        blade_thickness = st.slider('Blade Thickness (m)', *blisk_parameters['Blade Thickness (m)'][:4], help=blisk_parameters['Blade Thickness (m)'][4])
                    elif k == 'Disk Radius (m)':
                        disk_radius = st.slider('Disk Radius (m)', *blisk_parameters['Disk Radius (m)'][:4], help=blisk_parameters['Disk Radius (m)'][4])
                    elif k == 'Disk Thickness (m)':
                        disk_thickness = st.slider('Disk Thickness (m)', *blisk_parameters['Disk Thickness (m)'][:4], help=blisk_parameters['Disk Thickness (m)'][4])
                    elif k == 'Blade Segments':
                        blade_segments = st.slider('Blade Segments', *blisk_parameters['Blade Segments'][:4], help=blisk_parameters['Blade Segments'][4])
                    elif k == 'Disk Radial Segments':
                        radial_segments = st.slider('Disk Radial Segments', *blisk_parameters['Disk Radial Segments'][:4], help=blisk_parameters['Disk Radial Segments'][4])
            current_blisk = (mat, num_blades, blade_length, blade_width, blade_thickness, disk_radius, disk_thickness, blade_segments, radial_segments)
            if st.session_state.blisk_params is None or current_blisk != st.session_state.blisk_params:
                st.session_state.blisk_params = current_blisk
                if st.session_state.blisk_obj is not None:
                    st.info("Blisk parameters changed. Click 'Generate Blisk' to update.")
            show_preview = st.toggle('Show Preview', value=st.session_state.show_preview, key='blisk_preview_toggle')
            buton_1_0_0 = st.button('Generate Blisk', use_container_width=True)
            if buton_1_0_0:
                try:
                    st.session_state.blisk_obj = Blisk(material=mat, blade_thickness=blade_thickness, disk_thickness=disk_thickness, num_blades=num_blades, blade_length=blade_length, blade_width=blade_width, disk_radius=disk_radius, blade_segments=blade_segments, radial_segments=radial_segments)
                    st.session_state.blisk_obj.precompute_parameters()
                except Exception as e:
                    st.error(f"Failed to generate blisk: {str(e)}")
                    st.session_state.blisk_obj = None
            st.session_state.show_preview = show_preview
            if st.session_state.blisk_obj is not None and show_preview:
                st.markdown(f'<div class="column-header3">Blisk Preview</div>', unsafe_allow_html=True)
                st.pyplot(st.session_state.blisk_obj.plot(), use_container_width=True)

        # System Configuration Tab
        with col1_tabs[1]:
            st.markdown('Most system parameters are set by the blisk geometry and material. No additional configuration required.')
            if st.session_state.blisk_obj is not None:
                st.write(f"$\\mu$ (Mass Ratio): {st.session_state.blisk_obj.mu:.3f}")
                st.write(f"$\\kappa$ (Stiffness Ratio): {st.session_state.blisk_obj.kappa:.3f}")
            run_analysis = st.button('Analyse', use_container_width=True)
            show_results = st.toggle('Show Results', value=False, key='show_blisk_analysis_results_toggle')
            if run_analysis:
                try:
                    with st.spinner("Computing blisk response..."):
                        st.session_state.ba = BliskAnalysis(st.session_state.blisk_obj, time=2.0, intervals=200)
                        st.session_state.ba.compute_deformations()
                except Exception as e:
                    st.error(f"Blisk analysis failed: {str(e)}")
                    st.session_state.ba = None
            if show_results:
                results_container = st.container(border=True)
                with results_container:
                    if st.session_state.ba is not None:
                        st.markdown('<div class="column-header4">Key Numerical Results</div>', unsafe_allow_html=True)
                        st.write("Natural Frequencies (Hz):")
                        st.write(st.session_state.ba.natural_frequencies)
                        st.write("Eigenvalues:")
                        st.write(st.session_state.ba.eigenvalues)
                    else:
                        st.info('No results available. Please run analysis first!', icon="📊")

        # Parametric Study Tab
        with col1_tabs[2]:
            st.info("Parametric study is in development.")

        # Settings Tab
        with col1_tabs[3]:
            new_theory_state = st.toggle('Show Theory Column', value=st.session_state.show_theory, help="Toggle visibility of the theory column", key="blisk_theory_toggle_input")
            if new_theory_state != st.session_state.show_theory:
                st.session_state.show_theory = new_theory_state
                st.rerun()

# Results
with col2:
    st.markdown('<div class="column-header">Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header3">System Response Analysis</div>', unsafe_allow_html=True)
    cont2_1 = st.container(height=cont_height, border=True)
    with cont2_1:
        col2_tabs = st.tabs(["Deformation Plot", "Parametric Study", "Animation"])
        with col2_tabs[0]:
            col2_0_1, col2_0_2 = st.columns([0.7, 0.3])
            with col2_0_1:
                button_2_0_0 = st.button('Plot Deformations Against Time', use_container_width=True)
            with col2_0_2:
                button_2_0_1 = st.button('Reset Deformation Plot', use_container_width=True)
            if button_2_0_1:
                st.session_state.plots["Deformation"] = None
            buff = st.empty()
            if button_2_0_0:
                if st.session_state.ba is not None:
                    with buff:
                        st.info('Plotting...', icon="📊")
                    fig_def = st.session_state.ba.plot_deformations()
                    st.session_state.plots["Deformation"] = fig_def
                    buff.empty()
                else:
                    st.error('Run Analysis First!', icon="⚠️")
            if st.session_state.plots["Deformation"] is not None:
                buff.empty()
                st.pyplot(st.session_state.plots["Deformation"], use_container_width=True)
        with col2_tabs[1]:
            st.info("Parametric study visualization is in development.")
        with col2_tabs[2]:
            st.info("Animation is in development.")

# Theory
if st.session_state.show_theory and col3 is not None:
    with col3:
        st.markdown('<div class="column-header">Background</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-header3">Theory</div>', unsafe_allow_html=True)
        cont3_1 = st.container(height=cont_height, border=True)
        with cont3_1:
            col3_tabs = st.tabs(["Introduction", "Lumped Parameter Model Model", "Equations of Motion", "Modes", "Ask Aero"])

            # Introduction Tab
            with col3_tabs[0]:
                st.write("""
                A bladed disk (blisk) integrates blades and disk into a single component, eliminating joints and reducing weight.
                However, this monolithic structure also creates tightly coupled dynamic behavior that must be understood to avoid resonance or flutter.
                """)

            # Lumped Parameter Model Tab
            with col3_tabs[1]:
                st.image("<insert image>", caption="Lumped Parameter Model (LPM)", use_container_width=True)
                st.write(r"""
                The Lumped Parameter Model (LPM) provides a simplified yet insightful representation of blisk dynamics.

                Each sector of the blisk is modeled as a repeating unit comprising:
                - a single disk mass $m_d$ representing the disk segment,
                - a blade mass $m_b$ representing the flexible blade,
                - coupled by springs of stiffness $k_d$ (disk) and $k_b$ (blade),
                - with angular interactions governed by a coupling stiffness $k_c$.

                The angular phase shift between sectors is determined by the **nodal diameter** $n$.

                This idealized model captures the essential mode-splitting and wave propagation behavior observed in full blisks.
                """)

            # Equations of Motion Tab
            with col3_tabs[2]:
                st.markdown("""
                The equations of motion for a single sector (disk + blade) take the form of a 2×2 eigenproblem:
                """)
                st.latex(r"""
                \left( \mathbf{K}_n - \omega^2 \mathbf{M} \right) \mathbf{x} = \mathbf{0}
                """)
                st.markdown("""
                where $\\omega$ is the natural frequency and $\\mathbf{x}$ is the eigenvector. For each nodal diameter $n$, the stiffness matrix incorporates cyclic symmetry:
                """)
                st.latex(r"""
                \mathbf{K}_n =
                \begin{bmatrix}
                k_d + 4k_c \sin^2\left(\frac{n\pi}{N}\right) & -k_c \\
                -k_c & k_b
                \end{bmatrix}, \quad
                \mathbf{M} =
                \begin{bmatrix}
                m_d & 0 \\
                0 & m_b
                \end{bmatrix}
                """)
                st.markdown("""
                where $k_d$, $k_b$ are the disk and blade stiffnesses, $k_c$ is the coupling stiffness, and $m_d$, $m_b$ are the disk and blade masses. The nodal diameter $n$ controls how strongly sectors interact.

                """)

            # Modes
            with col3_tabs[3]:
                st.write("""
                Blisk modes fall into two categories:
                - **Disk-Dominated**: Low-n modes where the disk motion dominates, typically low frequency.
                - **Blade-Dominated**: High-n modes with high-frequency vibration localized to blades.

                Each mode can be classified by its **nodal diameter** $n$, representing the number of full sine waves in deformation around the circumference.
                """)

            # Ask Aero
            with col3_tabs[4]:
                st.info("Ask Aero is in development.")
