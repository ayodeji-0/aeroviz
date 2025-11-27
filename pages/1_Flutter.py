

import streamlit as st

# Import modules
from flutter.airfoil import Airfoil
from flutter.analysis import FlutterAnalysis
from flutter.parametric import ParametricStudy
from utils.flutter.constants import *
from styles.stylesheet import apply_all_styles

# Page Config
st.set_page_config(page_title='AeroViz', layout = 'wide', page_icon = "./icons/icon.ico", initial_sidebar_state = 'collapsed')
# Apply page styling
apply_all_styles()

# Page title
#st.title("Coupled Bending-Torsion Flutter")




## Session State Setup
if "airfoil_obj" not in st.session_state:
    st.session_state.airfoil_obj = None # Track changes in airfoil object

if "show_preview" not in st.session_state:
    st.session_state.show_preview = True # Track changes in preview state - airfoil

if "airfoil_params" not in st.session_state:
    st.session_state.airfoil_params = None  # Track changes in airfoil parameters

if "airfoil_param_values" not in st.session_state:
    # Store individual parameter values to persist between pill selections
    st.session_state.airfoil_param_values = {
        k: v[0][3] for k, v in airfoil_parameters.items()
    }

if "sys_params" not in st.session_state:
    st.session_state.sys_params = None  # To track changes in system config

if "sys_param_values" not in st.session_state:
    # Store individual parameter values to persist between pill selections
    st.session_state.sys_param_values = {
        k: v[1][3] for k, v in system_configuration.items()
    }

if "study_params" not in st.session_state:
    st.session_state.study_params = None  # To track changes in parametric study config

if "anim_properties" not in st.session_state:
    st.session_state.anim_properties = {
        "airfoil_color": "#000000",
        "transparency": 0.5, 
    }

properties = st.session_state.anim_properties  # Get properties at top level

if "fa" not in st.session_state:
    st.session_state.fa = None # To store the flutter analysis object

if "ps" not in st.session_state:
    st.session_state.ps = None # To store the parametric study object

if "plots" not in st.session_state:
    st.session_state.plots = {
                                "Displacement":None,
                                "Parametric Study":None,
                                "Incoming Feature":None,
                                "Animation":None                              
                              } # To store plots avoiding clearing when changing tabs

if "show_theory" not in st.session_state:
    st.session_state.show_theory = True  # To track whether to show theory section, defaults to True

## Page Setup
st.title('Coupled Bending Torsion Flutter')
# Dynamic column layout based on theory visibility
if st.session_state.show_theory:
    col1, col2, col3 = st.columns([0.5, 0.75, 0.5])
    cont_height = 1000
else:
    col1, col2 = st.columns([0.33, 0.66])
    col3 = None  # No theory column
    cont_height = 1000
## Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header3">Configure Section Model & Properties</div>', unsafe_allow_html=True)
    cont1_3 = st.container(border=True, height=cont_height)
    with cont1_3:
        col1_tabs = st.tabs(["Design Airfoil","System Configuration","Parametric Study", "Settings" ])
        
        # Design Airfoil Tab
        # with col1_tabs[0]:
    
        #     max_camber = st.slider('Max Camber', 0, 9, 0, 1)
        #     camber_position = st.slider('Camber Position', 0, 9, 0, 1)
        #     thickness = st.slider('Thickness', 0, 40, 12, 1)
        #     length = st.slider('Length', 0, 10, 1, 1)
        #     num_points = st.slider('Discretization', 10, 100, 100, 1)
        #     centrepos = st.slider('Centre Position', 0.0, 1.0, 0.5, 0.01)
        #     show_preview = st.checkbox('Show Preview', value=st.session_state.show_preview)

        #     buton_1_0_0 = st.button('Generate Airfoil', use_container_width=True)

        #     # Store current airfoil slider values as a tuple
        #     current_airfoil = (max_camber, camber_position, thickness, num_points, length, centrepos)

        #     # Set properties to default if not present
        #     properties = st.session_state.anim_properties

        #     # Check if any parameter has changed
        #     if st.session_state.airfoil_params is None or current_airfoil != st.session_state.airfoil_params:
        #         st.session_state.airfoil_obj = None  # Reset airfoil
        #         st.session_state.airfoil_params = current_airfoil  # Update stored parameters
                

        #     if buton_1_0_0:
        #         st.session_state.airfoil_obj = Airfoil(max_camber, camber_position, thickness, num_points, length, centrepos)
        #         st.session_state.airfoil_obj.generate_naca_airfoil4()

        #     # Persist the preview state
        #     st.session_state.show_preview = show_preview

        #     # Display the preview only if airfoil is not None and user wants to see it
        #     if st.session_state.airfoil_obj is not None and st.session_state.show_preview:
        #         st.markdown(f'<div class="column-header3">NACA {max_camber}{camber_position}{thickness} Preview</div>', unsafe_allow_html=True)
        #         #st.write(f"Airfoil {max_camber}{camber_position}{thickness}")
        #         st.pyplot(st.session_state.airfoil_obj.plot(color=properties['airfoil_color']), use_container_width=True)#find plotly equivalent for interactive plot
        with col1_tabs[0]:            
            airfoil_param = st.pills('Select Airfoil Parameters', airfoil_parameters.keys(), default=list(airfoil_parameters.keys())[:3], selection_mode='multi', key='airfoil_param_pills', help='Select the airfoil parameters to configure. You can select multiple parameters to adjust their values simultaneously.')
                    
            # Use the stored parameter values from session state
            max_camber = st.session_state.airfoil_param_values['Max Camber']
            camber_position = st.session_state.airfoil_param_values['Camber Position']
            thickness = st.session_state.airfoil_param_values['Thickness']
            num_points = st.session_state.airfoil_param_values['Discretization']
            length = st.session_state.airfoil_param_values['Length']
            centrepos = st.session_state.airfoil_param_values['Centre Position']
            
            # Create tuple from current values
            current_airfoil = (max_camber, camber_position, thickness, num_points, length, centrepos)

            # Check if any parameters are selected in the pills
            if not airfoil_param:
                st.info("Select parameter(s) to vary")
            else:
                # For each parameter that is selected in the pills
                for k, v in airfoil_parameters.items():
                    if k in airfoil_param:  # Check if this parameter is selected
                        # Use the stored value as the default instead of the original default
                        val = st.slider(k, v[0][0], v[0][1], st.session_state.airfoil_param_values[k], v[0][2], help=v[1], key=f"airfoil_{k}")
                        
                        # Store the value back to session state
                        st.session_state.airfoil_param_values[k] = val
                        
                        # Update the tuple at the right position
                        param_index = list(airfoil_parameters.keys()).index(k)
                        current_airfoil = current_airfoil[:param_index] + (val,) + current_airfoil[param_index+1:]  # Check if any parameter has changed
            
            if st.session_state.airfoil_params is None or current_airfoil != st.session_state.airfoil_params:
                st.session_state.airfoil_params = current_airfoil  # Just update parameters
                # Only reset if actually needed
                if st.session_state.airfoil_obj is not None:
                    st.info("Airfoil parameters changed. Click 'Generate Airfoil' to update.")
            
            # Preview and generation controls
            show_preview = st.toggle('Show Preview', value=st.session_state.show_preview, key='airfoil preview toggle')
            buton_1_0_0 = st.button('Generate Airfoil', use_container_width=True)

            # if buton_1_0_0:
            #     max_camber, camber_position, thickness, num_points, length, centrepos = current_airfoil
            #     st.session_state.airfoil_obj = Airfoil(max_camber, camber_position, thickness, num_points, length, centrepos)
            #     st.session_state.airfoil_obj.generate_naca_airfoil4()
            
            if buton_1_0_0:
                try:
                    max_camber, camber_position, thickness, num_points, length, centrepos = current_airfoil
                    st.session_state.airfoil_obj = Airfoil(max_camber, camber_position, thickness, num_points, length, centrepos)
                    st.session_state.airfoil_obj.generate_naca_airfoil4()
                except Exception as e:
                    st.error(f"Failed to generate airfoil: {str(e)}")
                    st.session_state.airfoil_obj = None

            # Display the preview only if airfoil is not None and user wants to see it
            if st.session_state.airfoil_obj is not None and show_preview:
                st.markdown(f'<div class="column-header3">NACA {round(max_camber)}{round(camber_position)}{round(thickness)} Preview</div>', unsafe_allow_html=True)
                st.pyplot(st.session_state.airfoil_obj.plot(color=properties['airfoil_color']), use_container_width=True)
                #st.plotly_chart(st.session_state.airfoil_obj.plotly_plot(color=properties['airfoil_color']), use_container_width=True)
        


        # System Configuration Tab
        with col1_tabs[1]:
            # Extract just the display names (first element of each value)
            param_names = [v[0] for v in system_configuration.values()]
    
            sys_param = st.pills('Select System Parameters', 
                        param_names,
                        selection_mode='multi', 
                        default=param_names[:3],
                        key='sys_param_pills', 
                        help='Select the system parameters to configure. You can select multiple parameters to adjust their values simultaneously.')
              # Use the stored parameter values from session state
            mu = st.session_state.sys_param_values['mu']
            sigma = st.session_state.sys_param_values['sigma'] 
            V = st.session_state.sys_param_values['V']
            a = st.session_state.sys_param_values['a']
            b = st.session_state.sys_param_values['b']
            e = st.session_state.sys_param_values['e']
            r = st.session_state.sys_param_values['r']
            w_theta = st.session_state.sys_param_values['w_theta']
            mode = mode_options['Steady - State Space']  # Default mode
            
            # Create the current params tuple for comparison
            current_sys_params = (mu, sigma, V, a, b, e, r, mode, w_theta)
            
            # Initialize sys_params if None
            if st.session_state.sys_params is None:
                st.session_state.sys_params = current_sys_params
                
            # Copy the current values from session state
            sys_params = st.session_state.sys_params

            # Check if any parameters are selected in the pills
            if not sys_param:
                st.info("Select parameter(s) to vary")
            else:
                # For each parameter that is selected in the pills
                for k, v in system_configuration.items():
                    # Check if this parameter's display name is in the selected pills
                    if v[0] in sys_param:
                        val = st.slider(v[0], v[1][0], v[1][1], st.session_state.sys_param_values[k], v[1][2], help=v[0], key=f"sys_{k}")
                        
                        # Store the value back to session state
                        st.session_state.sys_param_values[k] = val
                        
                        # Update the tuple at the right position
                        param_index = list(system_configuration.keys()).index(k)
                        sys_params = sys_params[:param_index] + (val,) + sys_params[param_index+1:]
            mode = mode_options[mode]  # This will be the exact string you need
            
            # Store system parameters as a tuple
            current_sys_params = (mu, sigma, V, a, b, e, r, mode, w_theta)

            # Check if any parameter has changed
            if st.session_state.sys_params is None or sys_params != st.session_state.sys_params:
                st.session_state.fa = None  # Reset analysis
                st.session_state.sys_params = sys_params # Update stored parameters


            
            run_analysis = st.button('Analyse', use_container_width=True)
            show_results = st.toggle('Show Results', value=False, key='show_analysis_results_toggle')

            # if run_analysis:
            #     st.session_state.fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
            #     st.session_state.fa.compute_response()
            if run_analysis:
                try:
                    with st.spinner("Computing flutter response..."):
                        st.session_state.fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
                        st.session_state.fa.compute_response()
                        st.info('System Configured')
                except Exception as e:
                    st.error(f"Flutter analysis failed: {str(e)}")
                    st.session_state.fa = None

            if show_results:
                results_container = st.container(border=True)
                with results_container:

                    if st.session_state.fa is not None:
                        st.markdown('<div class="column-header4">Key Numerical Results</div>', unsafe_allow_html=True)
                        st.write("Eigenvalues:")
                        st.write(st.session_state.fa.vals)
                        st.write("Damping Ratios:")
                        st.write(st.session_state.fa.zeta)
                        st.write("Frequencies:")
                        st.write(st.session_state.fa.omega)
                    else:
                        st.info('No results available. Please run analysis first!', icon="📊")

            #     mu = st.slider('Mass Ratio · $μ$', 0.1, 20.0, 0.1, 0.1, help="Mass per unit span to stiffness ratio)")
            #     sigma = st.slider('Frequency Ratio · $σ$', 0.1, 10.0, 0.1, 0.1)
            #     V = st.slider('Reduced Velocity · $V$', 0.1, 100.0, 0.1, 0.1)
            #     a = st.slider('Torsional Axis Location · $a$', 0.0, 1.0, 0.5, 0.01)
            #     b = st.slider('Semi-Chord Length · $b$', 0.0, 1.0, 0.5, 0.01)
            #     e = st.slider('Eccentricity · $e$', 0.0, 1.0, 0.5, 0.01)
            #     r = st.slider('Radius of Gyration · $r$', 0.0, 1.0, 0.5, 0.01)
            #     w_theta = st.slider('Torsional Vibration Frequency · $w_{\\theta}$', 0, 1000, 100, 1)
            #     #mode = st.selectbox('Aerodynamic Influence', [f'Steady - State Space', f'Quasi Steady - State Space'])
            #     #mode = st.selectbox('Aerodynamic', ['Steady', 'Quasi Steady'])
            #     mode_display = st.selectbox('Aerodynamic', list(mode_options.keys()))
            #     mode = mode_options[mode_display]  # This will be the exact string you need
                
            #     # Store system parameters as a tuple
            #     sys_params = (mu, sigma, V, a, b, e, r, mode, w_theta)

            #     # Check if any parameter has changed
            #     if st.session_state.sys_params is None or sys_params != st.session_state.sys_params:
            #         st.session_state.fa = None  # Reset analysis
            #         st.session_state.sys_params = sys_params # Update stored parameters


            # with col1_2:
            #     button_1_1_0 = st.button('Run Analysis', use_container_width=True)
            #     cont1_1_1 = st.container(border=True)
            #     with cont1_1_1:
            #         # Perform System Analysis
            #         st.markdown('<div class="column-header4">Key Numerical Results</div>', unsafe_allow_html=True)
            #         buff = st.empty()
            #         with buff:
            #             st.info('Results will be displayed here!', icon="📊")
            #     if button_1_1_0:
            #         with cont1_1_1:
            #             buff.empty()
            #             # Define the flutter analysis object
            #             st.session_state.fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
            #             # Compute the flutter response
            #             st.session_state.fa.compute_response()

                        
            #             with buff:
            #                 st.info('Loading Numerical Results...')#, icon="./icons/calculator.ico")
                        
            #             st.write("Eigenvalues:")
            #             st.write(st.session_state.fa.vals)
            #             st.write("Damping Ratios:")
            #             st.write(st.session_state.fa.zeta)
            #             st.write("Frequencies:")
            #             st.write(st.session_state.fa.omega)
            #             with buff:
            #                 st.info('Results Loaded!', icon="✅")
            #             buff.empty()

        # Parametric Study Tab
        with col1_tabs[2]:
            
            st.write("""
                    Investigate the effects of varying system configuration parameters on response characteristics. \n
                    Select the parameter to vary, input the desired range and step size. \n
                    Select the dependent variable(s) of interest.
                    Note: To run the study you must first configure the system.
                    """)

            study_param_y = st.pills('Select Dependent Variable(s)*', list(ps_dep_dict.keys()),selection_mode='multi',default = list(ps_dep_dict.keys())[:3] , help=help_text['dependent_variable'])

            #st.markdown('<div class="body">Perform a parametric study to investigate the effect of varying system parameters on flutter characteristics. Select the parameters to vary and fix, along with the range and step size for each parameter. The study will generate a plot showing the variation in flutter speed with the selected parameters.</div>', unsafe_allow_html=True)
            study_param_x = st.pills('Select Parameter to Vary', ps_indep_dict, selection_mode='single', key='study_param_x', help=help_text['independent_variable'])
            step = st.number_input('Step Size', 0.1, 100.0, 0.1, 0.1)
            min_val, max_val = st.slider("Select range for independent variable", min_value=0.0, max_value=5.0, value=(0.1, 2.0), step=step)
                
            

            # with col1_2:
                
            button_1_3_0 = st.button('Run Study', use_container_width=True)
            cont1_1_2 = st.container(border=True)

            # if button_1_3_0:
            #     st.session_state.ps = ParametricStudy(study_param_x, min_val, max_val, step, study_param_y)
            #     st.session_state.ps.run_study(sys_params=sys_params)

            if button_1_3_0:
                if not study_param_x:
                    st.error("Please select a parameter to vary.")
                elif not study_param_y:
                    st.error("Please select at least one dependent variable.")
                elif min_val >= max_val:
                    st.error("Maximum value must be greater than minimum value.")
                else:
                    try:
                        with st.spinner("Running parametric study..."):
                            st.session_state.ps = ParametricStudy(study_param_x, min_val, max_val, step, study_param_y)
                            st.session_state.ps.run_study(sys_params=sys_params)
                            st.info('Parametric Study Completed')
                    except Exception as e:
                        st.error(f"Parametric study failed: {str(e)}")
                        st.session_state.ps = None
            ##No results to show for now
            # show_study_results = st.toggle('Show Results', value=False, key='show_study_results_toggle')
            # # Display results if available and requested
            # if show_study_results:
            #     if st.session_state.ps is not None and st.session_state.ps.results is not None:
            #         study_param_y = st.session_state.ps.results
            #     else:
            #         st.info('No results available. Please run parametric study first!', icon="📊")
            
        # Settings Tab
        with col1_tabs[3]:
            # Theory toggle - use a callback approach
            new_theory_state = st.toggle(
                'Show Theory Column', 
                value=st.session_state.show_theory,
                help="Toggle visibility of the theory column",
                key="theory_toggle_input"
            )
            
            # Only update if actually changed to avoid unnecessary reruns
            if new_theory_state != st.session_state.show_theory:
                st.session_state.show_theory = new_theory_state
                st.rerun()  # Force a complete rerun with the new state
        
            #st.markdown('<div class="column-header2">Aesthetics</div>', unsafe_allow_html=True)
            # Base animation properties - aesthetics
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                properties['airfoil_color'] = st.color_picker('Airfoil Color', '#339999')
            with col1_2:
                properties['annotated_text_color'] = st.color_picker('Annotation Color', '#000000')
            
            # Number of modes to show
            num_modes = st.slider('Number of Modes to Show', 1, 4, 1, 1, help="Select the number of modes to display in the animation.")

            properties['transparency'] = st.slider('Airfoil Transparency', 0.0, 1.0, 0.5, 0.01)
            properties['show_chord'] = st.toggle('Show Chord', value=True)
            properties['angled_text'] = st.toggle('Show Angled Text', value=True)


            # Store the updated properties
            st.session_state.anim_properties = properties
            #st.markdown('<div class="column-header2">Playback</div>', unsafe_allow_html=True)
            # Other animation properties - playback
            duration = st.slider('Duration · s', 1, 10, 5, 1)
            fps = st.slider('Frame Rate · fps', 0, 120, 30, 10)
            #st.write(f"Frame Count: {int(duration * fps)}")


## Column 2 Setup
cont2_width = 600

## Results
with col2:
    st.markdown('<div class="column-header">Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header3">System Response Analysis</div>', unsafe_allow_html=True)

    with col2:
        cont2_1 = st.container(height = cont_height, border=True)
        with cont2_1: 
            col2_tabs = st.tabs(["Displacement Plot", "Parametric Study","Incoming Feature", "Animation"])
            
            # # Displacement Plot Tab
            # with col2_tabs[0]:
            #     col2_0_1, col2_0_2 = st.columns([0.7,0.3])
            #     with col2_0_1:
            #         button_2_0_0 = st.button('Plot Displacements Against Time', use_container_width=True)#
            #     with col2_0_2:
            #         button_2_0_1 = st.button('Reset Displacement Plot', use_container_width=True)

            #     if button_2_0_0:
            #         buff = st.empty()
            #         with buff:
            #             st.info('Plotting...', icon="📊")
            #         if st.session_state.fa is not None:
            #             fig_disp = st.session_state.fa.plot_displacements(duration=duration, width=cont2_width, height=cont_height)
            #             st.session_state.plots["Displacement"] = fig_disp
            #             st.markdown('<div class="column-header3">Displacement Plot</div>', unsafe_allow_html=True)
            #         else:
            #             st.error('Go to System Configuration, then Run Analysis First!', icon="⚠️")
            #         with buff:
            #             st.info('Rendering Complete!', icon="✅")
                    
            #         if st.session_state.plots["Displacement"] is not None:
            #             buff.empty()
            #             st.pyplot(st.session_state.plots["Displacement"], use_container_width=True)
            #         else:
            #             st.error('No data to plot. Run Analysis First!', icon="⚠️")
            #     if button_2_0_1:
            #         st.session_state.plots["Displacement"] = None
            #         st.info("Plot reset", icon="🔄")

            #
            with col2_tabs[0]:
                col2_0_1, col2_0_2 = st.columns([0.7,0.3])
                with col2_0_1:
                    button_2_0_0 = st.button('Plot Displacements Against Time', use_container_width=True)
                with col2_0_2:
                    button_2_0_1 = st.button('Reset Displacement Plot', use_container_width=True)
                
                if button_2_0_1:
                    st.session_state.plots["Displacement"] = None

                buff = st.empty()
                # Display existing plot or generate new one
                if button_2_0_0:
                     
                    if st.session_state.fa is not None:
                        with buff:
                            st.info('Plotting...', icon="📊")
                        #fig_disp = st.session_state.fa.plot_displacements(duration=duration, width=cont2_width, height=cont_height)
                        fig_disp = st.session_state.fa.plotly_plot_displacements(duration=duration, width=cont2_width, height=cont_height)
                        st.session_state.plots["Displacement"] = fig_disp
                        buff.empty()
                    else:
                        st.error('Go to System Configuration, then Run Analysis First!', icon="⚠️")
                
                # Always display the plot if available
                if st.session_state.plots["Displacement"] is not None:
                    #st.markdown('<div class="column-header3">Displacement Plot</div>', unsafe_allow_html=True)
                    buff.empty()  # Clear the buffer before displaying the plot
                    #st.pyplot(st.session_state.plots["Displacement"], use_container_width=True)
                    st.plotly_chart(st.session_state.plots["Displacement"], use_container_width=True)
        
            # Parametric Study Tab
            # with col2_tabs[1]:
            #     col2_1_1, col2_1_2 = st.columns([0.7,0.3])
            #     with col2_1_1:
            #         button_2_1_0 = st.button(f'Plot Chosen Variables Against {study_param_x}', use_container_width=True)#
            #     with col2_1_2:
            #         button_2_1_1 = st.button('Reset Parametric Plot', use_container_width=True)
            
            #     if button_2_1_0:
            #         buff = st.empty()
            #         if st.session_state.ps is not None and st.session_state.ps.plot() is not None:
            #             fig_ps = st.session_state.ps.plot()
            #             st.session_state.plots["Parametric Study"] = fig_ps
            #             st.markdown('<div class="column-header3">Parametric Study Plot</div>', unsafe_allow_html=True)
                     
            #             st.pyplot(fig_ps)
            #         else:
            #             st.error('Results are not available. Run Parametric Study First!', icon="⚠️")
            with col2_tabs[1]:
                col2_1_1, col2_1_2 = st.columns([0.7,0.3])
                with col2_1_1:
                    button_2_1_0 = st.button(f'Plot Chosen Variables Against {study_param_x}', use_container_width=True)
                with col2_1_2:
                    button_2_1_1 = st.button('Reset Parametric Plot', use_container_width=True)
                
                if button_2_1_1:
                    st.session_state.plots["Parametric Study"] = None
                    st.info("Parametric plot has been reset.")
                
                buff = st.empty()
                # Generate new plot if requested
                if button_2_1_0:
                    with buff:
                        st.info('Generating parametric plot...', icon="📊")
                    if st.session_state.ps is not None:
                        try:
                            fig_ps = st.session_state.ps.plotly_plot()
                            if fig_ps is not None:
                                st.session_state.plots["Parametric Study"] = fig_ps
                                buff.empty()
                            else:
                                buff.error('Plot generation failed. Check your parameter ranges.', icon="⚠️")
                        except Exception as e:
                            buff.error(f'Error generating plot: {str(e)}', icon="❌")
                    else:
                        buff.error('Run Parametric Study first!', icon="⚠️")
                
                # Always display plot if available
                if st.session_state.plots["Parametric Study"] is not None:
                    #st.markdown('<div class="column-header3">Parametric Study Plot</div>', unsafe_allow_html=True)
                    #st.pyplot(st.session_state.plots["Parametric Study"], use_container_width=True)
                    st.plotly_chart(st.session_state.plots["Parametric Study"], use_container_width=True)

            # # Incoming Feature Tab
            # with col2_tabs[2]:
            #     col2_2_1, col2_2_2 = st.columns([0.7,0.3])
            #     with col2_2_1:
            #         button_2_2_0 = st.button('Plot Incoming Feature', use_container_width=True)
            #     with col2_2_2:
            #         button_2_2_1 = st.button('Reset Incoming Feature', use_container_width=True)
            #     if button_2_2_0:
            #         st.info('This feature is coming soon!', icon="🚀")

            #     if button_2_2_1:
            #         st.session_state.plots["Incoming Feature"] = None

            with col2_tabs[2]:
                col2_2_1, col2_2_2 = st.columns([0.7,0.3])
                with col2_2_1:
                    button_2_2_0 = st.button('Plot Incoming Feature', use_container_width=True)
                with col2_2_2:
                    button_2_2_1 = st.button('Reset Incoming Feature', use_container_width=True)
                
                if button_2_2_1:
                    st.session_state.plots["Incoming Feature"] = None
                    st.info("Feature plot has been reset.")
                
                buff = st.empty()
                if button_2_2_0:
                    with buff:
                        st.info('This feature is in development!', icon="🚀")
                
                # Display existing plot if available (will be useful when feature is implemented)
                if st.session_state.plots["Incoming Feature"] is not None:
                    st.markdown('<div class="column-header3">Feature Plot</div>', unsafe_allow_html=True)
                    st.pyplot(st.session_state.plots["Incoming Feature"], use_container_width=True)

            # # Animation Tab
            # with col2_tabs[3]:
            #         button_2_3_0 = st.button('Animate Displacements', use_container_width=True)
            #         if button_2_3_0:
            #             # Quick Test against model workbook
            #             # mu = 0.1
            #             # sigma = 0.1
            #             # V = 1
            #             # a = 0.5
            #             # b = 0.5
            #             # e = 0.25
            #             # r = 0.1
            #             # mode = 'Steady - State Space'

            #             # fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
            #             # fa.compute_response()
            #             # st.components.v1.html(fa.animate_flutter(st.session_state.airfoil_obj.coords, duration, fps, properties), width = cont2_width,height =cont_height, scrolling=True)

            #             # st.write(f"Damping Ratios: {fa.zeta}")
            #             # st.write(f"Frequencies: {fa.omega}")

            #             buff = st.empty()
            #             with buff:
            #                 st.info('Animating...', icon="🔍")
            #             if st.session_state.fa and st.session_state.airfoil_obj is not None:
            #                 # Create animation if analysis is complete and airfoil is generated
            #                 anim = st.session_state.fa.animate_flutter(st.session_state.airfoil_obj.coords, duration, fps, properties=properties)
            #                 st.markdown('<div class="column-header3">Animation</div>', unsafe_allow_html=True)
            #                 #st.components.v1.html(anim.to_html5_video(), width=800, height=600, scrolling=False)
            #                 st.components.v1.html(anim, width = cont2_width,height =cont_height, scrolling=True)#, width=800, height=600)
            #             else:
            #                 st.error('Generate an Airfoil then Run Analysis First!', icon="⚠️")
            #             with buff:
            #                 st.info('Rendering Complete!', icon="✅")
            #             buff.empty()

            with col2_tabs[3]:
                col2_3_1, col2_3_2 = st.columns([0.7,0.3])
                with col2_3_1:
                    button_2_3_0 = st.button('Animate Displacements', use_container_width=True)
                with col2_3_2:
                    button_2_3_1 = st.button('Reset Animation', use_container_width=True)
                
                if button_2_3_1:
                    st.session_state.plots["Animation"] = None
                    st.info("Animation has been reset.")
                
                buff = st.empty()
                # Generate new animation if requested
                if button_2_3_0:
                    with buff:
                        st.info('Animating...', icon="🔍")
                    
                    if st.session_state.fa is not None and st.session_state.airfoil_obj is not None:
                        try:
                            # Create animation if analysis is complete and airfoil is generated
                            anim = st.session_state.fa.animate_flutter(st.session_state.airfoil_obj.coords, duration, fps, properties=properties, n_modes=num_modes)
                            st.session_state.plots["Animation"] = anim
                            buff.empty()
                        except Exception as e:
                            buff.error(f'Animation generation failed: {str(e)}', icon="❌")
                    else:
                        if st.session_state.fa is None:
                            buff.error('Please run analysis first!', icon="⚠️")
                        elif st.session_state.airfoil_obj is None:
                            buff.error('Please generate an airfoil first!', icon="⚠️")
                        else:
                            buff.error('Both airfoil and analysis are required!', icon="⚠️")
                
                # Display animation if available
                if st.session_state.plots["Animation"] is not None:
                    st.markdown('<div class="column-header3">Animation</div>', unsafe_allow_html=True)
                    st.components.v1.html(st.session_state.plots["Animation"], width=cont2_width, height=cont_height, scrolling=True)
                    #rst.pyplot(anim, use_container_width=True)

                #st.session_state.fa.debug_static_foil(st.session_state.airfoil_obj.coords)
## Theory
if st.session_state.show_theory and col3 is not None:
    with col3:
        st.markdown('<div class="column-header">Background</div>', unsafe_allow_html=True)

        st.markdown('<div class="column-header3">Theory</div>', unsafe_allow_html=True)
        
        cont3_1 = st.container(height=cont_height,border=True)
        with cont3_1:
            col3_tabs = st.tabs(["Introduction", "Typical Section Model", "Equations of Motion", "Loading Cases", "Eigenproblem", "Ask Aero"]) #new structure post report
        # Introduction Tab
        with col3_tabs[0]:
                #st.markdown('<div class ="column-header3">Aeroelastic Formulation</div>', unsafe_allow_html=True)
                st.write("""
                        Flutter is a dynamic aeroelastic instability caused by the interaction of aerodynamic, elastic, and inertial forces.
                        Unlike other oscillatory phenomena such as buffeting or gust response, flutter is self-excited: it does not require external periodic forcing. \n
                        Instead, small disturbances can be amplified due to a feedback loop between unsteady aerodynamic loading and structural motion. \n
                        This makes flutter particularly important to detect and mitigate during early design.
                        """)


        # Typical Section Model Tab
        with col3_tabs[1]:
            #st.markdown('<div class="column-header2">Aeroelastic Formulation</div>', unsafe_allow_html=True)
            st.image("./images/typical_section_model.png", caption="Typical Section Model", use_container_width=True)
            st.write("""
                        A widely used approach to study flutter is through the "Typical Section Model", which represents a cross-section of a wing or rotor blade. This model is simplified yet retains key flutter characteristics found in complex systems. It consists of an airfoil section elastically mounted on springs, with two degrees of freedom: \n
                        1. Plunge motion $h$ – Up and down movement.
                        2. Twist motion $θ$ – Rotation around a reference axis. \n
                        The springs represent bending $k_{h}$ and torsional stiffness $k_{\\theta}$ of the structure. The aerodynamic forces act at the aerodynamic center $x_{AC}$, while the mass is centered at $x_{CM}$. The flutter instability arises when these two motions couple under aerodynamic forces, leading to exponential growth in oscillations.
                        """)

        # Equations of Motion Tab
        with col3_tabs[2]:
            st.markdown("The equations of motion for the typical section model are derived from the Euler-Bernoulli beam theory. The coupled bending–torsion dynamics are expressed in matrix form as:")

            st.latex(r"""
            \begin{bmatrix}
            m & m b x_\theta \\
            m b x_\theta & I_\theta
            \end{bmatrix}
            \begin{bmatrix}
            \ddot{h} \\
            \ddot{\theta}
            \end{bmatrix}
            +
            \begin{bmatrix}
            k_h & 0 \\
            0 & k_\theta
            \end{bmatrix}
            \begin{bmatrix}
            h \\
            \theta
            \end{bmatrix}
            =
            \begin{bmatrix}
            F_h \\
            F_\theta
            \end{bmatrix}
            """)

            st.markdown("In Non-Dimensional Form (using $\\tilde{h} = h / b$, scaled by $\\omega_\\theta$):")

            st.latex(r"""
            \begin{bmatrix}
            1 & \mu x_\theta \\
            \mu x_\theta & \sigma
            \end{bmatrix}
            \begin{bmatrix}
            \ddot{\tilde{h}} \\
            \ddot{\theta}
            \end{bmatrix}
            +
            \begin{bmatrix}
            1 & 0 \\
            0 & 1
            \end{bmatrix}
            \begin{bmatrix}
            \tilde{h} \\
            \theta
            \end{bmatrix}
            =
            \begin{bmatrix}
            \frac{F_h}{m b \omega_\theta^2} \\
            \frac{F_\theta}{I_\theta \omega_\theta^2}
            \end{bmatrix}
            """)

        # Loading Cases Tab
        with col3_tabs[3]:
            st.markdown(r"""
            The loading cases for the typical section model can be categorized into two main types:

            **1. Steady-State**  
            The aerodynamic forces are computed from instantaneous angles of attack, with no contribution from plunge velocity or acceleration. The lift and moment coefficients remain constant and do not change with frequency.  
            The forcing vector becomes:
            
            $$
            \mathbf{F}_{\text{steady}} =
            -2\pi \rho U^2 b
            \begin{bmatrix}
            1 \\
            -(1/2 + a)
            \end{bmatrix}
            \theta
            $$

            **2. Quasi-Steady**  
            The aerodynamic forces depend on both the instantaneous angle of attack and its derivatives (plunge velocity, angular acceleration, pitch rate).  
            Lift and moment are functions of velocity:
            
            $$
            L = \pi \rho U^2 b \left(\theta + \frac{\dot{h}}{U} + \frac{b\dot{\theta}}{U} \right)
            $$
            
            which results in a forcing vector of the form:

            $$
            \mathbf{F}_{\text{quasi}} =
            \mathbf{C}(U) \dot{\mathbf{q}} + \mathbf{K}_a(U) \mathbf{q}
            $$

            where $\mathbf{q} = \begin{bmatrix} h \\ \theta \end{bmatrix}$ and aerodynamic matrices depend on $U$.
            """, unsafe_allow_html=True)


        # Eigenproblem Tab
        with col3_tabs[4]:
            # equations = st.latex(r"""
            #                     \begin{bmatrix}
            #                     m & m b x_\theta \\
            #                     m b x_\theta & I_\theta
            #                     \end{bmatrix}
            #                     \begin{bmatrix}
            #                     \ddot{h} \\
            #                     \ddot{\theta}
            #                     \end{bmatrix}
            #                     +
            #                     \begin{bmatrix}
            #                     k_h & 0 \\
            #                     0   & k_\theta
            #                     \end{bmatrix}
            #                     \begin{bmatrix}
            #                     h \\
            #                     \theta
            #                     \end{bmatrix}
            #                     =
            #                     \begin{bmatrix}
            #                     F_h \\
            #                     F_\theta
            #                     \end{bmatrix}
            #                     """),
            #                 st.write("In Non-Dimensional Form:")
            #                 st.latex(r"""
            #                         <incoming non-dimensional form>
            #                             """),
            #                 st.write("""
            #                         Where: \n
            #                         - $m$: mass per unit span  
            #                         - $I_{\\theta}$: mass moment of inertia  
            #                         - $k_{h}$,  $k_{\\theta}$: bending and torsional stiffness  
            #                         - $F_{h}$,  $F_{\\theta}$: aerodynamic forces in plunge and twist
            #                         """)

            #st.markdown('<div class="column-header2">Problem Definition</div>', unsafe_allow_html=True)

            st.markdown("The problem formulation in flutter analysis depends on the type of aerodynamic forces considered: \n")
            st.markdown("To analyse flutter, the system is recast as an eigenvalue problem by assuming harmonic motion of the form:")

            st.latex(r"""
            \mathbf{q}(t) = \mathbf{\tilde{q}} e^{\lambda t}
            \quad \Rightarrow \quad
            \ddot{\mathbf{q}} = \lambda^2 \mathbf{\tilde{q}}
            """)

            st.markdown("Substituting into the non-dimensional equations of motion yields:")

            st.latex(r"""
            \left( -\lambda^2 \mathbf{M} + \mathbf{K} + \mathbf{K}_a(U) + \lambda \mathbf{C}_a(U) \right) \mathbf{\tilde{q}} = 0
            """)

            st.markdown("This is a quadratic eigenvalue problem (QEP), where the eigenvalues $\\lambda$ represent complex growth rates. The real part indicates damping (stability), and the imaginary part corresponds to oscillation frequency. Flutter occurs when damping crosses zero:")

            st.latex(r"""
            \text{Re}(\lambda) = 0 \quad \Rightarrow \quad \text{onset of flutter}
            """)

            st.markdown("To solve numerically, the QEP is linearised into a first-order state-space system. The resulting eigenvalue spectrum reveals mode coalescence and identifies the critical flutter velocity.")

            
            if mode == 'Steady - State Space':
                st.markdown(""" 
                Considering steady aerodynamic forces, \n
            
                The aerodynamic forces are computed from instantaneous angles of attack.
                No contribution from plunge velocity or acceleration.
                The lift and moment coefficients remain constant and do not change with frequency.
                Equation of motion simplifies significantly, making it easier to solve for flutter speed.
                """)
            if mode == 'Quasi Steady - State Space':
                st.markdown(""" Considering quasi-steady aerodynamic forces, \n
                        
                The aerodynamic forces become functions of both the instantaneous angle of attack and its derivatives.
                Plunge velocity, angular acceleration, and pitch rate influence the aerodynamic loads.
                Requires solving a more complex coupled system, but provides a better approximation of real-world flutter behavior.
                """)
        
        # Ask AI Tab
        with col3_tabs[5]:
            # st.chat_message('Need some clarification?')
            # st.chat_input('Ask AeroViz AI a question')
            # st.write("Coming Soon!")
            st.title('Stay tuned!')