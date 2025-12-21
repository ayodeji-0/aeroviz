# File Structure


# Imports
import streamlit as st
import numpy as np

from styles.stylesheet import *

# Page Config
#st.set_page_config(layout="wide", page_icon="./icons/icon.ico"
st.set_page_config(page_title='AeroViz', layout = 'wide', page_icon = "./icons/icon.ico", initial_sidebar_state = 'auto')

# Apply page styling
apply_all_styles()

## Page Setup
cont_height = 150
col1, col2, col3 = st.columns([0.1, 3, 0.1]) # using columns to center the header text and image

with col2:
    # Header Text
    st.title("AeroViz")
    # Header Image
    st.image("./images/header_image.png", use_container_width = True)

  
    divider2()
    st.subheader("An interactive program to demonstrate aeromechanical phenomena in turbomachinery")

    ## Page Content
    with st.container(border = True):
        column_header5("Overview ")
        body("""
            AeroViz is an educational and interactive platform designed to help beginner and non-expert engineers visualize and understand aeromechanical phenomena in turbomachinery.
            Educational research has shown that interactive learning can be up to 60% more effective than traditional learning.
            The phenomena selection includes coupled bending-torsion flutter, analysis of tuned blisk dynamics, and linear cascade interactions.
            """)
        body("""Built for clarity and engagement, while prioritising interactivity and intuitivity, AeroViz uses mathematically accurate 
            computational solutions to mathematical models and key visualizations to explore the physics behind complex phenomena.
            Use the sidebar to begin exploring each module.""")

        column_header5("Motivational Objectives ")
        st.markdown("""
        - Make learning about aeroelasticity dynamic, fun, and accessible
        - Promote active learning 
        - Enable self-paced experimentation
        - Traditional resources are static and sometimes build on assumed background knowledge
        - Aeroelasticity is inherently dynamic, as such the learning experience should be too!
        - Built with beginners and students in mind
        - Backed by validated models and real research
        """)

        column_header5("Features")
        st.markdown("""
        - Customize parameters and simulate behavior in real-time
        - View brief animations and relevant plots
        - Conduct parametric studies
        - Read background theory alongside each demo
        - Ask AeroViz AI for guidance pulled straight from the research papers!
        """)

        column_header5("Stack")
        st.markdown("""
        - Python (NumPy, Plotly, etc.)
        - Deployed in Streamlit with Git-based version control
        - Utilizes OOP Frameworks with custom modules for PEP 8 compliance
        - SciPy for advanced numerical routines
        """)

        column_header5("About / Contact")
        st.markdown("""
        - Part of an ME4 Individual Research Project at Imperial College London
        - Developed by Ayodeji Adeniyi
        - Supervised by Dr. Sina Stapelfeldt , Department of Mechanical Engineering s.stapelfeldt@imperial.ac.uk
        """)

