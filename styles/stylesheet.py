"""Stylesheet for AeroViz application."""

import streamlit as st
from typing import Optional
import matplotlib.pyplot as plt
def set_page_style() -> None:
    """Set the main page styling."""
    st.markdown(
        """
        <style>
            .page-header {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                color: #000000;
                background-color: #999999;
            }
            .page-header2 {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                color: #000000;
                background-color: #999999;
            }
            .column-header { 
                text-align: center; 
                font-size: 20px; 
                font-weight: bold; 
                color: #000000; 
                background-color: #999999; 
                padding: 10px; 
                border-radius: 15px; 
            }
            .column-header2 { 
                text-align: center; 
                font-size: 15px; 
                font-weight: bold; 
                padding: 0px; 
                background-color: #999999; 
                border-radius: 15px; 
                color: #000000; 
            }
            .column-header3 { 
                text-align: center; 
                font-size: 15px; 
                font-weight: bold; 
                padding: 10px; 
                color:#ffffff; 
            }
            .column-header4 { 
                text-align: center; 
                font-size: 15px; 
                font-weight: bold; 
                padding: 10px; 
                color:#ffffff; 
                text-decoration: underline; 
            }
            .column-header5 { 
                text-align: left; 
                font-size: 28px; 
                font-weight: bold; 
                padding: 10px; 
                color:#ffffff; 
            }
            .body { 
                text-align: justify; 
                font-size: 16px; 
                color: #ffffff; 
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_tabs_style() -> None:
    """Set the tabs styling."""
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 5px;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_header_style() -> None:
    """Set the header styling."""
    st.markdown(
        """
        <style>
            .stAppHeader {
                background-color: rgba(255, 255, 255, 0.0);
                visibility: visible;
            }
            .block-container {
                padding-top: 0rem;
                padding-bottom: 0rem;
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_image_style() -> None:
    """Set the image styling."""
    st.markdown(
        """
        <style>
            div[data-testid="stImage"] {
                vertical-align: bottom;
                display: flex;
                align-items: flex-end;
            }
            div[data-testid="stImage"] > img {
                margin-bottom: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_divider_style() -> None:
    st.markdown("""
    <style>
        hr {
            border: none;
            border-top: 2px solid white;
            margin: -0.5rem 0 0.25rem 0;  /* top, right, bottom, left */
        }
    </style>
    """, unsafe_allow_html=True)

    #     # Style the divider with controlled spacing
    # st.markdown("""
    #     <style>
    #         .custom-divider {
    #             border: none;
    #             border-top: 2px solid white;
    #             margin: 0.2rem 0 0.3rem 0;  /* minimal spacing */
    #         }

    #         .icon-style {
    #             height: 80px;
    #             width: 80px;
    #             object-fit: contain;
    #             display: flex;
    #             align-items: center;
    #             justify-content: center;
    #         }

    #         .header-box {
    #             padding: 0rem 0.5rem 0rem 0.5rem;
    #         }
    #     </style>
    # """, unsafe_allow_html=True)

    # # # App Header Styling - page label vertical translation and background color
# st.markdown(
#     """
#         <style>
#                 .stAppHeader {
#                     background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
#                     visibility: visible;  /* Ensure the header is visible */
#                 }

#                .block-container {
#                     padding-top: 0rem; !important;
#                     padding-bottom: 0rem; !important;
#                     padding-left: 1.5rem;
#                     padding-right: 1.5rem;
#                 }
#         </style>
#         """,
#     unsafe_allow_html=True,
# )

def remove_decoration() -> None:
    st.markdown("""
        <style>
            [data-testid="stDecoration"] {
                background: #339999;
                border-radius: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

def global_plt_style() -> None:
        # Global Plot Styling
    plt.rcParams["text.color"] = "white"
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.labelcolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"


def apply_all_styles() -> None:
    """Apply all styles at once."""
    set_page_style()
    set_tabs_style()
    set_header_style()
    set_image_style()
    remove_decoration()
    global_plt_style()

# Functions to show custon components

def column_header(text: str) -> None:
    st.markdown(f"<div class='column-header'>{text}</div>", unsafe_allow_html=True)

def column_header2(text: str) -> None:
    st.markdown(f"<div class='column-header2'>{text}</div>", unsafe_allow_html=True)

def column_header3(text: str) -> None:
    st.markdown(f"<div class='column-header3'>{text}</div>", unsafe_allow_html=True)

def column_header4(text: str) -> None:
    st.markdown(f"<div class='column-header4'>{text}</div>", unsafe_allow_html=True) 

def column_header5(text: str) -> None:
    st.markdown(f"<div class='column-header5'>{text}</div>", unsafe_allow_html=True)

def body(text: str) -> None:
    st.markdown(f"<div class='body'>{text}</div>", unsafe_allow_html=True)

def image(path: str) -> None:
    st.markdown(f'<img src="{path}" class="icon-style">', unsafe_allow_html=True)

# Show a divider
def divider() -> None:
    st.markdown(
            """<div style="border-top: 2px solid white; margin-top: 1rem; margin-bottom: 1rem;"></div>""",
            unsafe_allow_html=True
        )

def divider2() -> None:
    st.markdown(
            """<div style="border-top: 2px dashed white; margin-top: 1rem; margin-bottom: 1rem;"></div>""",
            unsafe_allow_html=True
        )