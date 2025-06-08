import streamlit as st
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from styles.stylesheet import apply_all_styles

# Apply global styles
apply_all_styles()

# Setup: Google Sheets API
# Define your API access scope and authorize with your credentials file
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# 🔁 Placeholder: path to your downloaded Google Service Account key
creds = ServiceAccountCredentials.from_json_keyfile_name("config/google_credentials.json", scope)
client = gspread.authorize(creds)

# 🔁 Placeholder: your Google Sheet name or URL
sheet = client.open("aeroviz_feedback").sheet1  # Select first sheet

# Page UI
st.title("Feedback Form")
st.write("""
We value your feedback!\n
Please share your thoughts and suggestions, 
or report any bugs/issues you encountered.
""")

# Form
with st.form("feedback_form"):
    # User information (optional)
    st.write("Your Information (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
    with col2:
        email = st.text_input("Email")


    # sanitize name and email inputs
    name = name.strip() or "Anonymous"
    email = email.strip() or "-"

    category = st.selectbox(
        "Feedback Type", 
        ["Feature Request", "Bug Report", "UI/UX Improvement", "General Feedback", "Other"]
    )
    satisfaction = st.slider("How would you rate your experience?", 1, 5, 3)
    feedback_text = st.text_area("Your Feedback", height=150, placeholder="Describe your feature request, bug found, UI issue, or other feedback here...")
                                 
    # Feature prioritization (if applicable)
    if category == "Feature Request":
        priority = st.select_slider(
            "How important is this feature to you?",
            options=["Nice to have", "Important", "Critical"]
        )
    else:
        priority = None

    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        if feedback_text:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create row to insert
            row = [timestamp, name, email, category, satisfaction, feedback_text, priority]

            # Append to Google Sheet
            sheet.append_row(row)

            st.success("✅ Feedback submitted successfully!")
            st.write("Thanks for your feedback:")
            st.json(row)
        else:
            st.error("Please enter your feedback before submitting.")

# Alternative contact information
st.markdown("---")
st.write("You can also send an email directly to: contactaeroviz@gmail.com")

# Display some FAQ or features in progress
st.markdown("### Frequently Asked Questions")
with st.expander('In Progress Hotfixes'):
    st.write("""
    - **Plot Latex Rendering**: Support for rendering LaTeX in plots using some javascript injection.
    """)
with st.expander('Incoming Features'):
    st.write("""
    - **Feature Request Tracking**: A public system to track the status of feature requests.
    - **RAG Integration**: Implementing a Retrieval-Augmented Generation (RAG) system to provide domain specific chatting. Space constraints hindering this in deployed version for now.
    - **Mistuning Effects**: Update blisk page to include effects of mistuning on structural response.
    """)
with st.expander("How is feedback used?"):
    st.write("""
    Your feedback is regularly reviewed by our team to prioritize new features, 
    fix bugs, and improve the overall user experience. We use semantic analysis 
    to identify common themes and requests across all feedback submissions.
    """)

with st.expander("Can I track the status of my feature request?"):
    st.write("""
    Currently, we don't have a public tracking system for feature requests. 
    However, major updates based on user feedback will be announced in our 
    release notes and changelog.
    """)