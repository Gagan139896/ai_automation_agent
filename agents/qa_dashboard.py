import streamlit as st
import requests

# Backend URL
API_URL = "http://127.0.0.1:8000/run-qa-flow"

st.set_page_config(page_title="Multi-Stack QA Agent", layout="wide", page_icon="🧪")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .report-container { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        font-family: 'Source Code Pro', monospace;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Multi-Stack Autonomous QA Agent")
st.markdown("Automate Requirements → Code Generation → Execution → Reporting")

# --- DYNAMIC DROPDOWN CONFIG ---
languages_map = {
    "Python": ["Pytest", "Behave", "Robot Framework"],
    "Java": ["TestNG", "Cucumber", "JUnit"],
    "C#": ["NUnit", "SpecFlow"]
}

# --- SIDEBAR: Security ---
with st.sidebar:
    st.header("🔑 Access Control")
    user_token = st.text_input("API Token", type="password", help="Enter your secret key")
    st.divider()
    st.info(
        f"🚀 **Current Stack:**\n{st.session_state.get('lang', 'Not Selected')} - {st.session_state.get('fw', 'Not Selected')}")
    st.divider()
    st.write("Built with CrewAI & FastAPI | 2026")

# --- MAIN UI ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📋 Test Configuration")

    selected_lang = st.selectbox("💻 Programming Language", options=list(languages_map.keys()))
    selected_framework = st.selectbox("⚙️ Framework", options=languages_map[selected_lang])

    # Store in session for sidebar visibility
    st.session_state['lang'] = selected_lang
    st.session_state['fw'] = selected_framework

    with st.form("qa_form"):
        reqs = st.text_area("Testing Requirements / Scenarios",
                            placeholder="Write your Gherkin or manual steps here...",
                            height=200)

        app_url = st.text_input("Application URL")
        u_name = st.text_input("Username")
        u_pass = st.text_input("Password", type="password")

        uploaded_file = st.file_uploader("Upload Locator Excel (Strict Mapping)", type=["xlsx", "csv"])

        st.divider()
        submit_btn = st.form_submit_button("🚀 Start Autonomous Lifecycle")

with col2:
    st.subheader("📊 Generated Artifacts & Execution")

    if submit_btn:
        if not user_token:
            st.error("⚠️ Please enter API Token in sidebar.")
        elif not reqs or not app_url:
            st.warning("⚠️ Requirements and URL are mandatory.")
        else:
            with st.spinner(f"Agent is generating {selected_framework} code and launching browser..."):
                try:
                    payload = {
                        "reqs": reqs,
                        "url": app_url,
                        "username": u_name,
                        "password": u_pass,
                        "language": selected_lang,
                        "framework": selected_framework
                    }

                    files = {}
                    if uploaded_file:
                        files["locator_file"] = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)

                    headers = {"access_token": user_token}

                    # Backend Request
                    response = requests.post(API_URL, data=payload, files=files, headers=headers, timeout=600)

                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Mission Accomplished for {selected_framework}!")

                        # --- DISPLAY FULL STRUCTURED REPORT ---
                        st.markdown("---")
                        report_content = data.get("report", "No report generated.")

                        with st.container():
                            st.markdown(f"### 📦 Full Project Structure: {selected_framework}")
                            st.markdown(report_content)  # Agent ka pura markdown yahan dikhega

                        # Option to download the report as a text file
                        st.download_button("📥 Download Generated Code/Report",
                                           data=report_content,
                                           file_name=f"{selected_framework}_mission_report.md")
                    else:
                        st.error(f"❌ Backend Error: {response.text}")

                except Exception as e:
                    st.error(f"⚠️ Connection Failed: {str(e)}")
    else:
        st.info("Results will be displayed here once the agent completes the mission.")

# --- FOOTER ---
st.divider()
st.caption(f"Language: {selected_lang} | Framework: {selected_framework} | System: CrewAI 2.0 Agentic Framework")