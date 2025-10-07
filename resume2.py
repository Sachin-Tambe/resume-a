import streamlit as st
import google.generativeai as genai
import json
import re
import io
import csv
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

# --- Page Configuration ---
st.set_page_config(page_title="AI Resume Tailor", page_icon="📄", layout="wide")

# --- Gemini AI Configuration ---
# It's recommended to use Streamlit secrets for API keys
# For example: GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_API_KEY = "AIzaSyBjp6fulLfDnraRxdyyx6AvavCKSj53vsY" # Replace with your actual key or use secrets
model = None
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
    else:
        st.warning("⚠️ Please add your Gemini API Key to proceed.", icon="🔑")
except Exception as e:
    st.error(f"⚠️ Failed to configure Gemini AI: {e}", icon="🚨")

# --- CSV Data Storage ---
CSV_FILE = "resume_data.csv"

def save_to_csv(data):
    """Saves the entire resume data object to a single-row CSV file."""
    flat_data = {key: json.dumps(value) for key, value in data.items()}
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        writer.writeheader()
        writer.writerow(flat_data)
    st.success("Resume saved to resume_data.csv!")

def load_from_csv():
    """Loads the resume data from the CSV file."""
    if not os.path.exists(CSV_FILE): return None
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            flat_data = next(reader)
            return {key: json.loads(value) for key, value in flat_data.items()}
    except (StopIteration, json.JSONDecodeError):
        return None

# --- PDF GENERATION WITH REPORTLAB (ATS-FRIENDLY & COMPACT) ---
def create_resume_pdf(data):
    """Generates a high-quality, single-page PDF using the user-specified format."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=0.3*inch, rightMargin=0.3*inch, topMargin=0.3*inch, bottomMargin=0.3*inch)
    story = []
    styles = getSampleStyleSheet()

    # --- Custom Styles based on the target image ---
    styles.add(ParagraphStyle(name='Name', alignment=TA_CENTER, fontSize=24, fontName='Helvetica-Bold', leading=26, textColor=HexColor("#000000")))
    styles.add(ParagraphStyle(name='JobTitlePDF', alignment=TA_CENTER, fontSize=12, fontName='Helvetica', spaceAfter=6, textColor=HexColor("#000000")))
    styles.add(ParagraphStyle(name='ContactPDF', alignment=TA_CENTER, fontSize=9, fontName='Helvetica', textColor=HexColor("#000000")))
    
    styles.add(ParagraphStyle(name='SectionTitlePDF', fontSize=11, fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=0, textColor=HexColor("#000000")))
    
    styles.add(ParagraphStyle(name='BodyPDF', parent=styles['Normal'], fontSize=9.5, leading=11, spaceAfter=1, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='BodyPDFLeft', parent=styles['BodyPDF'], alignment=0)) # Left align
    styles.add(ParagraphStyle(name='BodyPDFRight', parent=styles['BodyPDF'], alignment=2)) # Right align
    
    bullet_style = ParagraphStyle('bullet', parent=styles['BodyPDF'], firstLineIndent=0, leftIndent=18, bulletIndent=5, spaceAfter=1)
    bullet_style.bulletText = '•'
    
    skill_style = ParagraphStyle(name='SkillPDF', parent=styles['BodyPDF'], leftIndent=0, spaceAfter=0)

    # --- Header ---
    story.append(Paragraph(data.get('fullName', 'SACHIN TAMBE'), styles['Name']))
    story.append(Paragraph(data.get('role', 'Data Analyst | Aspiring Data Scientist'), styles['JobTitlePDF']))
    
    # --- Dynamic Contact Line for PDF ---
    linkedin = data.get('linkedin', {})
    github = data.get('github', {})
    kaggle = data.get('kaggle', {})
    portfolio = data.get('portfolio', {})
    
    contact_parts = [
        data.get('phone',''),
        f"<a href='mailto:{data.get('email','')}'><u>{data.get('email','')}</u></a>",
        f"<a href='{linkedin.get('url','')}'><u>{linkedin.get('name','LinkedIn')}</u></a>",
        f"<a href='{github.get('url','')}'><u>{github.get('name','GitHub')}</u></a>",
        f"<a href='{kaggle.get('url','')}'><u>{kaggle.get('name','Kaggle')}</u></a>",
        f"<a href='{portfolio.get('url','')}'><u>{portfolio.get('name','Portfolio')}</u></a>"
    ]
    contact_line = " &nbsp;&nbsp;•&nbsp;&nbsp; ".join(filter(None, contact_parts))
    story.append(Paragraph(contact_line, styles['ContactPDF']))
    story.append(Spacer(1, 0.15*inch))

    # --- Reusable Line ---
    line = Table([['']], colWidths=['100%'], style=[('LINEBELOW', (0,0), (-1,0), 0.5, HexColor("#000000"))])

    # --- Summary Section ---
    story.append(Paragraph("Summary", styles['SectionTitlePDF']))
    story.append(line)
    story.append(Paragraph(data.get('summary', ''), styles['BodyPDF']))

    # --- Skills Section ---
    story.append(Paragraph("Skills", styles['SectionTitlePDF']))
    story.append(line)
    
    tech_skills = data.get('technicalSkills', {})
    if isinstance(tech_skills, dict):
        for category, skills in tech_skills.items():
            story.append(Paragraph(f"<b>{category}:</b> {', '.join(skills)}", skill_style))

    soft_skills = data.get('softSkills', [])
    if soft_skills:
        story.append(Paragraph(f"<b>Core Competencies:</b> {', '.join(soft_skills)}", skill_style))

    # --- Experience Section ---
    story.append(Paragraph("Experience", styles['SectionTitlePDF']))
    story.append(line)
    for exp in data.get('experience', []):
        header_table_style = [('VALIGN', (0,0), (-1,-1), 'TOP')]
        header_table = Table([
            [Paragraph(f"<b>{exp.get('title', '')}</b>", styles['BodyPDFLeft']), Paragraph(exp.get('duration', ''), styles['BodyPDFRight'])]
        ], colWidths=['70%', '30%'], style=header_table_style)
        story.append(header_table)
        story.append(Paragraph(f"<i>{exp.get('company', '')} ({exp.get('location', '')})</i>", styles['BodyPDF']))
        
        task_items = []
        for task in exp.get('tasks', []):
            task_items.append(Paragraph(task, bullet_style))
        story.append(KeepTogether(task_items))
        story.append(Spacer(1, 4))

    # --- Projects Section ---
    story.append(Paragraph("Projects", styles['SectionTitlePDF']))
    story.append(line)
    for proj in data.get('projects', []):
        link = proj.get('link', '').strip()
        title_text = f"<a href='{link}'><u>{proj.get('title', '')}</u></a>" if link and link != "#" else f"<b>{proj.get('title', '')}</b>"
        header_table_style = [('VALIGN', (0,0), (-1,-1), 'TOP')]
        header_table = Table([[Paragraph(title_text, styles['BodyPDFLeft']), ""]], colWidths=['100%', '0%'], style=header_table_style)
        story.append(header_table)
        story.append(Paragraph(proj.get('description', ''), bullet_style))
        story.append(Spacer(1, 4))

    # --- Certifications Section ---
    certs = data.get('certifications', [])
    if certs:
        story.append(Paragraph("Certifications", styles['SectionTitlePDF']))
        story.append(line)
        cert_items = []
        for cert in certs:
            link = cert.get('link', '').strip()
            cert_text = f"<a href='{link}'><u>{cert.get('name', '')}</u></a> ({cert.get('issuer', '')})" if link and link != "#" else f"<b>{cert.get('name', '')}</b> ({cert.get('issuer', '')})"
            cert_items.append(Paragraph(cert_text, bullet_style))
        story.append(KeepTogether(cert_items))

    # --- Education Section ---
    story.append(Paragraph("Education", styles['SectionTitlePDF']))
    story.append(line)
    for edu in data.get('education', []):
        header_table_style = [('VALIGN', (0,0), (-1,-1), 'TOP')]
        header_table = Table([
            [Paragraph(f"<b>{edu.get('degree', '')}</b>", styles['BodyPDFLeft']), Paragraph(edu.get('duration', ''), styles['BodyPDFRight'])]
        ], colWidths=['70%', '30%'], style=header_table_style)
        story.append(header_table)
        story.append(Paragraph(edu.get('institution', ''), styles['BodyPDF']))
        
    doc.build(story)
    return buffer.getvalue()

# --- Default Resume Data ---
def get_default_resume():
    """Populates the default resume data for a professional with 6-12 months of experience."""
    return {
        "fullName": "SACHIN TAMBE",
        "role": "Data Analyst | Aspiring Data Scientist",
        "phone": "+91-9076398319",
        "email": "tambesachin347@gmail.com",
        "linkedin": {"name": "LinkedIn", "url": "https://linkedin.com/in/sachin-tambe"},
        "github": {"name": "GitHub", "url": "https://github.com/sachintambe"},
        "kaggle": {"name": "Kaggle", "url": "https://www.kaggle.com/sachintambe"},
        "portfolio": {"name": "Portfolio", "url": "https://www.example.com"},
        "location": "Mumbai, Maharashtra",
        "summary": "Data Analyst with hands-on experience in data visualization, reporting automation, and statistical analysis in a manufacturing environment. Skilled in Python, SQL, Power BI, and Excel to deliver actionable insights and support data-driven decisions. Experienced in data cleaning, exploratory data analysis (EDA), and supervised machine learning (Regression, Classification). Supported stakeholders by providing reports and interactive dashboards to track business performance.",
        "technicalSkills": {
            "Programming & Databases": ["Python (pandas, numpy, scikit-learn)", "SQL (MySQL, PostgreSQL, Stored Procedures)", "MongoDB (basic)", "Shell Scripting", "ETL Processes"],
            "BI & Visualization": ["Power BI", "Tableau", "Excel (Advanced PivotTables, Power Query)", "Streamlit", "Matplotlib", "Seaborn"],
            "Applied Machine Learning": ["Regression Models", "Classification Models", "Clustering Algorithms", "Time Series Forecasting (ARIMA)", "Feature Engineering", "Random Forest"],
            "Developer Tools & Cloud": ["Git", "Jupyter Notebook", "Docker (basic)", "AWS (S3, EC2, RDS)", "Azure (basic)"],
            "AI & Frameworks": ["Consuming REST APIs (e.g., OpenAI)", "Prompt Engineering (basic)", "Introduction to LangChain", "Hugging Face Transformers", "Vector Databases (e.g., Pinecone)"]
        },
        "softSkills": ["Problem Solving", "Communication", "Teamwork", "Critical Thinking", "Stakeholder Engagement", "Data Storytelling"],
        "experience": [{"title": "Data Analyst", "company": "Art in Art", "duration": "Jun 2024 - Present", "location": "Mumbai, India", "tasks": ["Established automated data pipelines using (Python, SQL, pandas, openpyxl), reducing manual reporting effort by 70%.", "Designed and deployed Power BI dashboards for production, inventory, and sales KPIs, reviewed weekly by senior management.", "Partnered with multiple departments to define KPIs, improving decision-making efficiency by 60%. Conducted SQL and Excel analysis to identify cost-saving opportunities and performance trends across operations."]}],
        "education": [{"degree": "B.Sc. in Information Technology", "institution": "Somaiya Vidyavihar University", "duration": "Graduation: 2025", "location": "Mumbai, Maharashtra"}],
        "projects": [{"title": "Customer Churn Analysis", "description": "Analyzed customer behavior data (10k+ records) using Python and SQL to identify churn risk patterns. Delivered insights through an interactive dashboard, enabling targeted retention strategies.", "link": "https://github.com/example/churn-analysis"}, {"title": "Fraud Detection Insights", "description": "Analyzed financial transactions dataset to detect anomalies using statistical methods and reporting frameworks. Helped risk teams monitor suspicious cases, reducing false negatives in fraud detection.", "link": "#"}, {"title": "Sales Forecast Dashboard", "description": "Designed a forecasting dashboard integrating ARIMA-based trend models using Python and MySQL. Delivered demand and inventory forecasts with ±5% accuracy, aiding supply planning.", "link": "#"}],
        "certifications": [{"name": "Python for Data Science, AI & Development", "issuer": "IBM, Coursera", "link": "https://www.coursera.org/professional-certificates/ibm-data-science"}, {"name": "Databases and SQL for Data Science with Python", "issuer": "IBM, Coursera", "link": "#"}, {"name": "Foundations of Data Science", "issuer": "IBM, Coursera", "link": "#"}, {"name": "Data Engineering Foundations", "issuer": "IBM, Coursera", "link": "#"}]
    }

# --- AI Generation Functions ---

def generate_with_gemini(prompt):
    """Generates content using the Gemini model and cleans up JSON markdown."""
    if not model:
        st.error("Gemini model is not configured. Please add your API key.")
        return None
    try:
        response = model.generate_content(prompt).text
        # Clean up markdown fences (```json ... ```) that the model might add
        # re.DOTALL ensures this works even if the JSON is multi-line
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response, flags=re.DOTALL).strip()
        return cleaned_response
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return None

def parse_ai_json_response(response_str, expected_type, section_name):
    """
    Safely parses a JSON string from the AI and validates its Python type.
    Returns the parsed data or None if parsing/validation fails.
    """
    if not response_str:
        st.error(f"AI Error: The response for '{section_name}' was empty.")
        return None
    try:
        data = json.loads(response_str)
        if not isinstance(data, expected_type):
            st.error(f"AI Error: Expected a {expected_type.__name__} for '{section_name}', but received a {type(data).__name__}.")
            st.code(response_str) # Show the faulty response for debugging
            return None
        return data
    except json.JSONDecodeError as e:
        st.error(f"AI failed to generate valid JSON for '{section_name}': {e}")
        st.code(response_str)
        return None

def generate_tailored_content(jd, current_experience, core_skills):
    # --- PROMPTS ---
    skills_prompt = f"""
    Analyze the following job description for a Data Analyst/Scientist.
    Job Description: {jd}
    Your task is to generate a comprehensive, categorized list of technical skills in a JSON object format.
    Start with this baseline of essential skills: {json.dumps(core_skills)}
    Integrate relevant skills from the job description into the appropriate categories. If a new category is needed, create it.
    The final JSON object should be a complete, well-organized representation of skills for this specific job, built upon the essential baseline.
    The top-level key in your response must be "technicalSkills".
    Respond with only the raw JSON object.
    """
    projects_prompt = f"""
    Generate a JSON array of 3 unique, impressive resume projects for a Data Analyst.
    Critically analyze this job description and infuse relevant keywords and technologies from it into each project's "description".
    Job Description: {jd}
    Each object in the array must have "title", "description", and a valid "link".
    For the "description", write a concise 1-2 sentence summary using strong action verbs that highlight the project's impact and outcome.
    The tone should be professional yet human, like someone proudly describing their work, not just a list of technologies.
    Respond with only the raw JSON array.
    """
    experience_prompt = f"""
    Analyze the following job description: {jd}
    Now, take these existing resume tasks for a job role: {json.dumps(current_experience.get('tasks', []))}
    Rewrite these tasks to sound more human and achievement-oriented, while aligning with the job description by incorporating relevant keywords.
    Use strong action verbs (e.g., "Orchestrated," "Engineered," "Spearheaded") and focus on the quantifiable impact or result of each task.
    The rewritten tasks should sound like natural, accomplishment-driven bullet points on a real resume. Maintain the core responsibilities.
    Return a JSON array of the rewritten task strings. Respond with only the raw JSON array.
    """
    soft_skills_prompt = f"""
    Based on the following job description, extract an array of the 6-7 most important soft skills.
    Job Description: {jd}
    Respond with only a raw JSON array of strings.
    """

    # --- AI Calls ---
    skills_response = generate_with_gemini(skills_prompt)
    projects_response = generate_with_gemini(projects_prompt)
    experience_response = generate_with_gemini(experience_prompt)
    soft_skills_response = generate_with_gemini(soft_skills_prompt)

    # --- Use the helper to parse and validate all responses ---
    new_skills_data = parse_ai_json_response(skills_response, dict, "Technical Skills")
    new_projects = parse_ai_json_response(projects_response, list, "Projects")
    new_tasks = parse_ai_json_response(experience_response, list, "Experience Tasks")
    new_soft_skills = parse_ai_json_response(soft_skills_response, list, "Soft Skills")

    # If any response failed validation, stop immediately.
    if not all([new_skills_data, new_projects, new_tasks, new_soft_skills]):
        st.warning("Halting generation due to one or more AI response errors.")
        return None

    # --- NEW: Validate that all items within the 'new_projects' list are dictionaries ---
    if any(not isinstance(p, dict) for p in new_projects):
        st.error("AI Error: The 'Projects' response was not a list of project objects as expected.")
        st.code(json.dumps(new_projects, indent=2)) # Show the faulty structure
        return None

    # --- SAFER SUMMARY GENERATION ---
    tech_skills_dict = new_skills_data.get('technicalSkills', {})
    all_skills_flat = []
    if isinstance(tech_skills_dict, dict):
        for skill_list in tech_skills_dict.values():
            if isinstance(skill_list, list):
                all_skills_flat.extend(skill_list)

    summary_prompt = f"""
    Based on these skills: {', '.join(all_skills_flat)} and these projects: {', '.join([p.get('title', 'Untitled Project') for p in new_projects])}, write a concise 2-3 sentence professional summary for a Data Analyst.
    Write it in a professional and confident tone, as if a person is describing their achievements and capabilities. Make it impactful but natural, avoiding overly robotic language.
    """
    new_summary = generate_with_gemini(summary_prompt)
    if not new_summary: return None

    # --- ASSEMBLE FINAL DATA ---
    updated_experience = current_experience.copy()
    updated_experience['tasks'] = new_tasks

    return {
        "summary": new_summary,
        "technicalSkills": new_skills_data.get('technicalSkills', {}),
        "softSkills": new_soft_skills,
        "projects": new_projects,
        "experience": [updated_experience]
    }


# --- Initialize Session State ---
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = load_from_csv() or get_default_resume()

# --- UI Layout ---
st.title("📄 AI Resume Tailor")
with st.sidebar:
    st.header("Controls")
    edit_mode = st.checkbox("Enable Edit Mode")
    jd = st.text_area("Paste Job Description Here", height=250)

    if st.button("✨ Tailor Resume with AI", use_container_width=True, type="primary", disabled=(not model)):
        with st.spinner("🤖 AI is tailoring your resume..."):
            # --- Robustly get the current experience to be tailored ---
            experience_list = st.session_state.resume_data.get('experience', [])
            current_experience = None

            if isinstance(experience_list, list) and experience_list:
                if isinstance(experience_list[0], dict):
                    current_experience = experience_list[0]
                else:
                    st.error("The first experience entry is invalid. Please fix it in Edit Mode.")
            else:
                st.error("No experience data found. Please add an entry in Edit Mode to tailor it.")
            
            # --- Only proceed if the experience data is valid ---
            if current_experience:
                core_skills = get_default_resume()['technicalSkills']
                ai_content = generate_tailored_content(jd, current_experience, core_skills)
                
                if ai_content:
                    # REWRITE sections with new AI content
                    st.session_state.resume_data['summary'] = ai_content['summary']
                    st.session_state.resume_data['technicalSkills'] = ai_content.get('technicalSkills', {})
                    st.session_state.resume_data['softSkills'] = ai_content['softSkills']
                    st.session_state.resume_data['projects'] = ai_content['projects']
                    st.session_state.resume_data['experience'][0] = ai_content['experience'][0]
                    st.success("Resume tailored successfully!")
                    st.rerun()

    if st.button("💾 Save to CSV", use_container_width=True):
        save_to_csv(st.session_state.resume_data)

# --- Editable Data Display ---
resume = st.session_state.resume_data

if edit_mode:
    st.subheader("Personal Information")
    c1, c2 = st.columns(2)
    resume['fullName'] = c1.text_input("Full Name", resume.get('fullName', ''))
    resume['role'] = c2.text_input("Role / Title", resume.get('role', ''))
    c1, c2, c3 = st.columns(3)
    resume['phone'] = c1.text_input("Phone", resume.get('phone', ''))
    resume['email'] = c2.text_input("Email", resume.get('email', ''))
    resume['location'] = c3.text_input("Location", resume.get('location', ''))
    
    st.subheader("Professional Links")
    
    # --- Editable Link Sections ---
    link_keys = ['linkedin', 'github', 'kaggle', 'portfolio']
    for key in link_keys:
        if key not in resume or not isinstance(resume[key], dict):
             resume[key] = {"name": key.title(), "url": ""} # Initialize if not present
        
        c1, c2 = st.columns(2)
        resume[key]['name'] = c1.text_input(f"{key.title()} Name", resume[key].get('name', key.title()))
        resume[key]['url'] = c2.text_input(f"{key.title()} URL", resume[key].get('url', ''))

    st.divider()
else:
    st.header(resume.get('fullName', 'SACHIN TAMBE'))
    st.subheader(resume.get('role', 'Data Analyst | Aspiring Data Scientist'))
    
    # --- Dynamic Contact HTML for Web View with Data Validation ---
    # Ensure link data is in the correct dictionary format before use
    linkedin = resume.get('linkedin', {})
    if not isinstance(linkedin, dict):
        linkedin = {"name": "LinkedIn", "url": ""}
        resume['linkedin'] = linkedin # Correct the state

    github = resume.get('github', {})
    if not isinstance(github, dict):
        github = {"name": "GitHub", "url": ""}
        resume['github'] = github

    kaggle = resume.get('kaggle', {})
    if not isinstance(kaggle, dict):
        kaggle = {"name": "Kaggle", "url": ""}
        resume['kaggle'] = kaggle

    portfolio = resume.get('portfolio', {})
    if not isinstance(portfolio, dict):
        portfolio = {"name": "Portfolio", "url": ""}
        resume['portfolio'] = portfolio

    contact_html_parts = [
        resume.get('phone',''),
        f"<a href='mailto:{resume.get('email','')}'>{resume.get('email','')}</a>",
        f"<a href='{linkedin.get('url','')}' target='_blank'>{linkedin.get('name','LinkedIn')}</a>",
        f"<a href='{github.get('url','')}' target='_blank'>{github.get('name','GitHub')}</a>",
        f"<a href='{kaggle.get('url','')}' target='_blank'>{kaggle.get('name','Kaggle')}</a>",
        f"<a href='{portfolio.get('url','')}' target='_blank'>{portfolio.get('name','Portfolio')}</a>"
    ]
    contact_html = f"<p style='text-align: center;'>{' | '.join(filter(None, contact_html_parts))}</p>"
    contact_html += f"<p style='text-align: center;'>{resume.get('location','')}</p>"
    
    st.markdown(contact_html, unsafe_allow_html=True)
    st.divider()

resume['summary'] = st.text_area("Profile Summary", resume.get('summary', ''), height=100)

def display_and_edit_technical_skills(resume_data):
    st.subheader("Technical Skills")
    skills_data = resume_data.get('technicalSkills', {})

    if isinstance(skills_data, list):
        st.warning("Old technical skills format detected. Converting to new categorized format. Please review and save.")
        skills_dict = {
            "Programming & Databases": [s for s in skills_data if any(p in s.lower() for p in ['python', 'sql', 'mongo', 'shell'])],
            "BI & Visualization": [s for s in skills_data if any(p in s.lower() for p in ['power bi', 'excel', 'streamlit', 'tableau', 'matplotlib', 'seaborn'])],
            "Machine Learning": [s for s in skills_data if any(p in s.lower() for p in ['regression', 'classification', 'xgboost', 'random forest', 'arima', 'nlp'])],
        }
        all_categorized_skills = [s for sublist in skills_dict.values() for s in sublist]
        other_skills = [s for s in skills_data if s not in all_categorized_skills]
        if other_skills:
            skills_dict["Developer Tools & Cloud"] = other_skills
        
        resume_data['technicalSkills'] = skills_dict
    else:
        skills_dict = skills_data

    if edit_mode:
        categories = list(skills_dict.keys())
        for category in categories:
            with st.expander(category):
                new_category_name = st.text_input("Category Name", value=category, key=f"cat_name_{category}")
                skills_str = ", ".join(skills_dict.get(category, []))
                new_skills_str = st.text_area("Skills (comma-separated)", value=skills_str, key=f"skills_{category}")
                
                if new_category_name != category:
                    skills_dict[new_category_name] = skills_dict.pop(category)
                    category = new_category_name
                
                skills_dict[category] = [s.strip() for s in new_skills_str.split(',') if s.strip()]

                if st.button("Remove Category", key=f"del_cat_{category}", type="secondary"):
                    del skills_dict[category]
                    st.rerun()

        new_cat = st.text_input("New Category Name", key="new_category")
        if st.button("Add Category", key="add_category_btn"):
            if new_cat and new_cat not in skills_dict:
                skills_dict[new_cat] = []
                st.rerun()
    else:
        if isinstance(skills_dict, dict):
            for category, skills in skills_dict.items():
                st.markdown(f"**{category}:** {', '.join(skills)}")

display_and_edit_technical_skills(resume)

def edit_list_section(title, items, item_key):
    st.subheader(title)
    if edit_mode:
        for i in range(len(items) - 1, -1, -1):
            col1, col2 = st.columns([10, 1])
            items[i] = col1.text_input(f"label_{item_key}_{i}", value=items[i], label_visibility="collapsed")
            if col2.button("🗑️", key=f"del_{item_key}_{i}"):
                items.pop(i)
                st.rerun()
        new_item = st.text_input(f"Add new {title[:-1]}", key=f"add_{item_key}")
        if st.button(f"Add {title[:-1]}", key=f"add_btn_{item_key}"):
            if new_item:
                items.append(new_item)
                st.rerun()
    else:
        st.write(", ".join(items))

edit_list_section("Soft Skills", resume.get('softSkills', []), "soft")


def create_editable_section(section_title, section_key, template_item, display_func):
    st.subheader(section_title)
    # Ensure the section exists as a list before iterating
    if not isinstance(resume.get(section_key), list):
        resume[section_key] = []

    for i in range(len(resume.get(section_key, [])) - 1, -1, -1):
        item = resume[section_key][i]
        # Ensure item is a dictionary before processing
        if not isinstance(item, dict):
            # If item is not a dict, skip or replace it
            continue 

        if edit_mode:
            expander_title = item.get('title', item.get('degree', item.get('name', f"Entry {i+1}")))
            with st.expander(f"Edit {section_title[:-1]} {i+1}: {expander_title}"):
                # Use item.copy().items() to avoid issues while iterating and modifying
                for key, value in item.copy().items():
                    if isinstance(value, list):
                        item[key] = st.text_area(f"{key.title()}", "\n".join(value), key=f"{section_key}_{i}_{key}").split('\n')
                    else:
                        item[key] = st.text_input(f"{key.title()}", str(value), key=f"{section_key}_{i}_{key}")
                if st.button(f"Remove Entry {i+1}", key=f"del_{section_key}_{i}", type="secondary"):
                    resume[section_key].pop(i)
                    st.rerun()
        else:
            display_func(item)

    if edit_mode and st.button(f"Add {section_title[:-1]}", key=f"add_btn_{section_key}"):
        resume.setdefault(section_key, []).append(dict(template_item))
        st.rerun()

create_editable_section("Professional Experience", "experience", {"title": "New Role", "company": "Company", "duration": "Date - Date", "location": "City, Country", "tasks": ["Task 1"]},
    lambda item: st.markdown(f"**{item.get('title','')}** | _{item.get('company','')}_ | {item.get('duration','')}\n" + "".join([f"\n- {task}" for task in item.get('tasks', [])])))
create_editable_section("Education", "education", {"degree": "Degree Name", "institution": "University", "duration": "Year - Year", "location": "City, Country"},
    lambda item: st.markdown(f"**{item.get('degree','')}**, {item.get('institution','')} ({item.get('duration','')})"))
create_editable_section("Projects", "projects", {"title": "New Project", "description": "...", "link": "#"},
    lambda item: st.markdown(f"**<a href='{item.get('link', '#')}' target='_blank'>{item.get('title','')}</a>**: {item.get('description','')}", unsafe_allow_html=True))
create_editable_section("Certifications", "certifications", {"name": "Certification Name", "issuer": "Issuing Body", "link": "#"},
    lambda item: st.markdown(f"- <a href='{item.get('link', '#')}' target='_blank'>{item.get('name','')}</a> *({item.get('issuer','')})*", unsafe_allow_html=True))


# --- PDF Download Button ---
st.sidebar.divider()
pdf_data = create_resume_pdf(st.session_state.resume_data)
st.sidebar.download_button(label="⬇️ Download PDF", data=pdf_data, file_name=f"{resume.get('fullName', 'resume').replace(' ', '_')}_Resume.pdf", mime="application/pdf", use_container_width=True)

