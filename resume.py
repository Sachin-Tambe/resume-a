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
    if GEMINI_API_KEY != "YOUR_API_KEY_HERE":
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
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
    """Generates a high-quality, single-page PDF using ReportLab with all requested formatting."""
    buffer = io.BytesIO()
    # Aggressively tight margins for one-page fit
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=0.4*inch, rightMargin=0.4*inch, topMargin=0.2*inch, bottomMargin=0.2*inch)
    story = []
    styles = getSampleStyleSheet()

    primary_color = HexColor("#005A9C")
    # Adjusted styles for maximum compactness
    styles.add(ParagraphStyle(name='Name', alignment=TA_CENTER, fontSize=22, textColor=primary_color, fontName='Helvetica-Bold', leading=22))
    styles.add(ParagraphStyle(name='Contact', alignment=TA_CENTER, fontSize=8.5, spaceAfter=0, textColor=primary_color, leading=10))
    styles.add(ParagraphStyle(name='Location', alignment=TA_CENTER, fontSize=8.5, spaceAfter=4))
    styles.add(ParagraphStyle(name='SectionTitle', fontSize=10.5, textColor=primary_color, fontName='Helvetica-Bold', spaceBefore=2, spaceAfter=0))
    styles.add(ParagraphStyle(name='JobTitle', fontSize=9.5, fontName='Helvetica-Bold', leading=11, spaceAfter=1))
    
    bullet_style = styles['Bullet']
    bullet_style.leftIndent = 12
    bullet_style.firstLineIndent = -5
    bullet_style.spaceAfter = 0
    bullet_style.fontSize = 9
    bullet_style.leading = 10

    styles.add(ParagraphStyle(name='Body', parent=styles['Normal'], fontSize=9, leading=10))
    skill_style = ParagraphStyle(name='Skill', parent=styles['Body'], spaceAfter=0, leading=10)

    # --- Header ---
    contact_line_1 = f"""
        {data.get('phone','')} | <a href="mailto:{data.get('email','')}"><u>{data.get('email','')}</u></a> |
        <a href="{data.get('linkedin','')}"><u>LinkedIn</u></a> |
        <a href="{data.get('github','')}"><u>GitHub</u></a> |
        <a href="{data.get('kaggle','')}"><u>Kaggle</u></a> |
        <a href="{data.get('portfolio','')}"><u>Portfolio</u></a>
    """
    header_section = [
        Paragraph(data.get('fullName', 'SACHIN TAMBE'), styles['Name']),
        Paragraph(contact_line_1, styles['Contact']),
        Paragraph(data.get('location', 'Mumbai, Maharashtra'), styles['Location']),
        Spacer(1, 4)
    ]
    story.append(KeepTogether(header_section))


    line = Table([['']], colWidths=['100%'], style=[('LINEBELOW', (0,0), (-1,0), 0.5, HexColor("#DDDDDD"))])

    # --- Summary Section ---
    summary_section = [
        Paragraph("Profile Summary", styles['SectionTitle']),
        line,
        Spacer(1, 1),
        Paragraph(data.get('summary', ''), styles['Body'])
    ]
    story.append(KeepTogether(summary_section))

    # --- Technical Skills Section ---
    tech_skills_section = [
        Paragraph("Technical Skills", styles['SectionTitle']),
        line,
        Spacer(1, 1)
    ]
    tech_skills = data.get('technicalSkills', {})
    if isinstance(tech_skills, dict):
        for category, skills in tech_skills.items():
            tech_skills_section.append(Paragraph(f"• <b>{category}:</b> {', '.join(skills)}", skill_style))
    story.append(KeepTogether(tech_skills_section))

    # --- Soft Skills Section ---
    soft_skills_section = [
        Paragraph("Soft Skills", styles['SectionTitle']),
        line,
        Spacer(1, 1),
        Paragraph(", ".join(data.get('softSkills', [])), styles['Body'])
    ]
    story.append(KeepTogether(soft_skills_section))

    # --- Professional Experience Section ---
    story.append(Paragraph("Professional Experience", styles['SectionTitle']))
    story.append(line)
    story.append(Spacer(1, 1))
    for exp in data.get('experience', []):
        exp_block = [
            Paragraph(f"<b>{exp['title']}</b> | <i>{exp['company']}</i> | {exp['duration']}", styles['JobTitle'])
        ]
        for task in exp.get('tasks', []):
            exp_block.append(Paragraph(f"• {task}", bullet_style))
        story.append(KeepTogether(exp_block))


    # --- Education Section ---
    education_section = [
        Paragraph("Education", styles['SectionTitle']),
        line,
        Spacer(1, 1)
    ]
    for edu in data.get('education', []):
        education_section.append(Paragraph(f"<b>{edu['degree']}</b>, {edu['institution']} ({edu['duration']})", styles['Body']))
    story.append(KeepTogether(education_section))

    # --- Projects Section ---
    story.append(Paragraph("Projects", styles['SectionTitle']))
    story.append(line)
    story.append(Spacer(1, 1))
    for proj in data.get('projects', []):
        link = proj.get('link', '').strip()
        project_text = f"<a href='{link}'><u>{proj['title']}</u></a>: {proj['description']}" if link and link != "#" else f"<b>{proj['title']}</b>: {proj['description']}"
        project_block = [
            Paragraph(project_text, styles['Body']),
            Spacer(1, 1)
        ]
        story.append(KeepTogether(project_block))
    
    # --- Certifications Section ---
    certifications_section = [
        Paragraph("Certifications", styles['SectionTitle']),
        line,
        Spacer(1, 1)
    ]
    for cert in data.get('certifications', []):
        link = cert.get('link', '').strip()
        cert_text = f"<a href='{link}'><u>{cert['name']}</u></a> ({cert['issuer']})" if link and link != "#" else f"{cert['name']} ({cert['issuer']})"
        certifications_section.append(Paragraph(f"• {cert_text}", bullet_style))
    story.append(KeepTogether(certifications_section))

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
        "linkedin": "https://linkedin.com/in/sachin-tambe",
        "github": "https://github.com/sachintambe",
        "kaggle": "https://www.kaggle.com/sachintambe",
        "portfolio": "https://www.example.com",
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
    if not model:
        st.error("Gemini model is not configured. Please add your API key.")
        return None
    try:
        response = model.generate_content(prompt).text.strip()
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response)
        return cleaned_response
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return None

def generate_tailored_content(jd, current_experience, core_skills):
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

    if not all([skills_response, projects_response, experience_response, soft_skills_response]): return None

    try:
        new_skills_data = json.loads(skills_response)
        new_projects = json.loads(projects_response)
        new_tasks = json.loads(experience_response)
        new_soft_skills = json.loads(soft_skills_response)

    except json.JSONDecodeError as e:
        st.error(f"AI failed to generate valid JSON: {e}. Please try again.")
        return None
        
    summary_prompt = f"""
    Based on these skills: {', '.join(list(new_skills_data.get('technicalSkills', {}).values())[0]) if new_skills_data.get('technicalSkills') else ''} and these projects: {', '.join([p['title'] for p in new_projects])}, write a concise 2-3 sentence professional summary for a Data Analyst.
    Write it in a professional and confident tone, as if a person is describing their achievements and capabilities. Make it impactful but natural, avoiding overly robotic language.
    """
    new_summary = generate_with_gemini(summary_prompt)
    if not new_summary: return None
    
    updated_experience = current_experience.copy()
    updated_experience['tasks'] = new_tasks

    return {
        "summary": new_summary,
        "technicalSkills": new_skills_data.get('technicalSkills', {}),
        "softSkills": new_soft_skills,
        "projects": new_projects,
        "experience": [updated_experience] # Assuming we only update the first experience
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
            current_experience = st.session_state.resume_data.get('experience', [{}])[0]
            core_skills = get_default_resume()['technicalSkills'] # Use the default skills as the baseline
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
    c1, c2, c3, c4 = st.columns(4)
    resume['linkedin'] = c1.text_input("LinkedIn URL", resume.get('linkedin', ''))
    resume['github'] = c2.text_input("GitHub URL", resume.get('github', ''))
    resume['kaggle'] = c3.text_input("Kaggle URL", resume.get('kaggle', ''))
    resume['portfolio'] = c4.text_input("Portfolio URL", resume.get('portfolio', ''))
    st.divider()
else:
    st.header(resume.get('fullName', 'SACHIN TAMBE'))
    st.subheader(resume.get('role', 'Data Analyst | Aspiring Data Scientist'))
    contact_html = f"""
    <p style='text-align: center;'>
    {resume.get('phone','')} | <a href="mailto:{resume.get('email','')}">{resume.get('email','')}</a> | <a href="{resume.get('linkedin','')}" target="_blank">LinkedIn</a> | <a href="{resume.get('github','')}" target="_blank">GitHub</a> | <a href="{resume.get('kaggle','')}" target="_blank">Kaggle</a> | <a href="{resume.get('portfolio','')}" target="_blank">Portfolio</a>
    </p>
    <p style='text-align: center;'>{resume.get('location','')}</p>
    """
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
    for i in range(len(resume.get(section_key, [])) - 1, -1, -1):
        item = resume[section_key][i]
        if edit_mode:
            expander_title = item.get('title', item.get('degree', item.get('name')))
            with st.expander(f"Edit {section_title[:-1]} {i+1}: {expander_title}"):
                for key, value in item.items():
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
    lambda item: st.markdown(f"**{item['title']}** | _{item['company']}_ | {item['duration']}\n" + "".join([f"\n- {task}" for task in item.get('tasks', [])])))
create_editable_section("Education", "education", {"degree": "Degree Name", "institution": "University", "duration": "Year - Year", "location": "City, Country"},
    lambda item: st.markdown(f"**{item['degree']}**, {item['institution']} ({item['duration']})"))
create_editable_section("Projects", "projects", {"title": "New Project", "description": "...", "link": "#"},
    lambda item: st.markdown(f"**<a href='{item.get('link', '#')}' target='_blank'>{item['title']}</a>**: {item['description']}", unsafe_allow_html=True))
create_editable_section("Certifications", "certifications", {"name": "Certification Name", "issuer": "Issuing Body", "link": "#"},
    lambda item: st.markdown(f"- <a href='{item.get('link', '#')}' target='_blank'>{item['name']}</a> *({item['issuer']})*", unsafe_allow_html=True))


# --- PDF Download Button ---
st.sidebar.divider()
pdf_data = create_resume_pdf(st.session_state.resume_data)
st.sidebar.download_button(label="⬇️ Download PDF", data=pdf_data, file_name=f"{resume.get('fullName', 'resume').replace(' ', '_')}_Resume.pdf", mime="application/pdf", use_container_width=True)

