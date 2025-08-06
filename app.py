# AI Interview Simulator - Kurated.ai Style
# Simple structure: Just app.py + main.css in root directory

import streamlit as st
import google.generativeai as genai
import os
import json
import time
import pandas as pd
from typing import Dict, List, Optional
from io import BytesIO
import base64
from dotenv import load_dotenv
from datetime import datetime, timedelta

# File processing imports
import PyPDF2
from docx import Document
import mammoth

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS from root directory
def load_css():
    """Load main.css from root directory"""
    try:
        with open('main.css', 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ main.css not found in root directory. Using fallback styles.")
        # Fallback CSS
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Plus Jakarta Sans', sans-serif; }
        .main .block-container { background: #FFF9F0; padding: 2rem; }
        #MainMenu, footer, header { visibility: hidden; }
        .app-header { background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%); color: white; padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem; }
        .content-card { background: white; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #F3F4F6; }
        .stButton > button { background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%) !important; color: white !important; border: none !important; border-radius: 50px !important; padding: 0.75rem 2rem !important; font-weight: 600 !important; }
        </style>
        """, unsafe_allow_html=True)

# Gemini API Configuration
class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in your environment.")
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def generate_questions(self, resume_text: str, job_details: Dict, num_questions: int) -> List[str]:
        """Generate behavioral interview questions based on resume and job details."""
        prompt = f"""
        You are an expert behavioral interviewer. Generate exactly {num_questions} behavioral interview questions based on the resume and job description provided.

        RESUME CONTENT:
        {resume_text}

        JOB DETAILS:
        - Title: {job_details.get('job_title', 'N/A')}
        - Company: {job_details.get('company_name', 'N/A')}
        - Description: {job_details.get('job_description', 'N/A')}
        - Experience Level: {job_details.get('experience_years', 0)} years
        - Interview Duration: {job_details.get('duration', 15)} minutes

        REQUIREMENTS:
        1. Generate exactly {num_questions} questions - no more, no less
        2. Focus on HEARS method (Headline, Events, Actions, Results, Significance)
        3. Tailor questions to candidate's background and job requirements
        4. Include variety: leadership, problem-solving, conflict resolution, teamwork, adaptability, communication
        5. Match difficulty to experience level and interview duration
        6. Make questions specific and actionable
        7. Ensure questions encourage detailed responses covering all HEARS elements

        IMPORTANT: Return your response in this EXACT format as a valid JSON array:
        ["Question 1 text here", "Question 2 text here", "Question 3 text here"]

        Do not include any other text, explanations, or formatting. Just the JSON array with exactly {num_questions} questions.
        """
        
        try:
            response = self.model.generate_content(prompt)
            questions_text = response.text.strip()
            
            # Clean up the response text
            questions_text = questions_text.strip()
            
            # Remove any markdown formatting if present
            if questions_text.startswith('```'):
                lines = questions_text.split('\n')
                questions_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else questions_text
            
            # Try to extract JSON array
            start_idx = questions_text.find('[')
            end_idx = questions_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = questions_text[start_idx:end_idx]
                try:
                    questions = json.loads(json_text)
                    if isinstance(questions, list) and len(questions) >= num_questions:
                        return questions[:num_questions]
                    elif isinstance(questions, list):
                        fallback = self._get_fallback_questions(num_questions - len(questions))
                        return questions + fallback
                except json.JSONDecodeError:
                    pass
            
            # Enhanced fallback parsing
            questions = []
            lines = questions_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('"') and line.endswith('",'):
                    questions.append(line[1:-2])
                elif line.startswith('"') and line.endswith('"'):
                    questions.append(line[1:-1])
                elif line.startswith('- '):
                    questions.append(line[2:])
                elif line.startswith(f'{len(questions)+1}.'):
                    questions.append(line[len(f'{len(questions)+1}.'):].strip())
            
            if len(questions) < num_questions:
                fallback_questions = self._get_fallback_questions(num_questions - len(questions))
                questions.extend(fallback_questions)
            
            return questions[:num_questions]
                
        except Exception as e:
            st.error(f"Error generating questions: {str(e)}")
            return self._get_fallback_questions(num_questions)
    
    def generate_individual_feedback(self, question: str, answer: str, job_details: Dict) -> str:
        """Generate HEARS feedback for individual question."""
        prompt = f"""
        Analyze this single interview question and answer using the HEARS methodology:

        QUESTION: {question}
        CANDIDATE'S ANSWER: {answer}
        JOB CONTEXT: {job_details.get('job_title', 'N/A')} at {job_details.get('company_name', 'N/A')}

        Provide feedback in this format:

        ## 🎯 Question Analysis

        **H (Headline):** [Did they provide a clear situation summary? Rate 1-10]
        **E (Events):** [Did they describe specific events/challenges? Rate 1-10]
        **A (Actions):** [Did they detail their specific actions? Rate 1-10]
        **R (Results):** [Did they share measurable outcomes? Rate 1-10]
        **S (Significance):** [Did they demonstrate skills/learning? Rate 1-10]

        **Overall Score:** X/10
        **Strengths:** [2-3 key strengths in this response]
        **Areas for Improvement:** [1-2 specific suggestions]
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Unable to generate detailed feedback for this question."
    
    def generate_overall_feedback(self, all_responses: List, job_details: Dict) -> str:
        """Generate comprehensive HEARS methodology feedback."""
        responses_text = "\n\n".join([
            f"Q{i+1}: {response['question']}\nA{i+1}: {response['answer']}"
            for i, response in enumerate(all_responses)
        ])
        
        prompt = f"""
        Analyze this complete behavioral interview using the HEARS methodology:

        INTERVIEW RESPONSES: {responses_text}
        JOB CONTEXT: {job_details}
        INTERVIEW DURATION: {job_details.get('duration', 15)} minutes
        TOTAL QUESTIONS: {len(all_responses)}

        Provide comprehensive feedback in this EXACT format:

        # 🎯 OVERALL INTERVIEW FEEDBACK REPORT

        ## **📰 HEADLINE ANALYSIS**
        [How well did candidate provide situation summaries across all questions]
        **Headline Score: X/10**

        ## **📅 EVENTS ANALYSIS**  
        [Quality of situations/challenges described across all responses]
        **Events Score: X/10**
        • Key Event 1: [Brief description]
        • Key Event 2: [Brief description] 
        • Key Event 3: [Brief description]

        ## **⚡ ACTIONS ANALYSIS**
        [Depth and specificity of actions described]
        **Actions Score: X/10**
        • Strong Action Example: [Description]
        • Area for Improvement: [Suggestion]

        ## **🎊 RESULTS ANALYSIS**
        [Quality of outcomes and measurable impacts shared]
        **Results Score: X/10**
        • Quantified Result 1: [Description with numbers]
        • Quantified Result 2: [Description with numbers]

        ## **💡 SIGNIFICANCE ANALYSIS**
        **Skills Demonstrated:**
        - Leadership: [Analysis] - **Score: X/10**
        - Problem-Solving: [Analysis] - **Score: X/10**  
        - Communication: [Analysis] - **Score: X/10**
        - Teamwork: [Analysis] - **Score: X/10**
        - Adaptability: [Analysis] - **Score: X/10**

        ## **📈 OVERALL ASSESSMENT**
        **Interview Duration Performance:** [How well they used the time]
        **HEARS Methodology Adherence:** X/10
        **Top 3 Strengths:** [List with specific examples]
        **Top 3 Development Areas:** [Specific, actionable improvements]
        **Overall Interview Score:** **X/10**
        **Hiring Recommendation:** **[STRONG HIRE/HIRE/MAYBE/PASS]**

        ## **🚀 IMPROVEMENT RECOMMENDATIONS**
        **For Future Interviews:**
        [Specific, actionable advice based on HEARS gaps]
        
        **For Professional Development:**
        [Skills to develop based on responses]
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating comprehensive feedback: {str(e)}"
    
    def _get_fallback_questions(self, num_questions: int) -> List[str]:
        """Fallback questions if API fails."""
        fallback_questions = [
            "Tell me about a time when you had to lead a team through a difficult project. What was your approach and what were the results?",
            "Describe a situation where you had to solve a complex problem with limited resources. How did you handle it and what did you learn?",
            "Can you share an example of when you had to work with a difficult team member or stakeholder? What actions did you take?",
            "Tell me about a time when you had to adapt quickly to a significant change in your work environment. What was the outcome?",
            "Describe a situation where you made a mistake. How did you handle it and what did you learn from the experience?",
            "Give me an example of when you had to influence others without having direct authority over them. What was the result?",
            "Tell me about a time when you had to work under tight deadlines. How did you prioritize and manage your time?",
            "Describe a situation where you had to learn a new skill quickly to complete a project. What was the impact?",
            "Can you share an example of when you had to give difficult feedback to a colleague? How did you approach it?",
            "Tell me about a time when you had to make a decision with incomplete information. What was the outcome?",
            "Describe a situation where you had to manage competing priorities from different stakeholders. How did you handle it?",
            "Give me an example of when you went above and beyond what was expected in your role. What were the results?"
        ]
        
        return fallback_questions[:num_questions]

# File Processing Functions
class FileProcessor:
    @staticmethod
    def validate_file(uploaded_file) -> tuple[bool, str]:
        """Validate uploaded file size and format."""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if uploaded_file.size > max_size:
            return False, f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (10MB)"
        
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in allowed_extensions:
            return False, f"Unsupported file format. Please upload: {', '.join(allowed_extensions)}"
        
        return True, "File validated successfully"
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(docx_file) -> str:
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def extract_text_from_doc(doc_file) -> str:
        try:
            result = mammoth.extract_raw_text(doc_file)
            return result.value.strip()
        except Exception as e:
            raise Exception(f"Error reading DOC: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(txt_file) -> str:
        try:
            return txt_file.read().decode('utf-8').strip()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    @classmethod
    def process_resume_file(cls, uploaded_file) -> tuple[bool, str]:
        is_valid, message = cls.validate_file(uploaded_file)
        if not is_valid:
            return False, message
        
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.pdf':
                text = cls.extract_text_from_pdf(uploaded_file)
            elif file_extension == '.docx':
                text = cls.extract_text_from_docx(uploaded_file)
            elif file_extension == '.doc':
                text = cls.extract_text_from_doc(uploaded_file)
            elif file_extension == '.txt':
                text = cls.extract_text_from_txt(uploaded_file)
            else:
                return False, "Unsupported file format"
            
            if len(text.strip()) < 50:
                return False, "Resume appears to be empty or too short. Please upload a valid resume."
            
            return True, text
        
        except Exception as e:
            return False, f"Error processing file: {str(e)}"

# Timer functionality
class InterviewTimer:
    def __init__(self, duration_minutes: int):
        self.duration_seconds = duration_minutes * 60
        self.start_time = None
        self.question_start_time = None
    
    def start_interview(self):
        self.start_time = datetime.now()
    
    def start_question(self):
        self.question_start_time = datetime.now()
    
    def get_remaining_time(self) -> int:
        if not self.start_time:
            return self.duration_seconds
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = max(0, self.duration_seconds - elapsed)
        return int(remaining)
    
    def get_question_time(self) -> int:
        if not self.question_start_time:
            return 0
        
        elapsed = (datetime.now() - self.question_start_time).total_seconds()
        return int(elapsed)
    
    def format_time(self, seconds: int) -> str:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

# Session State Management
def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'stage': 'upload',
        'resume_text': "",
        'job_details': {},
        'interview_duration': 15,
        'num_questions': 3,
        'questions': [],
        'current_question_idx': 0,
        'conversation': [],
        'question_responses': [],
        'individual_feedback': [],
        'overall_feedback': "",
        'interview_completed': False,
        'timer': None,
        'question_timer_start': None,
        'gemini_client': None,
        'duration_selected': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.gemini_client is None:
        try:
            st.session_state.gemini_client = GeminiClient()
        except Exception as e:
            st.error(f"Failed to initialize AI client: {str(e)}")

# UI Components
def render_header():
    """Render application header."""
    st.markdown("""
    <div class="app-header">
        <h1>🚀 AI Interview Simulator</h1>
        <p>Master behavioral interviews with AI-powered HEARS methodology feedback</p>
    </div>
    """, unsafe_allow_html=True)

def render_progress_stepper():
    """Render enhanced progress stepper that looks more compelling."""
    stages = ['upload', 'details', 'interview', 'feedback']
    stage_names = ['Upload Resume', 'Job Details', 'Interview', 'Feedback']
    stage_icons = ['📄', '📝', '🎤', '📊']
    current_stage_idx = stages.index(st.session_state.stage)
    
    # Calculate progress percentage
    progress_value = current_stage_idx / (len(stages) - 1) if len(stages) > 1 else 0
    
    # Enhanced card container with better styling
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px; 
        padding: 3rem 2rem; 
        margin-bottom: 3rem; 
        box-shadow: 
            0 24px 48px rgba(0, 0, 0, 0.08),
            0 8px 16px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(245, 158, 11, 0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #F59E0B 0%, #FBB042 50%, #FBBF24 100%);
        "></div>
        <div style="text-align: center; margin-bottom: 3rem;">
            <h2 style="
                font-size: 1.75rem; 
                font-weight: 700; 
                color: #374151; 
                margin-bottom: 0.75rem;
                letter-spacing: -0.02em;
            ">Interview Progress</h2>
            <p style="
                font-size: 1rem; 
                color: #6B7280;
                font-weight: 500;
                max-width: 500px;
                margin: 0 auto;
                line-height: 1.5;
            ">Follow the steps to complete your practice interview journey</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced progress bar
    st.markdown(f"""
    <div style="
        background: #F3F4F6;
        height: 8px;
        border-radius: 4px;
        margin-bottom: 2rem;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    ">
        <div style="
            height: 100%;
            background: linear-gradient(90deg, #F59E0B 0%, #FBB042 50%, #FBBF24 100%);
            border-radius: 4px;
            width: {progress_value * 100}%;
            transition: width 0.6s ease-out;
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
        "></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced step indicators using columns
    cols = st.columns(len(stages))
    
    for i, (stage, name, icon) in enumerate(zip(stages, stage_names, stage_icons)):
        with cols[i]:
            if i < current_stage_idx:
                # Completed step
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem 0.5rem;">
                    <div style="
                        width: 56px; 
                        height: 56px; 
                        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                        border-radius: 50%; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        margin: 0 auto 1rem auto; 
                        color: white; 
                        font-weight: 700;
                        font-size: 1.25rem;
                        box-shadow: 
                            0 8px 16px rgba(16, 185, 129, 0.3),
                            0 4px 8px rgba(16, 185, 129, 0.2);
                        border: 3px solid rgba(255, 255, 255, 0.9);
                    ">
                        ✓
                    </div>
                    <div style="
                        font-size: 0.875rem; 
                        font-weight: 700; 
                        color: #065F46;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                    ">{name}</div>
                    <div style="
                        font-size: 0.75rem; 
                        color: #059669;
                        margin-top: 0.25rem;
                        font-weight: 600;
                    ">Completed</div>
                </div>
                """, unsafe_allow_html=True)
            elif i == current_stage_idx:
                # Active step
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem 0.5rem;">
                    <div style="
                        width: 56px; 
                        height: 56px; 
                        background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%);
                        border-radius: 50%; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        margin: 0 auto 1rem auto; 
                        color: white; 
                        font-weight: 700;
                        font-size: 1.25rem;
                        box-shadow: 
                            0 12px 24px rgba(245, 158, 11, 0.4),
                            0 4px 8px rgba(245, 158, 11, 0.3);
                        border: 3px solid rgba(255, 255, 255, 0.9);
                        animation: pulse-glow 2s infinite;
                    ">
                        {icon}
                    </div>
                    <div style="
                        font-size: 0.875rem; 
                        font-weight: 700; 
                        color: #92400E;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                    ">{name}</div>
                    <div style="
                        font-size: 0.75rem; 
                        color: #F59E0B;
                        margin-top: 0.25rem;
                        font-weight: 600;
                    ">In Progress</div>
                </div>
                <style>
                @keyframes pulse-glow {{
                    0%, 100% {{ 
                        box-shadow: 
                            0 12px 24px rgba(245, 158, 11, 0.4),
                            0 4px 8px rgba(245, 158, 11, 0.3);
                    }}
                    50% {{ 
                        box-shadow: 
                            0 16px 32px rgba(245, 158, 11, 0.5),
                            0 6px 12px rgba(245, 158, 11, 0.4);
                        transform: scale(1.05);
                    }}
                }}
                </style>
                """, unsafe_allow_html=True)
            else:
                # Locked step
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem 0.5rem;">
                    <div style="
                        width: 56px; 
                        height: 56px; 
                        background: #F9FAFB;
                        border: 3px solid #E5E7EB;
                        border-radius: 50%; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        margin: 0 auto 1rem auto; 
                        color: #9CA3AF; 
                        font-weight: 600;
                        font-size: 1.25rem;
                        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
                    ">
                        {icon}
                    </div>
                    <div style="
                        font-size: 0.875rem; 
                        font-weight: 600; 
                        color: #9CA3AF;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                    ">{name}</div>
                    <div style="
                        font-size: 0.75rem; 
                        color: #D1D5DB;
                        margin-top: 0.25rem;
                        font-weight: 500;
                    ">Pending</div>
                </div>
                """, unsafe_allow_html=True)

def render_upload_stage():
    """Render resume upload stage with enhanced design."""
    st.markdown('<div class="content-card animate-fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📄 Upload Your Resume</h2>
        <p class="section-subtitle">Start by uploading your resume to get personalized interview questions tailored to your experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'doc', 'docx', 'txt'],
        help="Maximum file size: 10MB. Supported formats: PDF, DOC, DOCX, TXT",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing your resume..."):
            success, result = FileProcessor.process_resume_file(uploaded_file)
            
            if success:
                st.session_state.resume_text = result
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: #065F46; padding: 1.5rem 2rem; border-radius: 16px; margin: 2rem 0; font-weight: 600; box-shadow: 0 8px 16px rgba(16, 185, 129, 0.2); border: 1px solid #6EE7B7; display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 1.5rem;">✅</span>
                    <span>Resume uploaded and processed successfully!</span>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📖 Resume Preview", expanded=False):
                    preview_text = result[:500] + "..." if len(result) > 500 else result
                    st.markdown(f"""
                    <div style="background: #F8FAFC; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3B82F6; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.6; color: #374151;">
                        {preview_text}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add spacing before button
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Continue to Job Details →", key="continue_to_details", use_container_width=True):
                        st.session_state.stage = 'details'
                        st.rerun()
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); color: #991B1B; padding: 1.5rem 2rem; border-radius: 16px; margin: 2rem 0; font-weight: 600; box-shadow: 0 8px 16px rgba(239, 68, 68, 0.2); border: 1px solid #FCA5A5; display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 1.5rem;">❌</span>
                    <span>{result}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Add tips section when no file is uploaded
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 176, 66, 0.05) 100%); border-radius: 16px; padding: 2rem; margin: 2rem 0; border-left: 6px solid #F59E0B;">
            <h4 style="margin: 0 0 1.5rem 0; color: #92400E; font-size: 1.125rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">💡</span>
                Tips for Best Results
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; color: #78350F; line-height: 1.6;">
                <div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">📋 Content Quality</div>
                    <div style="font-size: 0.875rem;">Ensure your resume is up-to-date with recent experience and specific achievements</div>
                </div>
                <div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">📊 Quantifiable Results</div>
                    <div style="font-size: 0.875rem;">Include metrics and numbers to showcase your impact and accomplishments</div>
                </div>
                <div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">🔧 Technical Skills</div>
                    <div style="font-size: 0.875rem;">List relevant technologies, tools, and frameworks you've worked with</div>
                </div>
                <div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">📄 File Format</div>
                    <div style="font-size: 0.875rem;">PDF format typically gives the best text extraction results</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_details_stage():
    """Render job details collection stage with duration selection."""
    st.markdown('<div class="content-card animate-fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">⏱️ Interview Setup</h2>
        <p class="section-subtitle">Configure your practice interview duration and provide job details</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Duration Selection
    st.markdown("### Choose Interview Duration")
    
    duration_options = [
        {"label": "Quick Practice", "duration": 15, "questions": 3, "desc": "Perfect for a quick skills check", "icon": "⚡"},
        {"label": "Standard Interview", "duration": 30, "questions": 6, "desc": "Most common interview length", "icon": "⏰"},
        {"label": "Comprehensive", "duration": 45, "questions": 9, "desc": "Deep dive interview practice", "icon": "📋"},
        {"label": "Extended Session", "duration": 60, "questions": 12, "desc": "Full interview simulation", "icon": "🎯"}
    ]
    
    # Duration selection buttons
    cols = st.columns(len(duration_options))
    for i, option in enumerate(duration_options):
        with cols[i]:
            if st.button(f"Select {option['duration']}min", key=f"dur_{i}", use_container_width=True):
                st.session_state.interview_duration = option["duration"]
                st.session_state.num_questions = option["questions"]
                st.session_state.duration_selected = True
                st.rerun()
    
    if st.session_state.duration_selected:
        st.markdown(f"""
        <div class="status-message status-success">
            <span>✅</span>
            <span>Selected: {st.session_state.interview_duration} minutes ({st.session_state.num_questions} questions)</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Job Details Form
    st.markdown("### Job Information")
    
    with st.form("job_details_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title *", placeholder="e.g., Senior Software Engineer")
            company_name = st.text_input("Company Name *", placeholder="e.g., TechCorp Inc.")
        
        with col2:
            experience_years = st.number_input("Years of Experience Required", min_value=0, max_value=50, value=3)
            industry = st.selectbox(
                "Industry (Optional)",
                ["", "Technology", "Healthcare", "Finance", "Marketing", "Sales", "Education", "Manufacturing", "Retail", "Other"]
            )
        
        job_description = st.text_area(
            "Job Description *",
            placeholder="Paste the complete job description here, including responsibilities, requirements, and qualifications...",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Generate Interview Questions", use_container_width=True)
        
        if submitted:
            if not job_title or not company_name or not job_description:
                st.markdown("""
                <div class="status-message status-error">
                    <span>❌</span>
                    <span>Please fill in all required fields (marked with *)</span>
                </div>
                """, unsafe_allow_html=True)
            elif not st.session_state.duration_selected:
                st.markdown("""
                <div class="status-message status-error">
                    <span>❌</span>
                    <span>Please select an interview duration first</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                job_details = {
                    'job_title': job_title,
                    'company_name': company_name,
                    'job_description': job_description,
                    'experience_years': experience_years,
                    'industry': industry,
                    'duration': st.session_state.interview_duration
                }
                
                st.session_state.job_details = job_details
                
                with st.spinner(f"🤖 Generating {st.session_state.num_questions} personalized interview questions..."):
                    try:
                        questions = st.session_state.gemini_client.generate_questions(
                            st.session_state.resume_text,
                            job_details,
                            st.session_state.num_questions
                        )
                        
                        st.session_state.questions = questions
                        st.session_state.timer = InterviewTimer(st.session_state.interview_duration)
                        st.session_state.stage = 'interview'
                        
                        st.markdown("""
                        <div class="status-message status-success">
                            <span>🎉</span>
                            <span>Questions generated successfully! Starting your interview...</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.markdown(f"""
                        <div class="status-message status-error">
                            <span>❌</span>
                            <span>Error generating questions: {str(e)}</span>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_interview_stage():
    """Render interactive interview stage with timer."""
    if not st.session_state.questions:
        st.error("No questions available. Please go back and regenerate questions.")
        return
    
    # Start interview timer if not started
    if st.session_state.timer and not st.session_state.timer.start_time:
        st.session_state.timer.start_interview()
    
    # Timer display
    if st.session_state.timer:
        remaining = st.session_state.timer.get_remaining_time()
        total_duration = st.session_state.interview_duration * 60
        time_str = st.session_state.timer.format_time(remaining)
        
        timer_class = "timer-display"
        if remaining < total_duration * 0.25:
            timer_class += " danger"
        elif remaining < total_duration * 0.5:
            timer_class += " warning"
        
        st.markdown(f"""
        <div class="{timer_class}">
            ⏱️ Time Remaining: {time_str}
        </div>
        """, unsafe_allow_html=True)
        
        if remaining <= 0:
            st.session_state.interview_completed = True
            st.session_state.stage = 'feedback'
            st.rerun()
    
    st.title("💬 Behavioral Interview")
    
    # Current question or completion
    if st.session_state.current_question_idx < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_question_idx]
        question_num = st.session_state.current_question_idx + 1
        
        # Start question timer if not started
        if not st.session_state.question_timer_start:
            st.session_state.question_timer_start = datetime.now()
            if st.session_state.timer:
                st.session_state.timer.start_question()
        
        # Progress indicator
        progress = min(st.session_state.current_question_idx / len(st.session_state.questions), 1.0)
        st.progress(progress)
        
        # Current question display
        st.markdown(f"""
        <div class="current-question animate-fade-in">
            <div class="question-number">Question {question_num}/{len(st.session_state.questions)}</div>
            <div class="question-text">{current_question}</div>
            <div class="hears-reminder">
                <div class="hears-title">💡 HEARS Method Guide</div>
                <div class="hears-grid">
                    <div class="hears-item">
                        <span class="hears-letter">H</span>
                        <span>Headline: Brief situation summary</span>
                    </div>
                    <div class="hears-item">
                        <span class="hears-letter">E</span>
                        <span>Events: Specific challenges/context</span>
                    </div>
                    <div class="hears-item">
                        <span class="hears-letter">A</span>
                        <span>Actions: Your detailed actions</span>
                    </div>
                    <div class="hears-item">
                        <span class="hears-letter">R</span>
                        <span>Results: Measurable outcomes</span>
                    </div>
                    <div class="hears-item">
                        <span class="hears-letter">S</span>
                        <span>Significance: Skills & lessons learned</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # User response input
        with st.form(f"response_form_{st.session_state.current_question_idx}"):
            user_response = st.text_area(
                "Your Answer (use HEARS method):",
                placeholder="Provide a comprehensive answer covering Headline, Events, Actions, Results, and Significance...",
                height=200,
                key=f"response_{st.session_state.current_question_idx}"
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                submitted = st.form_submit_button("Submit Answer", type="primary", use_container_width=True)
            
            with col2:
                if st.form_submit_button("Skip Question", type="secondary", use_container_width=True):
                    st.session_state.question_responses.append({
                        'question': current_question,
                        'answer': '[Question Skipped]',
                        'question_number': st.session_state.current_question_idx + 1
                    })
                    
                    st.session_state.current_question_idx += 1
                    st.session_state.question_timer_start = None
                    st.rerun()
            
            if submitted and user_response.strip():
                # Record the Q&A pair
                st.session_state.question_responses.append({
                    'question': current_question,
                    'answer': user_response.strip(),
                    'question_number': st.session_state.current_question_idx + 1
                })
                
                # Generate individual feedback
                with st.spinner("🤖 Analyzing your response using HEARS methodology..."):
                    try:
                        individual_feedback = st.session_state.gemini_client.generate_individual_feedback(
                            current_question,
                            user_response.strip(),
                            st.session_state.job_details
                        )
                        st.session_state.individual_feedback.append({
                            'question_number': st.session_state.current_question_idx + 1,
                            'feedback': individual_feedback
                        })
                    except Exception as e:
                        st.session_state.individual_feedback.append({
                            'question_number': st.session_state.current_question_idx + 1,
                            'feedback': f"Unable to generate feedback: {str(e)}"
                        })
                
                # Move to next question
                st.session_state.current_question_idx += 1
                st.session_state.question_timer_start = None
                
                if st.session_state.current_question_idx >= len(st.session_state.questions):
                    st.session_state.interview_completed = True
                
                st.rerun()
    
    else:
        # Interview completed
        st.session_state.interview_completed = True
        
        st.markdown("""
        <div class="content-card animate-fade-in">
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎉</div>
                <h2 style="color: var(--accent-primary); margin-bottom: 1rem;">Interview Completed!</h2>
                <p style="font-size: 1.125rem; color: var(--text-secondary); margin-bottom: 2rem;">
                    Excellent work! You've completed all questions. Ready to get your comprehensive HEARS methodology feedback?
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Get My HEARS Feedback Report", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating your comprehensive HEARS methodology feedback report..."):
                    try:
                        overall_feedback = st.session_state.gemini_client.generate_overall_feedback(
                            st.session_state.question_responses,
                            st.session_state.job_details
                        )
                        st.session_state.overall_feedback = overall_feedback
                        st.session_state.stage = 'feedback'
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating feedback: {str(e)}")

def render_feedback_stage():
    """Render comprehensive HEARS feedback report."""
    st.title("📊 HEARS Methodology Feedback Report")
    
    if not st.session_state.question_responses:
        st.error("No interview responses available. Please complete the interview first.")
        return
    
    # Interview Summary
    st.markdown(f"""
    <div class="feedback-card">
        <h3 style="margin-bottom: 1rem; color: var(--accent-primary);">📋 Interview Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div><strong>Position:</strong> {st.session_state.job_details.get('job_title', 'N/A')}</div>
            <div><strong>Company:</strong> {st.session_state.job_details.get('company_name', 'N/A')}</div>
            <div><strong>Duration:</strong> {st.session_state.interview_duration} minutes</div>
            <div><strong>Questions:</strong> {len(st.session_state.question_responses)} of {st.session_state.num_questions}</div>
            <div><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual Question Feedback
    st.subheader("📝 Individual Question Analysis")
    
    for i, response in enumerate(st.session_state.question_responses):
        with st.expander(f"Question {response['question_number']}: Analysis", expanded=False):
            st.markdown(f"""
            <div class="feedback-individual">
                <h4>Question:</h4>
                <p>{response['question']}</p>
                
                <h4>Your Answer:</h4>
                <p style="background: #f8fafc; padding: 1rem; border-radius: 5px;">{response['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(st.session_state.individual_feedback):
                st.markdown(st.session_state.individual_feedback[i]['feedback'])
            else:
                st.info("Individual feedback not available for this question.")
    
    st.divider()
    
    # Overall HEARS Feedback
    st.subheader("🎯 Overall HEARS Analysis")
    
    if st.session_state.overall_feedback:
        st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.overall_feedback)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Generating overall feedback...")
        if st.button("Generate Overall Feedback"):
            with st.spinner("🤖 Creating comprehensive HEARS analysis..."):
                try:
                    overall_feedback = st.session_state.gemini_client.generate_overall_feedback(
                        st.session_state.question_responses,
                        st.session_state.job_details
                    )
                    st.session_state.overall_feedback = overall_feedback
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating overall feedback: {str(e)}")
    
    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Download Report", type="secondary"):
            report_content = f"""
# AI Interview Simulator - HEARS Methodology Report

**Interview Details:**
- Position: {st.session_state.job_details.get('job_title', 'N/A')}
- Company: {st.session_state.job_details.get('company_name', 'N/A')}
- Duration: {st.session_state.interview_duration} minutes
- Questions Completed: {len(st.session_state.question_responses)}/{st.session_state.num_questions}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Individual Question Analysis

"""
            
            for i, response in enumerate(st.session_state.question_responses):
                report_content += f"""
### Question {response['question_number']}

**Question:** {response['question']}

**Your Answer:** {response['answer']}

**HEARS Analysis:**
"""
                if i < len(st.session_state.individual_feedback):
                    report_content += f"{st.session_state.individual_feedback[i]['feedback']}\n\n"
                else:
                    report_content += "Individual feedback not available.\n\n"
            
            report_content += f"""
---

## Overall HEARS Analysis

{st.session_state.overall_feedback if st.session_state.overall_feedback else 'Overall feedback not generated.'}

---

*Generated by AI Interview Simulator using HEARS Methodology*
"""
            
            st.download_button(
                label="Download Complete HEARS Report",
                data=report_content,
                file_name=f"interview_hears_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
    
    with col2:
        if st.button("🔄 Practice Again", type="primary"):
            for key in ['stage', 'questions', 'current_question_idx', 'conversation', 'question_responses', 'individual_feedback', 'overall_feedback', 'interview_completed', 'timer', 'question_timer_start']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.stage = 'details'
            st.rerun()
    
    with col3:
        if st.button("📝 New Position", type="secondary"):
            for key in ['stage', 'job_details', 'interview_duration', 'num_questions', 'questions', 'current_question_idx', 'conversation', 'question_responses', 'individual_feedback', 'overall_feedback', 'interview_completed', 'timer', 'question_timer_start', 'duration_selected']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.interview_duration = 15
            st.session_state.num_questions = 3
            st.session_state.stage = 'details'
            st.rerun()
    
    with col4:
        if st.button("🏠 Start Over", type="secondary"):
            for key in list(st.session_state.keys()):
                if key != 'gemini_client':
                    del st.session_state[key]
            st.session_state.interview_duration = 15
            st.session_state.num_questions = 3
            st.rerun()

# Main Application
def main():
    """Main application entry point."""
    # CRITICAL: Load CSS first
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render components
    render_header()
    render_progress_stepper()
    
    # Route to appropriate stage
    if st.session_state.stage == 'upload':
        render_upload_stage()
    elif st.session_state.stage == 'details':
        render_details_stage()
    elif st.session_state.stage == 'interview':
        render_interview_stage()
    elif st.session_state.stage == 'feedback':
        render_feedback_stage()
    else:
        st.error("Unknown stage. Please restart the application.")

if __name__ == "__main__":
    main()
