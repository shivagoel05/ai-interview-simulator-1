# AI Interview Simulator - Enhanced Chatbot Style Interface
# Improved UI similar to Interviews.ai with conversational flow

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
        # Enhanced fallback CSS with chatbot styling
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        :root {
          --bg-primary: #FFF9F0;
          --bg-chat: #FEFCF8;
          --accent-primary: #F59E0B;
          --accent-secondary: #FBB042;
          --text-primary: #374151;
          --text-secondary: #6B7280;
          --border-primary: #E5E7EB;
        }
        
        * { font-family: 'Plus Jakarta Sans', sans-serif; }
        .main .block-container { background: var(--bg-primary); padding: 1rem; }
        #MainMenu, footer, header { visibility: hidden; }
        
        .chat-container {
          max-height: 70vh;
          overflow-y: auto;
          padding: 1rem;
          background: var(--bg-chat);
          border-radius: 16px;
          margin-bottom: 1rem;
          border: 1px solid var(--border-primary);
        }
        
        .message {
          margin-bottom: 1rem;
          animation: fadeInUp 0.5s ease-out;
        }
        
        .message.ai {
          text-align: left;
        }
        
        .message.user {
          text-align: right;
        }
        
        .message-bubble {
          display: inline-block;
          max-width: 70%;
          padding: 1rem 1.5rem;
          border-radius: 20px;
          font-size: 0.95rem;
          line-height: 1.5;
        }
        
        .message.ai .message-bubble {
          background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%);
          color: white;
          border-bottom-left-radius: 8px;
        }
        
        .message.user .message-bubble {
          background: #E5E7EB;
          color: var(--text-primary);
          border-bottom-right-radius: 8px;
        }
        
        .typing-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 1rem;
          color: var(--text-secondary);
        }
        
        .typing-dots {
          display: flex;
          gap: 0.25rem;
        }
        
        .dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent-primary);
          animation: typing 1.5s infinite;
        }
        
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
          0%, 60%, 100% { transform: scale(1); opacity: 0.5; }
          30% { transform: scale(1.2); opacity: 1; }
        }
        
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .app-header {
          background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%);
          color: white;
          padding: 2rem;
          border-radius: 16px;
          text-align: center;
          margin-bottom: 2rem;
          box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
        }
        
        .stButton > button {
          background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%) !important;
          color: white !important;
          border: none !important;
          border-radius: 50px !important;
          padding: 0.75rem 2rem !important;
          font-weight: 600 !important;
          transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
          transform: translateY(-2px) !important;
          box-shadow: 0 8px 25px rgba(245, 158, 11, 0.4) !important;
        }
        </style>
        """, unsafe_allow_html=True)

# Gemini API Configuration (same as before)
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

# File Processing Functions (same as before)
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

# Timer functionality (same as before)
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
        'chat_messages': [],
        'question_responses': [],
        'individual_feedback': [],
        'overall_feedback': "",
        'interview_completed': False,
        'timer': None,
        'question_timer_start': None,
        'gemini_client': None,
        'duration_selected': False,
        'waiting_for_response': False,
        'current_question_displayed': False
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
        <p>Master behavioral interviews with AI-powered conversation and HEARS methodology feedback</p>
    </div>
    """, unsafe_allow_html=True)

def render_modern_progress():
    """Render modern progress indicator."""
    stages = ['upload', 'details', 'interview', 'feedback']
    stage_names = ['Upload Resume', 'Setup Interview', 'Practice Interview', 'Get Feedback']
    stage_icons = ['📄', '⚙️', '💬', '📊']
    current_stage_idx = stages.index(st.session_state.stage)
    
    # Create progress bar
    progress_percentage = (current_stage_idx / (len(stages) - 1)) * 100
    
    st.markdown(f"""
    <div style="background: white; border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #374151; font-size: 1.25rem;">Interview Progress</h3>
            <span style="color: #6B7280; font-size: 0.875rem;">Step {current_stage_idx + 1} of {len(stages)}</span>
        </div>
        
        <div style="background: #F3F4F6; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 1.5rem;">
            <div style="background: linear-gradient(90deg, #F59E0B 0%, #FBB042 100%); height: 100%; width: {progress_percentage}%; transition: width 0.6s ease;"></div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat({len(stages)}, 1fr); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    for i, (stage, name, icon) in enumerate(zip(stages, stage_names, stage_icons)):
        status_class = ""
        if i < current_stage_idx:
            status_class = "completed"
        elif i == current_stage_idx:
            status_class = "active"
        else:
            status_class = "pending"
        
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="
                    width: 40px; 
                    height: 40px; 
                    border-radius: 50%; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    margin: 0 auto 0.5rem; 
                    font-size: 1.25rem;
                    {
                        'background: #10B981; color: white;' if status_class == 'completed' else
                        'background: linear-gradient(135deg, #F59E0B 0%, #FBB042 100%); color: white;' if status_class == 'active' else
                        'background: #F3F4F6; color: #9CA3AF;'
                    }
                ">
                    {'✓' if status_class == 'completed' else icon}
                </div>
                <div style="font-size: 0.75rem; color: {'#10B981' if status_class == 'completed' else '#F59E0B' if status_class == 'active' else '#9CA3AF'}; font-weight: 600;">
                    {name}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_upload_stage():
    """Render resume upload stage with modern design."""
    render_modern_progress()
    
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 3rem; box-shadow: 0 8px 32px rgba(0,0,0,0.08); max-width: 800px; margin: 0 auto;">
        <div style="text-align: center; margin-bottom: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
            <h2 style="color: #374151; margin-bottom: 1rem; font-size: 2rem;">Upload Your Resume</h2>
            <p style="color: #6B7280; font-size: 1.125rem; line-height: 1.6;">
                Start your interview practice by uploading your resume. Our AI will analyze your experience 
                and create personalized behavioral interview questions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'doc', 'docx', 'txt'],
            help="Maximum file size: 10MB. Supported formats: PDF, DOC, DOCX, TXT",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Analyzing your resume..."):
                success, result = FileProcessor.process_resume_file(uploaded_file)
                
                if success:
                    st.session_state.resume_text = result
                    
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: #065F46; padding: 1.5rem; border-radius: 12px; margin: 2rem 0; display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 2rem;">✅</span>
                        <div>
                            <div style="font-weight: 700; font-size: 1.125rem;">Resume Successfully Processed!</div>
                            <div style="font-size: 0.875rem; opacity: 0.8;">Your resume has been analyzed and is ready for interview preparation.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 Resume Preview", expanded=False):
                        preview_text = result[:500] + "..." if len(result) > 500 else result
                        st.markdown(f"""
                        <div style="background: #F8FAFC; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3B82F6; font-family: monospace; font-size: 0.875rem; line-height: 1.6;">
                            {preview_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Continue to Interview Setup →", type="primary", use_container_width=True):
                        st.session_state.stage = 'details'
                        st.rerun()
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); color: #991B1B; padding: 1.5rem; border-radius: 12px; margin: 2rem 0; display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 2rem;">❌</span>
                        <div>
                            <div style="font-weight: 700;">Upload Error</div>
                            <div style="font-size: 0.875rem;">{result}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_details_stage():
    """Render job details and setup stage."""
    render_modern_progress()
    
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 3rem; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 2rem;">
        <div style="text-align: center; margin-bottom: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚙️</div>
            <h2 style="color: #374151; margin-bottom: 1rem; font-size: 2rem;">Setup Your Interview</h2>
            <p style="color: #6B7280; font-size: 1.125rem; line-height: 1.6;">
                Configure your practice session and provide job details for personalized questions.
            </p>
        </div>
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
    
    cols = st.columns(len(duration_options))
    for i, option in enumerate(duration_options):
        with cols[i]:
            if st.button(
                f"{option['icon']}\n\n**{option['label']}**\n\n{option['duration']} min • {option['questions']} questions\n\n{option['desc']}", 
                key=f"dur_{i}", 
                use_container_width=True
            ):
                st.session_state.interview_duration = option["duration"]
                st.session_state.num_questions = option["questions"]
                st.session_state.duration_selected = True
                st.rerun()
    
    if st.session_state.duration_selected:
        st.success(f"✅ Selected: {st.session_state.interview_duration} minutes ({st.session_state.num_questions} questions)")
    
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
        
        submitted = st.form_submit_button("🚀 Start Interview", type="primary", use_container_width=True)
        
        if submitted:
            if not job_title or not company_name or not job_description:
                st.error("❌ Please fill in all required fields (marked with *)")
            elif not st.session_state.duration_selected:
                st.error("❌ Please select an interview duration first")
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
                
                with st.spinner(f"🤖 Generating {st.session_state.num_questions} personalized questions..."):
                    try:
                        questions = st.session_state.gemini_client.generate_questions(
                            st.session_state.resume_text,
                            job_details,
                            st.session_state.num_questions
                        )
                        
                        st.session_state.questions = questions
                        st.session_state.timer = InterviewTimer(st.session_state.interview_duration)
                        st.session_state.stage = 'interview'
                        st.session_state.chat_messages = []  # Reset chat
                        
                        # Add welcome message
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"Welcome to your {st.session_state.interview_duration}-minute interview practice for **{job_title}** at **{company_name}**! 🎉\n\nI'll be asking you {st.session_state.num_questions} behavioral questions. Remember to use the **HEARS method** for each response:\n\n**H**eadline - Brief situation summary\n**E**vents - Specific challenges/context  \n**A**ctions - Your detailed actions\n**R**esults - Measurable outcomes\n**S**ignificance - Skills & lessons learned\n\nReady to begin? Let's start with your first question! 🚀",
                            "timestamp": datetime.now()
                        })
                        
                        st.success("🎉 Questions generated! Starting your interview...")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating questions: {str(e)}")

def render_chat_message(message):
    """Render a single chat message."""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", datetime.now())
    
    if role == "assistant":
        st.markdown(f"""
        <div class="message ai">
            <div class="message-bubble">
                {content}
            </div>
            <div style="font-size: 0.75rem; color: #9CA3AF; margin-top: 0.5rem;">
                AI Interviewer • {timestamp.strftime('%H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message user">
            <div class="message-bubble">
                {content}
            </div>
            <div style="font-size: 0.75rem; color: #9CA3AF; margin-top: 0.5rem;">
                You • {timestamp.strftime('%H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_typing_indicator():
    """Show typing indicator."""
    st.markdown("""
    <div class="typing-indicator">
        <div style="font-size: 1.5rem;">🤖</div>
        <div>
            <div>AI Interviewer is thinking...</div>
            <div class="typing-dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_interview_stage():
    """Render chatbot-style interview interface."""
    render_modern_progress()
    
    # Timer display
    if st.session_state.timer:
        if not st.session_state.timer.start_time:
            st.session_state.timer.start_interview()
        
        remaining = st.session_state.timer.get_remaining_time()
        total_duration = st.session_state.interview_duration * 60
        time_str = st.session_state.timer.format_time(remaining)
        
        timer_color = "#10B981"  # Green
        if remaining < total_duration * 0.25:
            timer_color = "#EF4444"  # Red
        elif remaining < total_duration * 0.5:
            timer_color = "#F59E0B"  # Orange
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background: {timer_color}; color: white; padding: 1rem 2rem; border-radius: 50px; text-align: center; margin-bottom: 2rem; font-weight: 700; font-size: 1.125rem;">
                ⏱️ Time Remaining: {time_str}
            </div>
            """, unsafe_allow_html=True)
        
        if remaining <= 0:
            st.session_state.interview_completed = True
            st.session_state.stage = 'feedback'
            st.rerun()
    
    # Main interview interface
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 2px solid #F3F4F6;">
            <div style="font-size: 3rem;">🤖</div>
            <div>
                <h2 style="margin: 0; color: #374151;">AI Interview Simulator</h2>
                <p style="margin: 0; color: #6B7280;">Behavioral Interview Practice</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            render_chat_message(message)
        
        # Show current question if needed
        if (st.session_state.current_question_idx < len(st.session_state.questions) and 
            not st.session_state.current_question_displayed and 
            not st.session_state.waiting_for_response):
            
            current_question = st.session_state.questions[st.session_state.current_question_idx]
            question_num = st.session_state.current_question_idx + 1
            
            # Add question to chat
            question_message = {
                "role": "assistant",
                "content": f"**Question {question_num}/{len(st.session_state.questions)}:**\n\n{current_question}\n\n*Please provide a detailed response using the HEARS method. Take your time to think through your answer.*",
                "timestamp": datetime.now()
            }
            
            st.session_state.chat_messages.append(question_message)
            st.session_state.current_question_displayed = True
            st.session_state.question_timer_start = datetime.now()
            if st.session_state.timer:
                st.session_state.timer.start_question()
            
            render_chat_message(question_message)
        
        # Show typing indicator if waiting
        if st.session_state.waiting_for_response:
            render_typing_indicator()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    if (st.session_state.current_question_idx < len(st.session_state.questions) and 
        st.session_state.current_question_displayed and 
        not st.session_state.waiting_for_response):
        
        with st.form("response_form", clear_on_submit=True):
            user_response = st.text_area(
                "Your Response:",
                placeholder="Type your detailed response here using the HEARS method...",
                height=120,
                key="current_response"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Send Response", type="primary", use_container_width=True)
            
            if submitted and user_response.strip():
                # Add user message to chat
                user_message = {
                    "role": "user",
                    "content": user_response.strip(),
                    "timestamp": datetime.now()
                }
                st.session_state.chat_messages.append(user_message)
                
                # Record response
                current_question = st.session_state.questions[st.session_state.current_question_idx]
                st.session_state.question_responses.append({
                    'question': current_question,
                    'answer': user_response.strip(),
                    'question_number': st.session_state.current_question_idx + 1
                })
                
                # Set waiting state
                st.session_state.waiting_for_response = True
                st.session_state.current_question_displayed = False
                st.session_state.current_question_idx += 1
                
                st.rerun()
    
    # Process AI response
    if st.session_state.waiting_for_response:
        time.sleep(1)  # Simulate thinking time
        
        # Generate feedback message
        if st.session_state.current_question_idx < len(st.session_state.questions):
            # More questions to go
            feedback_msg = {
                "role": "assistant", 
                "content": f"Thank you for that response! I can see you've covered some good ground there. Let's continue with the next question.\n\n*Moving to question {st.session_state.current_question_idx + 1}...*",
                "timestamp": datetime.now()
            }
        else:
            # Interview complete
            feedback_msg = {
                "role": "assistant",
                "content": "🎉 **Congratulations!** You've completed all the interview questions!\n\nThank you for sharing your experiences with me. You've provided some great insights into your background and capabilities.\n\nI'm now preparing your comprehensive **HEARS methodology feedback report** which will include:\n\n✅ Individual question analysis\n✅ Overall performance assessment\n✅ Specific improvement recommendations\n✅ Skills demonstration scores\n\nReady to see how you did? Let's review your feedback! 📊",
                "timestamp": datetime.now()
            }
            st.session_state.interview_completed = True
        
        st.session_state.chat_messages.append(feedback_msg)
        st.session_state.waiting_for_response = False
        
        # Generate individual feedback in background
        if len(st.session_state.question_responses) > len(st.session_state.individual_feedback):
            try:
                last_response = st.session_state.question_responses[-1]
                individual_feedback = st.session_state.gemini_client.generate_individual_feedback(
                    last_response['question'],
                    last_response['answer'],
                    st.session_state.job_details
                )
                st.session_state.individual_feedback.append({
                    'question_number': last_response['question_number'],
                    'feedback': individual_feedback
                })
            except Exception as e:
                st.session_state.individual_feedback.append({
                    'question_number': len(st.session_state.individual_feedback) + 1,
                    'feedback': f"Unable to generate feedback: {str(e)}"
                })
        
        st.rerun()
    
    # Complete interview button
    if st.session_state.interview_completed:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 View My HEARS Feedback Report", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating comprehensive feedback..."):
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
    """Render comprehensive feedback with modern design."""
    render_modern_progress()
    
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 3rem; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 2rem;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #374151; margin-bottom: 1rem; font-size: 2rem;">Your HEARS Feedback Report</h2>
            <p style="color: #6B7280; font-size: 1.125rem;">
                Comprehensive analysis of your interview performance
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.question_responses:
        st.error("No interview responses available. Please complete the interview first.")
        return
    
    # Interview Summary Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); border-radius: 16px; padding: 2rem; margin-bottom: 2rem; border-left: 6px solid #F59E0B;">
        <h3 style="margin-bottom: 1.5rem; color: #92400E; display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">📋</span>
            Interview Summary
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; font-size: 0.95rem;">
            <div><strong>Position:</strong> {st.session_state.job_details.get('job_title', 'N/A')}</div>
            <div><strong>Company:</strong> {st.session_state.job_details.get('company_name', 'N/A')}</div>
            <div><strong>Duration:</strong> {st.session_state.interview_duration} minutes</div>
            <div><strong>Questions:</strong> {len(st.session_state.question_responses)} of {st.session_state.num_questions}</div>
            <div><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall Feedback Section
    if st.session_state.overall_feedback:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 6px solid #10B981;">
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.overall_feedback)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Individual Question Analysis
    st.markdown("## 📝 Individual Question Analysis")
    
    for i, response in enumerate(st.session_state.question_responses):
        with st.expander(f"Question {response['question_number']}: Detailed Analysis", expanded=False):
            st.markdown(f"""
            <div style="background: #F8FAFC; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
                <h4 style="color: #374151; margin-bottom: 1rem;">❓ Question:</h4>
                <p style="font-style: italic; color: #4B5563; margin-bottom: 1.5rem;">{response['question']}</p>
                
                <h4 style="color: #374151; margin-bottom: 1rem;">💬 Your Answer:</h4>
                <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6;">
                    {response['answer']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(st.session_state.individual_feedback):
                st.markdown("#### 🎯 HEARS Analysis:")
                st.markdown(st.session_state.individual_feedback[i]['feedback'])
            else:
                st.info("Individual feedback not available for this question.")
    
    # Action Buttons
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Download Report", type="secondary", use_container_width=True):
            # Generate download content
            report_content = f"""
# AI Interview Simulator - HEARS Methodology Report

**Interview Details:**
- Position: {st.session_state.job_details.get('job_title', 'N/A')}
- Company: {st.session_state.job_details.get('company_name', 'N/A')}
- Duration: {st.session_state.interview_duration} minutes
- Questions Completed: {len(st.session_state.question_responses)}/{st.session_state.num_questions}
- Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

## Overall HEARS Analysis

{st.session_state.overall_feedback if st.session_state.overall_feedback else 'Overall feedback not generated.'}

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

*Generated by AI Interview Simulator using HEARS Methodology*
*Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*
"""
            
            st.download_button(
                label="📄 Download Complete Report",
                data=report_content,
                file_name=f"interview_hears_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    with col2:
        if st.button("🔄 Practice Again", type="primary", use_container_width=True):
            # Reset for same job
            keys_to_reset = ['stage', 'questions', 'current_question_idx', 'chat_messages', 'question_responses', 
                           'individual_feedback', 'overall_feedback', 'interview_completed', 'timer', 
                           'question_timer_start', 'waiting_for_response', 'current_question_displayed']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.stage = 'details'
            st.rerun()
    
    with col3:
        if st.button("📝 New Position", type="secondary", use_container_width=True):
            # Reset for different job
            keys_to_reset = ['stage', 'job_details', 'interview_duration', 'num_questions', 'questions', 
                           'current_question_idx', 'chat_messages', 'question_responses', 'individual_feedback', 
                           'overall_feedback', 'interview_completed', 'timer', 'question_timer_start', 
                           'duration_selected', 'waiting_for_response', 'current_question_displayed']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.interview_duration = 15
            st.session_state.num_questions = 3
            st.session_state.stage = 'details'
            st.rerun()
    
    with col4:
        if st.button("🏠 Start Over", type="secondary", use_container_width=True):
            # Complete reset
            for key in list(st.session_state.keys()):
                if key != 'gemini_client':
                    del st.session_state[key]
            st.session_state.interview_duration = 15
            st.session_state.num_questions = 3
            st.rerun()

# Main Application
def main():
    """Main application entry point."""
    # Load CSS first
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
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
