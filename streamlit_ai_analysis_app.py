import streamlit as st
import pandas as pd
import json
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Dict, List, Optional, Tuple
import uuid
import traceback
import sys

# Optional imports with error handling
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.error("PyMuPDF not installed. PDF processing will be limited.")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR libraries not available. Image-based PDF processing will be limited.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="AI Model Analysis Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0;
        text-align: center;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .bard-badge { background-color: #e3f2fd; color: #1565c0; }
    .ais-badge { background-color: #f3e5f5; color: #7b1fa2; }
    .chatgpt-badge { background-color: #e8f5e8; color: #2e7d32; }
    .claude-badge { background-color: #fff3e0; color: #ef6c00; }
    .processing-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
    .error { background-color: #ffebee; border-left: 4px solid #f44336; }
    .warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .info { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .file-info {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AIModelAnalyzer:
    def __init__(self):
        self.supported_models = {
            'Bard 2.5 Pro': {'variants': ['bard 2.5 pro', 'bard pro', 'gemini 2.5 pro'], 'color': '#1565c0'},
            'AIS 2.5 Pro': {'variants': ['ais 2.5 pro', 'ais pro', 'anthropic 2.5 pro'], 'color': '#7b1fa2'},
            'AIS 2.5 Flash': {'variants': ['ais 2.5 flash', 'ais flash', 'anthropic flash'], 'color': '#9c27b0'},
            'cGPT o3': {'variants': ['cgpt o3', 'chatgpt o3', 'gpt o3', 'openai o3'], 'color': '#2e7d32'},
            'cGPT 4o': {'variants': ['cgpt 4o', 'chatgpt 4o', 'gpt 4o', 'gpt-4o'], 'color': '#1976d2'},
            'Claude': {'variants': ['claude', 'claude 3', 'claude sonnet', 'claude opus'], 'color': '#ef6c00'}
        }
        
        self.comparison_pairs = [
            ('Bard 2.5 Pro', 'AIS 2.5 Pro'),
            ('AIS 2.5 Pro', 'cGPT o3'),
            ('AIS 2.5 Flash', 'cGPT 4o'),
            ('Bard 2.5 Pro', 'cGPT o3'),
            ('Bard 2.5 Flash', 'cGPT 4o')
        ]

    def validate_file(self, file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if file is None:
            return False, "No file provided"
        
        # Check file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            return False, "File too large (max 10MB)"
        
        # Check file type
        if file.type not in ['application/pdf', 'text/plain']:
            return False, "Only PDF and TXT files are supported"
        
        return True, "File is valid"

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise Exception("PyMuPDF not available. Cannot process PDF files.")
        
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_content += f"\n--- PAGE {page_num + 1} ---\n"
                    text_content += text
                else:
                    # Try OCR if page has no text
                    if OCR_AVAILABLE:
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(image)
                        if ocr_text.strip():
                            text_content += f"\n--- PAGE {page_num + 1} (OCR) ---\n"
                            text_content += ocr_text
                    else:
                        text_content += f"\n--- PAGE {page_num + 1} (NO TEXT) ---\n"
            
            doc.close()
            return text_content
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def detect_ai_model(self, text: str) -> str:
        """Detect AI model from text using pattern matching"""
        if not text:
            return 'Unknown'
            
        text_lower = text.lower()
        
        for model, config in self.supported_models.items():
            for variant in config['variants']:
                if variant in text_lower:
                    return model
        
        # Additional pattern matching for specific interfaces
        if 'bard' in text_lower or 'gemini' in text_lower:
            if '2.5' in text_lower and 'pro' in text_lower:
                return 'Bard 2.5 Pro'
            elif 'flash' in text_lower:
                return 'Bard 2.5 Flash'
        
        if 'chatgpt' in text_lower or 'gpt' in text_lower:
            if 'o3' in text_lower:
                return 'cGPT o3'
            elif '4o' in text_lower:
                return 'cGPT 4o'
        
        if 'ais' in text_lower or 'anthropic' in text_lower:
            if 'flash' in text_lower:
                return 'AIS 2.5 Flash'
            elif 'pro' in text_lower:
                return 'AIS 2.5 Pro'
        
        return 'Unknown'

    def extract_conversation_from_text(self, text: str, page_num: int) -> List[Dict]:
        """Extract conversation data from raw text"""
        conversations = []
        
        if not text.strip():
            return conversations
        
        # Split by potential conversation boundaries
        sections = re.split(r'\n\s*\n', text)
        
        for section in sections:
            if len(section.strip()) < 10:  # Skip very short sections
                continue
                
            conversation_data = {
                'conversation_id': str(uuid.uuid4()),
                'page': page_num,
                'ai_model': 'Unknown',
                'user_prompt': '',
                'ai_response': '',
                'response_type': 'unknown',
                'conversation_context': '',
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'model_version': '',
                    'response_length': 0,
                    'has_code': False,
                    'has_math': False,
                    'processing_quality': 'good'
                }
            }
            
            # Detect AI model
            conversation_data['ai_model'] = self.detect_ai_model(section)
            
            # Extract user prompt and AI response
            lines = section.split('\n')
            user_prompt = ''
            ai_response = ''
            current_section = 'unknown'
            temp_text = ''
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for user indicators
                if re.search(r'^(user|you|human|question):', line, re.IGNORECASE):
                    if current_section == 'ai' and temp_text:
                        ai_response = temp_text
                    current_section = 'user'
                    temp_text = line.split(':', 1)[1].strip() if ':' in line else line
                elif re.search(r'^(assistant|ai|bot|bard|chatgpt|claude|ais):', line, re.IGNORECASE):
                    if current_section == 'user' and temp_text:
                        user_prompt = temp_text
                    current_section = 'ai'
                    temp_text = line.split(':', 1)[1].strip() if ':' in line else line
                elif current_section == 'user':
                    temp_text += ' ' + line
                elif current_section == 'ai':
                    temp_text += ' ' + line
                else:
                    # Try to detect implicit conversation structure
                    if '?' in line and len(line) > 10:
                        if not user_prompt:
                            user_prompt = line
                            current_section = 'user'
                    elif len(line) > 20 and current_section == 'user':
                        ai_response = line
                        current_section = 'ai'
                        temp_text = line
            
            # Finalize extraction
            if current_section == 'ai' and temp_text:
                ai_response = temp_text
            elif current_section == 'user' and temp_text:
                user_prompt = temp_text
            
            # Only add if we have meaningful content
            if user_prompt.strip() or ai_response.strip():
                conversation_data['user_prompt'] = user_prompt.strip()
                conversation_data['ai_response'] = ai_response.strip()
                
                # Classify response type
                conversation_data['response_type'] = self.classify_response_type(ai_response)
                
                # Extract metadata
                conversation_data['metadata']['response_length'] = len(ai_response)
                conversation_data['metadata']['has_code'] = bool(re.search(r'```|`[^`]+`|def |class |import |function', ai_response))
                conversation_data['metadata']['has_math'] = bool(re.search(r'\d+\.\d+|\+|\-|\*|\/|=|\^|\$.*\$', ai_response))
                
                # Extract model version if available
                version_match = re.search(r'(2\.5|4o|o3|pro|flash)', section.lower())
                if version_match:
                    conversation_data['metadata']['model_version'] = version_match.group(1)
                
                conversations.append(conversation_data)
        
        return conversations

    def classify_response_type(self, response: str) -> str:
        """Classify the type of AI response"""
        if not response:
            return 'empty'
        
        response_lower = response.lower()
        
        if 'step' in response_lower and ('1.' in response or '2.' in response):
            return 'step_by_step'
        elif '```' in response or 'def ' in response or 'class ' in response:
            return 'code'
        elif len(response) < 50:
            return 'direct'
        elif 'error' in response_lower or 'sorry' in response_lower or 'cannot' in response_lower:
            return 'error'
        elif any(word in response_lower for word in ['because', 'therefore', 'however', 'explanation']):
            return 'explanation'
        else:
            return 'informative'

    def process_file_content(self, file, progress_callback=None) -> List[Dict]:
        """Process file content and return structured conversation data"""
        conversations = []
        
        try:
            # Validate file
            is_valid, message = self.validate_file(file)
            if not is_valid:
                raise Exception(message)
            
            if progress_callback:
                progress_callback(10, f"Reading {file.name}...")
            
            # Extract text based on file type
            if file.type == 'application/pdf':
                file_bytes = file.read()
                text_content = self.extract_text_from_pdf(file_bytes)
            else:  # text/plain
                text_content = str(file.read(), "utf-8")
            
            if progress_callback:
                progress_callback(50, f"Processing text from {file.name}...")
            
            # Split text into pages
            pages = re.split(r'\n--- PAGE \d+ ---\n', text_content)
            
            for i, page_text in enumerate(pages, 1):
                if page_text.strip():
                    if progress_callback:
                        progress_callback(50 + (i / len(pages)) * 40, f"Processing page {i} of {len(pages)}...")
                    
                    page_conversations = self.extract_conversation_from_text(page_text, i)
                    conversations.extend(page_conversations)
            
            if progress_callback:
                progress_callback(100, f"Completed processing {file.name}")
            
            return conversations
            
        except Exception as e:
            raise Exception(f"Error processing {file.name}: {str(e)}")

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    return AIModelAnalyzer()

def display_file_info(file):
    """Display file information"""
    file_size = file.size / 1024  # Convert to KB
    size_unit = "KB"
    if file_size > 1024:
        file_size /= 1024
        size_unit = "MB"
    
    st.markdown(f"""
    <div class="file-info">
        <strong>📄 {file.name}</strong><br>
        Size: {file_size:.1f} {size_unit}<br>
        Type: {file.type}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Model Analysis Tool</h1>
        <p>Extract and analyze conversations from AI model comparison screenshots</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = get_analyzer()
    
    # System status
    with st.sidebar:
        st.title("🎯 Configuration")
        
        # System capabilities
        st.subheader("🔧 System Status")
        st.write(f"PDF Processing: {'✅' if PDF_AVAILABLE else '❌'}")
        st.write(f"OCR Support: {'✅' if OCR_AVAILABLE else '❌'}")
        st.write(f"Anthropic API: {'✅' if ANTHROPIC_AVAILABLE else '❌'}")
        
        # Model selection
        st.subheader("Supported AI Models")
        for model, config in analyzer.supported_models.items():
            st.markdown(f"<span class='model-badge' style='background-color: {config['color']}20; color: {config['color']}'>{model}</span>", unsafe_allow_html=True)
        
        st.subheader("📊 Comparison Pairs")
        for pair in analyzer.comparison_pairs:
            st.write(f"• {pair[0]} vs {pair[1]}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "📊 Analysis Dashboard", "🔍 Detailed View", "📥 Export Data"])
    
    with tab1:
        st.header("📁 Document Upload & Processing")
        
        # Sample data option
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Upload PDF files containing AI conversation screenshots for analysis")
        with col2:
            if st.button("📋 Load Sample Data", type="secondary"):
                st.session_state.sample_data = True
                st.session_state.processing_complete = False
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF files containing AI conversation screenshots"
        )
        
        # Display file information
        if uploaded_files:
            st.subheader("📋 Uploaded Files")
            for file in uploaded_files:
                display_file_info(file)
        
        # Process button
        if uploaded_files or st.session_state.get('sample_data', False):
            if st.button("🚀 Process Documents", type="primary"):
                st.session_state.processing_complete = False
                process_documents(uploaded_files, analyzer)
    
    with tab2:
        if st.session_state.get('processing_complete', False) and st.session_state.processed_data:
            show_analysis_dashboard(st.session_state.processed_data, analyzer)
        else:
            st.info("Please upload and process documents first.")
    
    with tab3:
        if st.session_state.get('processing_complete', False) and st.session_state.processed_data:
            show_detailed_view(st.session_state.processed_data, analyzer)
        else:
            st.info("Please upload and process documents first.")
    
    with tab4:
        if st.session_state.get('processing_complete', False) and st.session_state.processed_data:
            show_export_options(st.session_state.processed_data, analyzer)
        else:
            st.info("Please upload and process documents first.")

def process_documents(uploaded_files, analyzer):
    """Process uploaded documents or load sample data"""
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if st.session_state.get('sample_data', False):
                # Load sample data
                status_text.markdown('<div class="processing-status info">Loading sample data...</div>', unsafe_allow_html=True)
                progress_bar.progress(50)
                
                sample_data = get_sample_data()
                st.session_state.processed_data = sample_data
                st.session_state.sample_data = False
                st.session_state.processing_complete = True
                
                progress_bar.progress(100)
                status_text.markdown('<div class="processing-status success">✅ Sample data loaded successfully!</div>', unsafe_allow_html=True)
                
                # Show summary
                st.success(f"Loaded {len(sample_data)} sample conversations")
                return
            
            if not uploaded_files:
                st.error("Please upload files or load sample data.")
                return
            
            processed_conversations = []
            
            for i, file in enumerate(uploaded_files):
                file_progress = int((i / len(uploaded_files)) * 100)
                
                try:
                    def progress_callback(progress, message):
                        overall_progress = int(file_progress + (progress / len(uploaded_files)))
                        progress_bar.progress(min(overall_progress, 99))
                        status_text.markdown(f'<div class="processing-status info">{message}</div>', unsafe_allow_html=True)
                    
                    # Process the file
                    conversations = analyzer.process_file_content(file, progress_callback)
                    processed_conversations.extend(conversations)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
            
            progress_bar.progress(100)
            status_text.markdown('<div class="processing-status success">✅ Processing complete!</div>', unsafe_allow_html=True)
            
            # Store processed data
            st.session_state.processed_data = processed_conversations
            st.session_state.processing_complete = True
            
            # Show summary
            if processed_conversations:
                st.success(f"Processed {len(processed_conversations)} conversations from {len(uploaded_files)} files")
            else:
                st.warning("No conversations were extracted from the uploaded files. Please check the file format and content.")
                
        except Exception as e:
            st.error(f"Critical error during processing: {str(e)}")
            st.error("Please try again or contact support.")

def get_sample_data():
    """Return sample processed data"""
    return [
        {
            'conversation_id': 'conv_001',
            'page': 1,
            'ai_model': 'Bard 2.5 Pro',
            'user_prompt': 'What is 124.55 rounded to the nearest tenth?',
            'ai_response': '124.6',
            'response_type': 'direct',
            'conversation_context': 'Follow-up question about steps',
            'timestamp': '2024-01-15T10:30:00',
            'metadata': {
                'model_version': '2.5 Pro',
                'response_length': 5,
                'has_code': False,
                'has_math': True,
                'processing_quality': 'good'
            }
        },
        {
            'conversation_id': 'conv_002',
            'page': 1,
            'ai_model': 'AIS 2.5 Pro',
            'user_prompt': 'What is 124.55 rounded to the nearest tenth?',
            'ai_response': '124.6\n\nThe answer is 124.6. To round to the nearest tenth, I look at the hundredths digit (5). Since it\'s 5 or greater, I round the tenths digit up from 5 to 6.',
            'response_type': 'explanation',
            'conversation_context': 'Direct comparison with Bard',
            'timestamp': '2024-01-15T10:30:00',
            'metadata': {
                'model_version': '2.5 Pro',
                'response_length': 142,
                'has_code': False,
                'has_math': True,
                'processing_quality': 'good'
            }
        },
        {
            'conversation_id': 'conv_003',
            'page': 2,
            'ai_model': 'AIS 2.5 Pro',
            'user_prompt': 'Explain quantum computing in simple terms',
            'ai_response': 'Quantum computing is a revolutionary approach to computation that harnesses the principles of quantum mechanics. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or "qubits" that can exist in multiple states simultaneously through a phenomenon called superposition.',
            'response_type': 'explanation',
            'conversation_context': 'Technical explanation comparison',
            'timestamp': '2024-01-15T10:31:00',
            'metadata': {
                'model_version': '2.5 Pro',
                'response_length': 285,
                'has_code': False,
                'has_math': False,
                'processing_quality': 'good'
            }
        },
        {
            'conversation_id': 'conv_004',
            'page': 2,
            'ai_model': 'cGPT o3',
            'user_prompt': 'Explain quantum computing in simple terms',
            'ai_response': 'Think of quantum computing like this: regular computers are like a coin that\'s either heads or tails. Quantum computers are like a spinning coin that\'s both heads AND tails at the same time until it lands.',
            'response_type': 'explanation',
            'conversation_context': 'Analogy-based explanation',
            'timestamp': '2024-01-15T10:31:00',
            'metadata': {
                'model_version': 'o3',
                'response_length': 198,
                'has_code': False,
                'has_math': False,
                'processing_quality': 'good'
            }
        },
        {
            'conversation_id': 'conv_005',
            'page': 3,
            'ai_model': 'AIS 2.5 Flash',
            'user_prompt': 'Write a Python function to calculate factorial',
            'ai_response': 'Here\'s a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError("Factorial is not defined for negative numbers")\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```',
            'response_type': 'code',
            'conversation_context': 'Programming task',
            'timestamp': '2024-01-15T10:32:00',
            'metadata': {
                'model_version': '2.5 Flash',
                'response_length': 234,
                'has_code': True,
                'has_math': False,
                'processing_quality': 'good'
            }
        },
        {
            'conversation_id': 'conv_006',
            'page': 3,
            'ai_model': 'cGPT 4o',
            'user_prompt': 'Write a Python function to calculate factorial',
            'ai_response': 'Here are two approaches:\n\n```python\n# Recursive approach\ndef factorial_recursive(n):\n    if n <= 1:\n        return 1\n    return n * factorial_recursive(n - 1)\n\n# Iterative approach\ndef factorial_iterative(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n```\n\nThe iterative version is more efficient for large numbers.',
            'response_type': 'code',
            'conversation_context': 'Multiple solution approaches',
            'timestamp': '2024-01-15T10:32:00',
            'metadata': {
                'model_version': '4o',
                'response_length': 394,
                'has_code': True,
                'has_math': False,
                'processing_quality': 'good'
            }
        }
    ]

def show_analysis_dashboard(data, analyzer):
    """Show analysis dashboard with charts and metrics"""
    st.header("📊 Analysis Dashboard")
    
    if not data:
        st.warning("No data available for analysis.")
        return
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", len(df))
        
        with col2:
            st.metric("AI Models", df['ai_model'].nunique())
        
        with col3:
            st.metric("Pages Processed", df['page'].nunique())
        
        with col4:
            avg_response_length = df['metadata'].apply(lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0).mean()
            st.metric("Avg Response Length", f"{avg_response_length:.0f} chars")
        
        # Model distribution
        st.subheader("🤖 Model Distribution")
        model_counts = df['ai_model'].value_counts()
        
        if not model_counts.empty:
            fig_pie = px.pie(
                values=model_counts.values,
                names=model_counts.index,
                title="Distribution of AI Models"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Response types
        st.subheader("📝 Response Type Analysis")
        response_types = df['response_type'].value_counts()
        
        if not response_types.empty:
            fig_bar = px.bar(
                x=response_types.index,
                y=response_types.values,
                title="Response Types Distribution"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Model comparison
        st.subheader("⚡ Model Performance Comparison")
        
        # Calculate metrics by model
        model_metrics = df.groupby('ai_model').agg({
            'metadata': lambda x: sum(item.get('response_length', 0) if isinstance(item, dict) else 0 for item in x) / len(x),
            'response_type': lambda x: (x == 'code').sum(),
            'conversation_id': 'count'
        }).round(2)
        
        model_metrics.columns = ['Avg Response Length', 'Code Responses', 'Total Responses']
        
        st.dataframe(model_metrics, use_container_width=True)
        
        # Comparison pairs analysis
        st.subheader("🔄 Comparison Pairs Analysis")
        
        for pair in analyzer.comparison_pairs:
            model1, model2 = pair
            pair_data = df[df['ai_model'].isin([model1, model2])]
            
            if len(pair_data) > 0:
                with st.expander(f"{model1} vs {model2}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        model1_data = pair_data[pair_data['ai_model'] == model1]
                        if len(model1_data) > 0:
                            st.write(f"**{model1}**")
                            st.write(f"Responses: {len(model1_data)}")
                            avg_len = model1_data['metadata'].apply(lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0).mean()
                            st.write(f"Avg Length: {avg_len:.0f} chars")
                    
                    with col2:
                        model2_data = pair_data[pair_data['ai_model'] == model2]
                        if len(model2_data) > 0:
                            st.write(f"**{model2}**")
                            st.write(f"Responses: {len(model2_data)}")
                            avg_len = model2_data['metadata'].apply(lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0).mean()
                            st.write(f"Avg Length: {avg_len:.0f} chars")
    
    except Exception as e:
        st.error(f"Error displaying dashboard: {str(e)}")
        st.error("Please check your data format and try again.")

def show_detailed_view(data, analyzer):
    """Show detailed view of conversations"""
    st.header("🔍 Detailed Conversation View")
    
    if not data:
        st.warning("No data available for detailed view.")
        return
    
    try:
        df = pd.DataFrame(data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_filter = st.selectbox(
                "Filter by AI Model",
                options=['All'] + sorted(df['ai_model'].unique().tolist())
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Response Type",
                options=['All'] + sorted(df['response_type'].unique().tolist())
            )
        
        with col3:
            page_filter = st.selectbox(
                "Filter by Page",
                options=['All'] + sorted(df['page'].unique().tolist())
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if model_filter != 'All':
            filtered_df = filtered_df[filtered_df['ai_model'] == model_filter]
        
        if type_filter != 'All':
            filtered_df = filtered_df[filtered_df['response_type'] == type_filter]
        
        if page_filter != 'All':
            filtered_df = filtered_df[filtered_df['page'] == page_filter]
        
        # Search
        search_term = st.text_input("🔍 Search conversations...")
        if search_term:
            mask = (
                filtered_df['user_prompt'].str.contains(search_term, case=False, na=False) |
                filtered_df['ai_response'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        # Display conversations
        st.write(f"Showing {len(filtered_df)} conversations")
        
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Page {row['page']} - {row['ai_model']} - {row['response_type']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Metadata:**")
                    st.write(f"Model: {row['ai_model']}")
                    metadata = row['metadata'] if isinstance(row['metadata'], dict) else {}
                    st.write(f"Version: {metadata.get('model_version', 'N/A')}")
                    st.write(f"Response Length: {metadata.get('response_length', 0)} chars")
                    st.write(f"Has Code: {metadata.get('has_code', False)}")
                    st.write(f"Has Math: {metadata.get('has_math', False)}")
                
                with col2:
                    st.write("**User Prompt:**")
                    st.write(row['user_prompt'])
                    
                    st.write("**AI Response:**")
                    st.write(row['ai_response'])
                    
                    if row['conversation_context']:
                        st.write("**Context:**")
                        st.write(row['conversation_context'])
    
    except Exception as e:
        st.error(f"Error displaying detailed view: {str(e)}")

def show_export_options(data, analyzer):
    """Show export options"""
    st.header("📥 Export Data")
    
    if not data:
        st.warning("No data available for export.")
        return
    
    try:
        df = pd.DataFrame(data)
        
        # Export formats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Export Formats")
            
            # JSON export
            if st.button("📄 Export as JSON"):
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # CSV export
            if st.button("📊 Export as CSV"):
                # Flatten the data for CSV
                flat_data = []
                for item in data:
                    metadata = item.get('metadata', {}) if isinstance(item.get('metadata'), dict) else {}
                    flat_item = {
                        'conversation_id': item.get('conversation_id', ''),
                        'page': item.get('page', 0),
                        'ai_model': item.get('ai_model', ''),
                        'user_prompt': item.get('user_prompt', ''),
                        'ai_response': item.get('ai_response', ''),
                        'response_type': item.get('response_type', ''),
                        'conversation_context': item.get('conversation_context', ''),
                        'timestamp': item.get('timestamp', ''),
                        'model_version': metadata.get('model_version', ''),
                        'response_length': metadata.get('response_length', 0),
                        'has_code': metadata.get('has_code', False),
                        'has_math': metadata.get('has_math', False)
                    }
                    flat_data.append(flat_item)
                
                csv_df = pd.DataFrame(flat_data)
                csv_str = csv_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("📋 Summary Report")
            
            # Generate summary report
            if st.button("📋 Generate Summary Report"):
                report = generate_summary_report(data, analyzer)
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"ai_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        # Preview data
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in export options: {str(e)}")

def generate_summary_report(data, analyzer):
    """Generate a summary report"""
    try:
        df = pd.DataFrame(data)
        
        report = f"""
AI Model Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Conversations: {len(df)}
Pages Processed: {df['page'].nunique()}
AI Models Analyzed: {', '.join(df['ai_model'].unique())}

MODEL DISTRIBUTION
==================
{df['ai_model'].value_counts().to_string()}

RESPONSE TYPE ANALYSIS
======================
{df['response_type'].value_counts().to_string()}

MODEL PERFORMANCE METRICS
=========================
"""
        
        for model in df['ai_model'].unique():
            model_data = df[df['ai_model'] == model]
            avg_length = model_data['metadata'].apply(lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0).mean()
            code_responses = model_data['metadata'].apply(lambda x: x.get('has_code', False) if isinstance(x, dict) else False).sum()
            math_responses = model_data['metadata'].apply(lambda x: x.get('has_math', False) if isinstance(x, dict) else False).sum()
            
            report += f"""
{model}:
  - Total Responses: {len(model_data)}
  - Average Response Length: {avg_length:.0f} characters
  - Code Responses: {code_responses}
  - Math Responses: {math_responses}
  - Response Types: {model_data['response_type'].value_counts().to_dict()}
"""
        
        report += f"""

COMPARISON PAIRS ANALYSIS
=========================
"""
        
        for pair in analyzer.comparison_pairs:
            model1, model2 = pair
            pair_data = df[df['ai_model'].isin([model1, model2])]
            
            if len(pair_data) > 0:
                report += f"""
{model1} vs {model2}:
  - Total conversations: {len(pair_data)}
  - {model1} responses: {len(pair_data[pair_data['ai_model'] == model1])}
  - {model2} responses: {len(pair_data[pair_data['ai_model'] == model2])}
"""
        
        return report
    
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    main()
