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

    def detect_ai_model(self, text: str) -> str:
        """Detect AI model from text using pattern matching"""
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

    def extract_conversation_from_text(self, text: str, page_num: int) -> Dict:
        """Extract conversation data from raw text"""
        lines = text.split('\n')
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
        conversation_data['ai_model'] = self.detect_ai_model(text)
        
        # Extract user prompt and AI response
        user_prompt = ''
        ai_response = ''
        context = ''
        
        # Pattern matching for different conversation formats
        current_section = 'unknown'
        temp_text = ''
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for user indicators
            if re.search(r'^(user|you|human|question):', line, re.IGNORECASE):
                current_section = 'user'
                temp_text = line.split(':', 1)[1].strip() if ':' in line else line
            elif re.search(r'^(assistant|ai|bot|' + conversation_data['ai_model'].lower() + '):', line, re.IGNORECASE):
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
                if len(line) > 10 and '?' in line:
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
        
        conversation_data['user_prompt'] = user_prompt.strip()
        conversation_data['ai_response'] = ai_response.strip()
        
        # Classify response type
        conversation_data['response_type'] = self.classify_response_type(ai_response)
        
        # Extract metadata
        conversation_data['metadata']['response_length'] = len(ai_response)
        conversation_data['metadata']['has_code'] = bool(re.search(r'```|`[^`]+`|def |class |import |function', ai_response))
        conversation_data['metadata']['has_math'] = bool(re.search(r'\d+\.\d+|\+|\-|\*|\/|=|\^|\$.*\$', ai_response))
        
        # Extract model version if available
        version_match = re.search(r'(2\.5|4o|o3|pro|flash)', text.lower())
        if version_match:
            conversation_data['metadata']['model_version'] = version_match.group(1)
        
        return conversation_data

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

    def process_pdf_text(self, text_content: str) -> List[Dict]:
        """Process extracted PDF text and return structured conversation data"""
        # Split text into pages (assuming page breaks are marked)
        pages = text_content.split('\n--- PAGE BREAK ---\n')
        
        conversations = []
        for i, page_text in enumerate(pages, 1):
            if page_text.strip():
                conversation = self.extract_conversation_from_text(page_text, i)
                if conversation['user_prompt'] or conversation['ai_response']:
                    conversations.append(conversation)
        
        return conversations

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    return AIModelAnalyzer()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Model Analysis Tool</h1>
        <p>Extract and analyze conversations from AI model comparison screenshots</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = get_analyzer()
    
    # Sidebar
    st.sidebar.title("🎯 Configuration")
    
    # Model selection
    st.sidebar.subheader("Supported AI Models")
    for model, config in analyzer.supported_models.items():
        st.sidebar.markdown(f"<span class='model-badge' style='background-color: {config['color']}20; color: {config['color']}'>{model}</span>", unsafe_allow_html=True)
    
    st.sidebar.subheader("📊 Comparison Pairs")
    for pair in analyzer.comparison_pairs:
        st.sidebar.write(f"• {pair[0]} vs {pair[1]}")
    
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
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF files containing AI conversation screenshots"
        )
        
        if uploaded_files or st.session_state.get('sample_data', False):
            if st.button("🚀 Process Documents", type="primary"):
                process_documents(uploaded_files, analyzer)
    
    with tab2:
        if 'processed_data' in st.session_state:
            show_analysis_dashboard(st.session_state.processed_data, analyzer)
        else:
            st.info("Please upload and process documents first.")
    
    with tab3:
        if 'processed_data' in st.session_state:
            show_detailed_view(st.session_state.processed_data, analyzer)
        else:
            st.info("Please upload and process documents first.")
    
    with tab4:
        if 'processed_data' in st.session_state:
            show_export_options(st.session_state.processed_data, analyzer)
        else:
            st.info("Please upload and process documents first.")

def process_documents(uploaded_files, analyzer):
    """Process uploaded documents or load sample data"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if st.session_state.get('sample_data', False):
        # Load sample data
        status_text.markdown('<div class="processing-status info">Loading sample data...</div>', unsafe_allow_html=True)
        sample_data = get_sample_data()
        st.session_state.processed_data = sample_data
        st.session_state.sample_data = False
        progress_bar.progress(100)
        status_text.markdown('<div class="processing-status success">✅ Sample data loaded successfully!</div>', unsafe_allow_html=True)
        return
    
    if not uploaded_files:
        st.error("Please upload files or load sample data.")
        return
    
    processed_conversations = []
    
    for i, file in enumerate(uploaded_files):
        progress = int((i / len(uploaded_files)) * 100)
        progress_bar.progress(progress)
        status_text.markdown(f'<div class="processing-status info">Processing {file.name}...</div>', unsafe_allow_html=True)
        
        try:
            # Read file content
            if file.type == 'text/plain':
                text_content = str(file.read(), "utf-8")
            else:
                # For PDF files, simulate text extraction
                text_content = simulate_pdf_extraction(file.name)
            
            # Process the text
            conversations = analyzer.process_pdf_text(text_content)
            processed_conversations.extend(conversations)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    progress_bar.progress(100)
    status_text.markdown('<div class="processing-status success">✅ Processing complete!</div>', unsafe_allow_html=True)
    
    # Store processed data
    st.session_state.processed_data = processed_conversations
    
    # Show summary
    st.success(f"Processed {len(processed_conversations)} conversations from {len(uploaded_files)} files")

def simulate_pdf_extraction(filename):
    """Simulate PDF text extraction with realistic AI conversation data"""
    sample_conversations = [
        """
        Page 1 - Bard 2.5 Pro vs AIS 2.5 Pro Comparison

        User: What is 124.55 rounded to the nearest tenth?

        Bard 2.5 Pro: 124.6

        User: Can you show me the steps?

        Bard 2.5 Pro: Of course! Here are the steps to round 124.55 to the nearest tenth:

        1. Identify the tenths place: In 124.55, the tenths place is 5
        2. Look at the hundredths place: The next digit is 5
        3. Apply rounding rule: Since 5 ≥ 5, we round up
        4. Result: 124.6

        AIS 2.5 Pro: 124.6

        The answer is 124.6. To round to the nearest tenth, I look at the hundredths digit (5). Since it's 5 or greater, I round the tenths digit up from 5 to 6.
        """,
        """
        Page 2 - AIS 2.5 Pro vs cGPT o3 Comparison

        User: Explain quantum computing in simple terms

        AIS 2.5 Pro: Quantum computing is a revolutionary approach to computation that harnesses the principles of quantum mechanics. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or "qubits" that can exist in multiple states simultaneously through a phenomenon called superposition.

        Key concepts:
        - Superposition: Qubits can be in multiple states at once
        - Entanglement: Qubits can be correlated in ways that classical physics doesn't allow
        - Quantum interference: Allows quantum algorithms to amplify correct answers

        cGPT o3: Think of quantum computing like this: regular computers are like a coin that's either heads or tails. Quantum computers are like a spinning coin that's both heads AND tails at the same time until it lands.

        This "spinning coin" state lets quantum computers try many solutions simultaneously, making them potentially much faster for certain problems like breaking codes or simulating molecules.
        """,
        """
        Page 3 - AIS 2.5 Flash vs cGPT 4o Comparison

        User: Write a Python function to calculate factorial

        AIS 2.5 Flash: Here's a Python function to calculate factorial:

        ```python
        def factorial(n):
            if n < 0:
                raise ValueError("Factorial is not defined for negative numbers")
            if n == 0 or n == 1:
                return 1
            return n * factorial(n - 1)
        ```

        cGPT 4o: Here are two approaches:

        ```python
        # Recursive approach
        def factorial_recursive(n):
            if n <= 1:
                return 1
            return n * factorial_recursive(n - 1)

        # Iterative approach
        def factorial_iterative(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        ```

        The iterative version is more efficient for large numbers.
        """,
        """
        Page 4 - Bard 2.5 Pro vs cGPT o3 Complex Reasoning

        User: If a train leaves Station A at 2:00 PM traveling at 60 mph, and another train leaves Station B at 2:30 PM traveling at 80 mph toward Station A, and the stations are 280 miles apart, when will they meet?

        Bard 2.5 Pro: I need to solve this step-by-step:

        Given information:
        - Train A: leaves at 2:00 PM, speed = 60 mph
        - Train B: leaves at 2:30 PM, speed = 80 mph
        - Distance between stations = 280 miles

        Let t = time (in hours) after 2:00 PM when trains meet

        Train A travels for t hours: distance = 60t
        Train B travels for (t - 0.5) hours: distance = 80(t - 0.5)

        When they meet: 60t + 80(t - 0.5) = 280
        60t + 80t - 40 = 280
        140t = 320
        t = 32/14 = 16/7 ≈ 2.286 hours

        They meet at 2:00 PM + 2.286 hours = 4:17 PM

        cGPT o3: Let me work through this systematically:

        Setup:
        - Train A starts at 2:00 PM at 60 mph
        - Train B starts at 2:30 PM at 80 mph
        - Total distance: 280 miles

        At time t hours after 2:00 PM:
        - Train A position: 60t miles from Station A
        - Train B position: 80(t - 0.5) miles from Station B (only if t ≥ 0.5)

        Meeting condition: 60t + 80(t - 0.5) = 280
        60t + 80t - 40 = 280
        140t = 320
        t = 32/14 = 16/7 hours ≈ 2 hours 17 minutes

        Meeting time: 2:00 PM + 2:17 = 4:17 PM
        """
    ]
    
    return "\n--- PAGE BREAK ---\n".join(sample_conversations)

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
        avg_response_length = df['metadata'].apply(lambda x: x.get('response_length', 0)).mean()
        st.metric("Avg Response Length", f"{avg_response_length:.0f} chars")
    
    # Model distribution
    st.subheader("🤖 Model Distribution")
    model_counts = df['ai_model'].value_counts()
    
    fig_pie = px.pie(
        values=model_counts.values,
        names=model_counts.index,
        title="Distribution of AI Models"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Response types
    st.subheader("📝 Response Type Analysis")
    response_types = df['response_type'].value_counts()
    
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
        'metadata': lambda x: sum(item.get('response_length', 0) for item in x) / len(x),
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
                        avg_len = model1_data['metadata'].apply(lambda x: x.get('response_length', 0)).mean()
                        st.write(f"Avg Length: {avg_len:.0f} chars")
                
                with col2:
                    model2_data = pair_data[pair_data['ai_model'] == model2]
                    if len(model2_data) > 0:
                        st.write(f"**{model2}**")
                        st.write(f"Responses: {len(model2_data)}")
                        avg_len = model2_data['metadata'].apply(lambda x: x.get('response_length', 0)).mean()
                        st.write(f"Avg Length: {avg_len:.0f} chars")

def show_detailed_view(data, analyzer):
    """Show detailed view of conversations"""
    st.header("🔍 Detailed Conversation View")
    
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
                st.write(f"Version: {row['metadata'].get('model_version', 'N/A')}")
                st.write(f"Response Length: {row['metadata'].get('response_length', 0)} chars")
                st.write(f"Has Code: {row['metadata'].get('has_code', False)}")
                st.write(f"Has Math: {row['metadata'].get('has_math', False)}")
            
            with col2:
                st.write("**User Prompt:**")
                st.write(row['user_prompt'])
                
                st.write("**AI Response:**")
                st.write(row['ai_response'])
                
                if row['conversation_context']:
                    st.write("**Context:**")
                    st.write(row['conversation_context'])

def show_export_options(data, analyzer):
    """Show export options"""
    st.header("📥 Export Data")
    
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
                flat_item = {
                    'conversation_id': item['conversation_id'],
                    'page': item['page'],
                    'ai_model': item['ai_model'],
                    'user_prompt': item['user_prompt'],
                    'ai_response': item['ai_response'],
                    'response_type': item['response_type'],
                    'conversation_context': item['conversation_context'],
                    'timestamp': item['timestamp'],
                    'model_version': item['metadata'].get('model_version', ''),
                    'response_length': item['metadata'].get('response_length', 0),
                    'has_code': item['metadata'].get('has_code', False),
                    'has_math': item['metadata'].get('has_math', False)
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

def generate_summary_report(data, analyzer):
    """Generate a summary report"""
    df = pd.DataFrame(data)
    
    report = f"""
AI Model Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Conversations: {len(df)}
Pages Processed: {df['page'].nunique()}
AI Models Analyzed: {', '.join(df['ai_model'].unique())}
Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}

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
        avg_length = model_data['metadata'].apply(lambda x: x.get('response_length', 0)).mean()
        code_responses = model_data['metadata'].apply(lambda x: x.get('has_code', False)).sum()
        math_responses = model_data['metadata'].apply(lambda x: x.get('has_math', False)).sum()
        
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

if __name__ == "__main__":
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    
    main()
