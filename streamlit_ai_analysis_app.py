import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import uuid
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Model Conversation Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for better comparison view
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .prompt-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        border-left: 5px solid #007bff;
    }
    
    .prompt-id {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #6c757d;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        word-break: break-all;
    }
    
    .prompt-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #495057;
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .comparison-container {
        display: flex;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .model-column {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    
    .gemini-column { border-left-color: #4285f4; }
    .chatgpt-column { border-left-color: #10a37f; }
    .bard-column { border-left-color: #4285f4; }
    .claude-column { border-left-color: #ff6b35; }
    .ais-column { border-left-color: #7b1fa2; }
    .unknown-column { border-left-color: #6c757d; }
    
    .model-header {
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .gemini-header { background: #4285f4; color: white; }
    .chatgpt-header { background: #10a37f; color: white; }
    .bard-header { background: #4285f4; color: white; }
    .claude-header { background: #ff6b35; color: white; }
    .ais-header { background: #7b1fa2; color: white; }
    .unknown-header { background: #6c757d; color: white; }
    
    .response-content {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 0.95rem;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
    }
    
    .response-content h1, .response-content h2, .response-content h3 {
        color: #495057;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .response-content ul, .response-content ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .response-content li {
        margin-bottom: 0.5rem;
    }
    
    .response-content strong {
        color: #495057;
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stat-item {
        display: inline-block;
        margin: 0 2rem 0 0;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #495057;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .processing-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    
    .follow-up-section {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    
    .follow-up-text {
        font-style: italic;
        color: #856404;
    }
    
    .no-data {
        text-align: center;
        padding: 3rem;
        color: #6c757d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedAIModelDetector:
    def __init__(self):
        self.model_patterns = {
            'Gemini': {
                'patterns': [
                    r'gemini', r'bard', r'google ai studio', r'google ai', r'palm'
                ],
                'variants': ['2.5 Pro', '2.5 Flash', 'Pro', 'Flash', '1.5 Pro'],
                'css_class': 'gemini',
                'interface_indicators': ['google ai studio', 'gemini 2.5 pro', 'thoughts', 'auto']
            },
            'ChatGPT': {
                'patterns': [
                    r'chatgpt', r'gpt', r'openai', r'cgpt'
                ],
                'variants': ['o3', '4o', '4', '3.5', 'o1'],
                'css_class': 'chatgpt',
                'interface_indicators': ['chatgpt', 'openai', 'thought for']
            },
            'Claude': {
                'patterns': [
                    r'claude', r'anthropic claude'
                ],
                'variants': ['3.5 Sonnet', '3 Opus', '3 Haiku', 'Sonnet', 'Opus'],
                'css_class': 'claude',
                'interface_indicators': ['claude', 'anthropic']
            },
            'AIS': {
                'patterns': [
                    r'ais', r'anthropic system'
                ],
                'variants': ['2.5 Pro', '2.5 Flash', 'Pro', 'Flash'],
                'css_class': 'ais',
                'interface_indicators': ['ais', 'anthropic system']
            }
        }
    
    def detect_model_from_page(self, page_text: str, page_num: int) -> Tuple[str, str]:
        """Detect AI model from page text with enhanced logic"""
        text_lower = page_text.lower()
        
        # Check for model name pages (like "ChatGPT" or "Gemini")
        if len(page_text.strip()) < 50:  # Likely a title page
            for model_name, config in self.model_patterns.items():
                for pattern in config['patterns']:
                    if re.search(pattern, text_lower):
                        return model_name, model_name
        
        # Check for interface indicators
        for model_name, config in self.model_patterns.items():
            for indicator in config['interface_indicators']:
                if indicator in text_lower:
                    # Try to detect specific variant
                    for variant in config['variants']:
                        if variant.lower() in text_lower:
                            return model_name, f"{model_name} {variant}"
                    return model_name, model_name
        
        # Fallback to pattern matching
        for model_name, config in self.model_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, text_lower):
                    for variant in config['variants']:
                        if variant.lower() in text_lower:
                            return model_name, f"{model_name} {variant}"
                    return model_name, model_name
        
        return 'Unknown', 'Unknown Model'
    
    def extract_conversation_id_and_prompt(self, first_page_text: str) -> Tuple[str, str]:
        """Extract conversation ID and initial prompt from first page"""
        lines = first_page_text.strip().split('\n')
        
        conversation_id = ""
        initial_prompt = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('ID:'):
                conversation_id = line[3:].strip()
            elif line.startswith('Initial Prompt:'):
                initial_prompt = line[15:].strip()
        
        return conversation_id, initial_prompt
    
    def group_pages_by_model(self, pdf_pages: List[str]) -> Dict:
        """Group PDF pages by detected AI model"""
        if not pdf_pages:
            return {}
        
        # Extract ID and prompt from first page
        conversation_id, initial_prompt = self.extract_conversation_id_and_prompt(pdf_pages[0])
        
        model_groups = {}
        current_model = None
        current_pages = []
        
        for i, page_text in enumerate(pdf_pages[1:], 1):  # Skip first page (ID/prompt)
            detected_model, full_model_name = self.detect_model_from_page(page_text, i)
            
            # Check if this is a new model or continuation
            if detected_model != 'Unknown' and detected_model != current_model:
                # Save previous model's pages
                if current_model and current_pages:
                    if current_model not in model_groups:
                        model_groups[current_model] = {
                            'full_name': current_model,
                            'pages': [],
                            'content': ""
                        }
                    model_groups[current_model]['pages'].extend(current_pages)
                    model_groups[current_model]['content'] += '\n'.join(current_pages)
                
                # Start new model group
                current_model = detected_model
                current_pages = [page_text]
                
                # Update full name if we have a more specific version
                if current_model not in model_groups:
                    model_groups[current_model] = {
                        'full_name': full_model_name,
                        'pages': [],
                        'content': ""
                    }
                else:
                    model_groups[current_model]['full_name'] = full_model_name
            else:
                # Continue with current model
                if current_model:
                    current_pages.append(page_text)
        
        # Save final model's pages
        if current_model and current_pages:
            if current_model not in model_groups:
                model_groups[current_model] = {
                    'full_name': current_model,
                    'pages': [],
                    'content': ""
                }
            model_groups[current_model]['pages'].extend(current_pages)
            model_groups[current_model]['content'] += '\n'.join(current_pages)
        
        return {
            'conversation_id': conversation_id,
            'initial_prompt': initial_prompt,
            'models': model_groups
        }
    
    def clean_and_format_response(self, raw_content: str) -> str:
        """Clean and format the AI response content"""
        # Remove UI elements and clean text
        lines = raw_content.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            r'google ai studio',
            r'get api key',
            r'dashboard',
            r'documentation',
            r'run settings',
            r'token count',
            r'temperature',
            r'media resolution',
            r'thinking',
            r'structured output',
            r'code execution',
            r'function calling',
            r'advanced settings',
            r'chatgpt',
            r'share',
            r'ask anything',
            r'tools',
            r'thought for \d+ seconds'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip UI elements
            skip_line = False
            for pattern in skip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    skip_line = True
                    break
            
            if not skip_line and len(line) > 3:
                cleaned_lines.append(line)
        
        # Join and format
        content = '\n'.join(cleaned_lines)
        
        # Format sections and lists
        content = re.sub(r'\n(\d+\.\s)', r'\n\n\1', content)  # Add space before numbered lists
        content = re.sub(r'\n([A-Z][^.]*:)', r'\n\n**\1**', content)  # Bold section headers
        content = re.sub(r'\n•\s', r'\n• ', content)  # Clean bullet points
        
        return content

def read_pdf_enhanced(uploaded_file) -> List[str]:
    """Enhanced PDF text extraction with page separation"""
    try:
        import fitz
        
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append(text)
        
        doc.close()
        return pages
        
    except ImportError:
        st.error("PyMuPDF is not installed. Please install it with: pip install PyMuPDF")
        return []
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Model Conversation Analyzer</h1>
        <p>Advanced PDF analysis for side-by-side AI model comparisons</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    detector = EnhancedAIModelDetector()
    
    # File upload section
    st.markdown("### 📄 Upload PDF File")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file with AI conversation screenshots",
        type=['pdf'],
        help="Upload a PDF containing AI model conversation screenshots for comparison analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size = uploaded_file.size / 1024  # KB
        size_unit = "KB"
        if file_size > 1024:
            file_size /= 1024
            size_unit = "MB"
        
        st.info(f"📁 **{uploaded_file.name}** ({file_size:.1f} {size_unit})")
        
        # Process button
        if st.button("🚀 Analyze Conversations", type="primary"):
            with st.spinner("Processing PDF and analyzing conversations..."):
                try:
                    # Extract pages
                    pdf_pages = read_pdf_enhanced(uploaded_file)
                    
                    if pdf_pages:
                        # Store pages for debugging
                        st.session_state.pdf_pages = pdf_pages
                        
                        # Group by model
                        analysis_result = detector.group_pages_by_model(pdf_pages)
                        
                        if analysis_result and analysis_result.get('models'):
                            # Store in session state
                            st.session_state.analysis_result = analysis_result
                            
                            st.markdown('<div class="processing-status success">✅ Analysis complete!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="processing-status warning">⚠️ No AI model conversations detected in the PDF.</div>', unsafe_allow_html=True)
                            # Store empty result to avoid None errors
                            st.session_state.analysis_result = {
                                'conversation_id': '',
                                'initial_prompt': '',
                                'models': {}
                            }
                    else:
                        st.markdown('<div class="processing-status error">❌ Failed to extract text from PDF.</div>', unsafe_allow_html=True)
                        # Store empty result to avoid None errors
                        st.session_state.analysis_result = {
                            'conversation_id': '',
                            'initial_prompt': '',
                            'models': {}
                        }
                except Exception as e:
                    st.markdown(f'<div class="processing-status error">❌ Error during analysis: {str(e)}</div>', unsafe_allow_html=True)
                    # Store empty result to avoid None errors
                    st.session_state.analysis_result = {
                        'conversation_id': '',
                        'initial_prompt': '',
                        'models': {}
                    }
    
    # Display results
    if ('analysis_result' in st.session_state and 
        st.session_state.analysis_result is not None and 
        st.session_state.analysis_result.get('models')):
        
        result = st.session_state.analysis_result
        
        # Show conversation ID and prompt
        if result.get('conversation_id') or result.get('initial_prompt'):
            st.markdown("### 📋 Conversation Details")
            
            if result.get('conversation_id'):
                st.markdown(f"""
                <div class="prompt-section">
                    <strong>Conversation ID:</strong>
                    <div class="prompt-id">{result['conversation_id']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if result.get('initial_prompt'):
                st.markdown(f"""
                <div class="prompt-section">
                    <strong>Initial Prompt:</strong>
                    <div class="prompt-text">{result['initial_prompt']}</div>
                </div>
                """, unsafe_allow_html=True)
        
                # Summary statistics
        models = result.get('models', {})
        if models:
            st.markdown("### 📊 Analysis Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{len(models)}</div>
                    <div class="stat-label">AI Models</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_pages = sum(len(model_data.get('pages', [])) for model_data in models.values())
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{total_pages}</div>
                    <div class="stat-label">Total Pages</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_chars = sum(len(model_data.get('content', '')) for model_data in models.values())
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{total_chars:,}</div>
                    <div class="stat-label">Total Characters</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Side-by-side comparison
            st.markdown("### 🔄 Side-by-Side Model Comparison")
            
            model_list = list(models.keys())
            
            if len(model_list) >= 2:
                # Create columns for comparison
                cols = st.columns(len(model_list))
                
                for i, (model_name, col) in enumerate(zip(model_list, cols)):
                    model_data = models[model_name]
                    css_class = detector.model_patterns.get(model_name, {}).get('css_class', 'unknown')
                    
                    with col:
                        st.markdown(f"""
                        <div class="model-column {css_class}-column">
                            <div class="model-header {css_class}-header">
                                {model_data.get('full_name', model_name)}
                            </div>
                            <div class="response-content">
                                {detector.clean_and_format_response(model_data.get('content', ''))}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                # Single model view
                for model_name, model_data in models.items():
                    css_class = detector.model_patterns.get(model_name, {}).get('css_class', 'unknown')
                    
                    st.markdown(f"""
                    <div class="model-column {css_class}-column">
                        <div class="model-header {css_class}-header">
                            {model_data.get('full_name', model_name)}
                        </div>
                        <div class="response-content">
                            {detector.clean_and_format_response(model_data.get('content', ''))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("### 📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Comparison Data"):
                    comparison_text = f"""
AI Model Conversation Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONVERSATION ID: {result.get('conversation_id', 'N/A')}
INITIAL PROMPT: {result.get('initial_prompt', 'N/A')}

{'='*50}
"""
                    
                    for model_name, model_data in models.items():
                        comparison_text += f"""
{model_data.get('full_name', model_name)}
{'-'*len(model_data.get('full_name', model_name))}
{detector.clean_and_format_response(model_data.get('content', ''))}

{'='*50}
"""
                    
                    st.text_area("Comparison Data", comparison_text, height=300)
            
            with col2:
                if st.button("📊 Download Analysis"):
                    # Create structured data
                    analysis_data = {
                        'conversation_id': result.get('conversation_id', ''),
                        'initial_prompt': result.get('initial_prompt', ''),
                        'timestamp': datetime.now().isoformat(),
                        'models': {}
                    }
                    
                    for model_name, model_data in models.items():
                        analysis_data['models'][model_name] = {
                            'full_name': model_data.get('full_name', model_name),
                            'content': detector.clean_and_format_response(model_data.get('content', '')),
                            'page_count': len(model_data.get('pages', [])),
                            'character_count': len(model_data.get('content', ''))
                        }
                    
                    import json
                    json_str = json.dumps(analysis_data, indent=2)
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"ai_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.markdown('<div class="processing-status warning">⚠️ No AI model data found in the analysis result.</div>', unsafe_allow_html=True)
    
    else:
        # Check if we have an empty analysis result (processing was attempted but failed)
        if ('analysis_result' in st.session_state and 
            st.session_state.analysis_result is not None and 
            not st.session_state.analysis_result.get('models')):
            
            st.markdown("""
            <div class="processing-status warning">
                <h3>⚠️ Analysis Complete - No Model Conversations Found</h3>
                <p>The PDF was processed but no AI model conversations were detected. This could be due to:</p>
                <ul>
                    <li>The PDF format is different from expected</li>
                    <li>The text extraction didn't capture conversation structure</li>
                    <li>The AI model patterns weren't recognized</li>
                </ul>
                <p>Try uploading a different PDF or check the file format.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # No processing has been attempted yet
            st.markdown("""
            <div class="no-data">
                <h3>📄 Upload a PDF to begin analysis</h3>
                <p>Select a PDF file containing AI conversation screenshots for detailed comparison analysis.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = {
            'conversation_id': '',
            'initial_prompt': '',
            'models': {}
        }
    if 'pdf_pages' not in st.session_state:
        st.session_state.pdf_pages = []
    
    main()