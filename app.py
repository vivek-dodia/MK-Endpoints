import streamlit as st
import os
import uuid  # Add this import
from dotenv import load_dotenv
import time
import warnings
from src.api_utils import is_id_endpoint, connect_to_mikrotik, safe_disconnect, call_mikrotik_api
from src.nlp_utils import load_resources, generate_related_questions, format_responses_to_natural_language_stream
from llm_wrapper import LLMWrapper

# Silence warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")
warnings.filterwarnings("ignore", message="Model was trained with padding")

# Configure page first
st.set_page_config(
    page_title="MikroTik API Finder",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Configure Qdrant and other services
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "mikrotik-GET"

# MikroTik Router settings
MIKROTIK_IP = os.environ.get("test_harlem_mikrotik", "127.0.0.1")
MIKROTIK_USER = os.environ.get("test_harlem_mikrotik_user", "")
MIKROTIK_PW = os.environ.get("test_harlem_mikrotik_pw", "")

# Initialize direct LLM wrapper with Gemini Flash as primary and DeepSeek as fallback
llm = LLMWrapper(
    default_model="gemini-flash",
    fallbacks=["deepseek-chat"]
)

# Initialize session state variables
def init_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'mikrotik_connected' not in st.session_state:
        st.session_state.mikrotik_connected = False
    if 'mikrotik_api' not in st.session_state:
        st.session_state.mikrotik_api = None
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = False
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'submitted_query' not in st.session_state:
        st.session_state.submitted_query = ""
    if 'last_search_time' not in st.session_state:
        st.session_state.last_search_time = None
    if 'last_search_count' not in st.session_state:
        st.session_state.last_search_count = 0

# Custom CSS for styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-green { background-color: #28a745; }
    .status-red { background-color: #dc3545; }
    .status-gray { background-color: #6c757d; }
    .status-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Custom sidebar toggle */
    .sidebar-toggle {
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 1000;
        background-color: #fdfdf7;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Sidebar panel */
    .sidebar-panel {
        position: fixed;
        top: 0;
        right: 0;
        width: 350px;
        height: 100vh;
        background-color: #fdfdf7;
        padding: 20px;
        box-shadow: -2px 0 5px rgba(0,0,0,0.1);
        overflow-y: auto;
        transition: transform 0.3s ease;
        z-index: 999;
    }
    .sidebar-panel.closed {
        transform: translateX(350px);
    }

    /* Main content adjustments */
    .main-content {
        transition: margin-right 0.3s ease;
    }
    .main-content.sidebar-open {
        margin-right: 350px;
    }

    /* Buttons in sidebar */
    .sidebar-panel button {
        margin-bottom: 8px;
    }

    /* Theme customization */
    .stApp {
        background-color: #fdfdf7;
    }
    .css-12oz5g7, .css-1adrfps, .css-1cypcdb {
        background-color: #fdfdf7 !important;
    }
    .reportview-container {
        background-color: #fdfdf7;
        color: #4d505e;
    }
    .css-hi6a2p {
        color: #4d505e !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Render the sidebar
def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        
        # LLM settings
        st.subheader("LLM Settings")
        
        # Model selection - only show models for which we have API keys
        available_models = []
        if os.environ.get("GOOGLE_API_KEY"):
            available_models.append(("gemini-flash", "Gemini 2.0 Flash-Lite (Fast)"))
            available_models.append(("gemini-pro", "Gemini Pro (Better quality)"))
            
        if os.environ.get("DEEPSEEK_API_KEY"):
            available_models.append(("deepseek-chat", "DeepSeek Chat"))
            available_models.append(("deepseek-coder", "DeepSeek Coder"))
        
        if available_models:
            selected_model = st.selectbox(
                "Primary Model", 
                options=[m[0] for m in available_models],
                format_func=lambda x: dict(available_models).get(x, x),
                index=0
            )
            
            # Update LLM wrapper with selected model
            if selected_model != llm.default_model:
                llm.default_model = selected_model
        else:
            st.warning("No API keys found for any LLM provider. Please set API keys in your .env file.")
        
        use_reasoning = st.checkbox("Use reasoning (two-stage processing)", value=True,
                                  help="Improves quality for complex queries but takes longer")
        st.session_state.use_reasoning = use_reasoning
        
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
        score_threshold = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        
        exclude_id_endpoints = st.checkbox("Exclude ID-specific endpoints", value=True, 
                                         help="Filter out endpoints with {id} parameters that require specific identifiers")
        
        st.markdown("---")
        
        # Router connection settings section
        render_router_connection()
        
        st.markdown("---")
        
        # Related Questions section
        render_related_questions()
        
        st.markdown("---")
        
        # Example queries section
        render_example_queries()

# Router connection settings
def render_router_connection():
    st.header("Router Connection")
    
    # Connection status indicator
    conn_status_container = st.container()
    router_ip = st.text_input("Router IP", MIKROTIK_IP)
    router_user = st.text_input("Username", MIKROTIK_USER)
    router_password = st.text_input("Password", MIKROTIK_PW, type="password")
    router_port = st.number_input("Port", min_value=1, max_value=65535, value=8728)
    use_ssl = st.checkbox("Use SSL", value=False)
    
    # Toggle between mock and real data
    use_mock = st.checkbox("Use Mock Data", value=False, 
                          help="Use simulated data instead of connecting to a real router")
    st.session_state.use_mock_data = use_mock
    
    # Display connection status indicator
    with conn_status_container:
        if st.session_state.use_mock_data:
            st.markdown(
                f"""<div class="status-container">
                    <div class="status-indicator status-gray"></div>
                    <div>Using mock data</div>
                </div>""", 
                unsafe_allow_html=True
            )
        elif st.session_state.mikrotik_connected:
            st.markdown(
                f"""<div class="status-container">
                    <div class="status-indicator status-green"></div>
                    <div>Connected to {router_ip}</div>
                </div>""", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class="status-container">
                    <div class="status-indicator status-red"></div>
                    <div>Not connected</div>
                </div>""", 
                unsafe_allow_html=True
            )
    
    if not use_mock:
        # Test connection button
        if st.button("Test Connection"):
            conn_test_status = st.empty()
            conn_test_status.info("Testing connection...")
            
            # Disconnect any existing connection
            safe_disconnect()
            
            # Try to connect
            api = connect_to_mikrotik(router_ip, router_user, router_password, router_port, use_ssl)
            if api:
                # Test with a simple command
                try:
                    resource = api.get_resource('/system/resource')
                    result = resource.get()[0]
                    conn_test_status.markdown(
                        f"""✅ Connection successful!  
                        Router uptime: {result.get('uptime', 'Unknown')}  
                        Version: {result.get('version', 'Unknown')}"""
                    )
                    # Force refresh the connection status indicator
                    st.rerun()
                except Exception as e:
                    conn_test_status.error(f"Connection test failed: {str(e)}")
            else:
                conn_test_status.error("Connection failed")

# Related questions section
def render_related_questions():
    st.header("Related Questions")
    if 'related_questions' in st.session_state:
        for question in st.session_state.related_questions:
            if st.button(question, key=f"related_{hash(question)}"):
                st.session_state.query = question
                st.session_state.query_input = question
                st.session_state.submitted_query = question
                st.rerun()

# Example queries section
def render_example_queries():
    st.header("Example Queries")
    example_queries = [
        "Show me all interfaces and their IP addresses",
        "List all DHCP leases",
        "What firewall rules are active?",
        "Show wireless clients connected to my network",
        "Check bridge port status",
        "View all DNS settings"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{hash(query)}"):
            st.session_state.query = query
            st.session_state.query_input = query
            st.session_state.submitted_query = query
            st.rerun()

# Render the main content
def render_main_content():
    # Callback to update submitted query when form is submitted
    def submit_query():
        st.session_state.submitted_query = st.session_state.query_input
    
    # Main query input with form
    with st.form(key="query_form"):
        query_input = st.text_input(
            "Ask about MikroTik configuration:", 
            value=st.session_state.query,
            key="query_input"
        )
        submit_button = st.form_submit_button("Search", on_click=submit_query)
    
    # Conversation history display
    if len(st.session_state.conversation_history) > 0:
        with st.expander("Conversation History", expanded=False):
            for i, exchange in enumerate(st.session_state.conversation_history):
                st.markdown(f"**You**: {exchange['query']}")
                st.markdown(f"**Assistant**: {exchange['response']}")
                st.markdown("---")
    
    # Process the query
    if st.session_state.submitted_query:
        process_query(st.session_state.submitted_query)

# Process the query and display results
def process_query(query):
    # Get settings from session state
    num_results = st.session_state.get('num_results', 5)
    score_threshold = st.session_state.get('score_threshold', 0.35)
    exclude_id_endpoints = st.session_state.get('exclude_id_endpoints', True)
    
    # Create placeholder for streaming updates
    process_status = st.empty()
    endpoints_status = st.empty()
    api_status = st.empty()
    result_container = st.empty()
    tech_details_container = st.container()
    
    process_status.info("🔍 Finding relevant API endpoints...")
    
    # Record start time
    start_time = time.time()
    
    # Load model and client
    model, client = load_resources(QDRANT_URL, QDRANT_API_KEY)
    
    # Generate embedding for query
    query_embedding = model.encode(query)
    
    # Search Qdrant for similar endpoints
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=num_results * 2,  # Get more to allow for filtering
        score_threshold=score_threshold
    )
    
    # Filter out ID endpoints if option is selected
    if exclude_id_endpoints:
        search_results = [result for result in search_results if not is_id_endpoint(result.payload["path"])]
        search_results = search_results[:num_results]  # Trim back to requested limit
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Display results
    if search_results:
        # Clear any process status
        process_status.empty()
        
        # Show selected endpoints with integrated stats
        endpoints_text = f"### Selected API Endpoints ({len(search_results)} found in {elapsed_time:.2f}s):\n"
        for i, result in enumerate(search_results):
            endpoints_text += f"{i+1}. **{result.payload['path']}** (Score: {result.score:.2f})\n"
        endpoints_status.markdown(endpoints_text)
        
        # Generate related questions
        related_questions = generate_related_questions(query, search_results, llm)
        st.session_state.related_questions = related_questions
        
        # Connect to router if not already connected
        if not st.session_state.use_mock_data and not st.session_state.mikrotik_connected:
            api_status.info("🔌 Connecting to MikroTik...")
            connect_to_mikrotik(MIKROTIK_IP, MIKROTIK_USER, MIKROTIK_PW)
        
        # Execute API calls to all relevant endpoints
        api_status.info("📡 Querying MikroTik router...")
        endpoints_with_responses = []
        
        for i, result in enumerate(search_results):
            endpoint = result.payload
            
            # Update status with current endpoint
            api_status.info(f"📡 Querying endpoint {i+1}/{len(search_results)}: {endpoint['path']}")
            
            # Execute API call
            response = call_mikrotik_api(
                endpoint["path"], 
                status_placeholder=api_status
            )
            
            # Store endpoint and response
            endpoints_with_responses.append({
                "endpoint": endpoint,
                "response": response,
                "score": result.score
            })
        
        # Format all responses into natural language with streaming
        api_status.empty()  # Clear any previous status messages
        consolidated_response = format_responses_to_natural_language_stream(
            query, 
            endpoints_with_responses,
            result_container,
            llm
        )
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            "query": query,
            "response": consolidated_response,
            "endpoints": [ep["endpoint"]["path"] for ep in endpoints_with_responses],
            "timestamp": time.time()
        })
        
        # Collapsible technical details
        with tech_details_container:
            with st.expander("🔍 Technical Details"):
                tabs = st.tabs([f"{i+1}. {ep['endpoint']['path']} ({ep['score']:.2f})" for i, ep in enumerate(endpoints_with_responses)])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        endpoint_data = endpoints_with_responses[i]
                        endpoint = endpoint_data["endpoint"]
                        response = endpoint_data["response"]
                        
                        st.markdown(f"**Summary**: {endpoint.get('summary', 'No summary available')}")
                        st.markdown(f"**API Family**: {endpoint.get('api_family', 'Unknown')}")
                        
                        st.markdown("### Parameters")
                        if 'parameter_descriptions' in endpoint and endpoint['parameter_descriptions']:
                            param_data = []
                            for param, desc in endpoint['parameter_descriptions'].items():
                                example = endpoint.get('parameter_examples', {}).get(param, "")
                                param_data.append([param, desc, example])
                            
                            if param_data:
                                st.table(param_data)
                        
                        st.markdown("### Raw Response")
                        st.json(response)
        
        # Update metrics
        st.session_state.last_search_time = elapsed_time
        st.session_state.last_search_count = len(search_results)
    else:
        process_status.warning("No matching API endpoints found. Try a different query or lower the similarity threshold.")

# Render the footer with metrics
def render_footer():
    st.markdown("---")
    
    # Display metrics in footer
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        if st.session_state.last_search_time:
            st.metric("Last search time", f"{st.session_state.last_search_time:.2f}s")
    with metrics_col2:
        if st.session_state.last_search_count:
            st.metric("Endpoints found", f"{st.session_state.last_search_count}")
    with metrics_col3:
        if len(st.session_state.conversation_history) > 0:
            st.metric("Conversation length", f"{len(st.session_state.conversation_history)}")
    
    st.markdown("This tool uses vector similarity to find the most relevant MikroTik API endpoints and consolidates information into natural language.")

# Add JavaScript for sidebar toggle
def add_javascript():
    st.markdown("""
    <script>
    // Sidebar toggle functionality
    document.addEventListener('DOMContentLoaded', function() {
        // Get sidebar elements
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebarPanel = document.querySelector('.sidebar-panel');
        const mainContent = document.querySelector('.main-content');
        
        // Toggle sidebar on button click
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', function() {
                sidebarPanel.classList.toggle('closed');
                mainContent.classList.toggle('sidebar-open');
                
                // Update toggle icon
                if (sidebarPanel.classList.contains('closed')) {
                    sidebarToggle.textContent = '≫';
                } else {
                    sidebarToggle.textContent = '≪';
                }
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Main function to run the app
def main():
    import uuid  # For session ID
    
    # Initialize session state
    init_session_state()
    
    # Load custom CSS
    load_custom_css()
    
    # App Header
    st.title("🔍 MikroTik API Finder")
    st.markdown("Ask questions in natural language and get answers from the right MikroTik API endpoints.")
    
    # Sidebar toggle button
    st.markdown(
        f"""
        <div class="sidebar-toggle" onclick="document.querySelector('.sidebar-panel').classList.toggle('closed'); document.querySelector('.main-content').classList.toggle('sidebar-open');">
            {'≪' if st.session_state.sidebar_expanded else '≫'}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Main content area with proper class for sidebar interactions
    st.markdown('<div class="main-content"></div>', unsafe_allow_html=True)
    
    # Create fixed-position sidebar panel
    sidebar_html = f"""
    <div class="sidebar-panel {'closed' if not st.session_state.sidebar_expanded else ''}">
        <h3>Settings</h3>
        <p>Toggle settings, view related questions, and examples here.</p>
    </div>
    """
    st.markdown(sidebar_html, unsafe_allow_html=True)
    
    # Render the sidebar
    render_sidebar()
    
    # Create a container for the main content
    main_col = st.container()
    
    with main_col:
        render_main_content()
    
    # Render the footer
    render_footer()
    
    # Add JavaScript for sidebar toggle
    add_javascript()
    
    # Cleanup on app exit
    if hasattr(st, 'on_session_end'):
        st.on_session_end(safe_disconnect)

if __name__ == "__main__":
    main()