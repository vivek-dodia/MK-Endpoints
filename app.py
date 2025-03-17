import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json
import time
import re
import uuid
import routeros_api
import warnings
from typing import List, Dict, Any, Generator
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

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Track connection status
if 'mikrotik_connected' not in st.session_state:
    st.session_state.mikrotik_connected = False

# Store the RouterOS API connection
if 'mikrotik_api' not in st.session_state:
    st.session_state.mikrotik_api = None

# Initialize clients
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return model, client

model, client = load_resources()

def is_id_endpoint(path):
    """Check if the endpoint contains an {id} parameter in the path"""
    # Look for path segments with {xyz} pattern
    return bool(re.search(r'/\{[^/]+\}', path))

def connect_to_mikrotik(ip, username, password, port=8728, ssl=False):
    """Connect to MikroTik router using RouterOS API"""
    try:
        connection = routeros_api.RouterOsApiPool(
            host=ip,
            username=username,
            password=password,
            port=port if not ssl else 8729,
            use_ssl=ssl,
            plaintext_login=True
        )
        api = connection.get_api()
        st.session_state.mikrotik_api = api
        st.session_state.mikrotik_connected = True
        return api
    
    except routeros_api.exceptions.RouterOsApiConnectionError as e:
        st.error(f"Connection failed: {str(e)}")
        st.session_state.mikrotik_connected = False
        return None
    
    except routeros_api.exceptions.RouterOsApiError as e:
        st.error(f"API error: {str(e)}")
        st.session_state.mikrotik_connected = False
        return None
    
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
        st.session_state.mikrotik_connected = False
        return None

def safe_disconnect():
    """Safely disconnect from MikroTik router"""
    if st.session_state.mikrotik_api:
        try:
            st.session_state.mikrotik_api.disconnect()
            st.session_state.mikrotik_api = None
            st.session_state.mikrotik_connected = False
        except:
            pass

def call_mikrotik_api(endpoint, parameters=None, router_ip=None, status_placeholder=None):
    """Call the MikroTik API with proper error handling and fallbacks"""
    # Update status if placeholder provided
    if status_placeholder:
        status_placeholder.write(f"Querying: {endpoint}")
    
    # Use the actual router if credentials are available
    if MIKROTIK_USER and MIKROTIK_PW and not st.session_state.get('use_mock_data', False):
        try:
            # Ensure we have a connection
            api = st.session_state.mikrotik_api
            if not api:
                if status_placeholder:
                    status_placeholder.write(f"Establishing connection to router...")
                api = connect_to_mikrotik(MIKROTIK_IP, MIKROTIK_USER, MIKROTIK_PW)
                if not api:
                    raise Exception("Could not establish connection")
            
            # Map endpoint path to actual RouterOS API path
            # This is needed because the OpenAPI spec may use different paths
            command = endpoint.rstrip('/')
                
            # Fix common command path issues
            command = command.replace('/print', '')  # Remove print suffix
            
            # Remove leading slash if present
            if command.startswith('/'):
                command = command[1:]
                
            # Get the appropriate resource
            try:
                if status_placeholder:
                    status_placeholder.write(f"Accessing resource: /{command}")
                resource = api.get_resource('/' + command)
                result = resource.get()
                if status_placeholder:
                    status_placeholder.write(f"✅ Data retrieved from: {endpoint}")
                return result
            except Exception as e:
                if status_placeholder:
                    status_placeholder.write(f"⚠️ Error accessing {command}: {str(e)}")
                
                # Try to handle special cases or path mappings
                if "no such command" in str(e).lower():
                    # Try common variants
                    alt_commands = []
                    
                    # Handle special cases
                    if 'interface/print' in endpoint:
                        alt_commands.append('interface')
                    elif 'ip/address/print' in endpoint:
                        alt_commands.append('ip/address')
                    elif 'system/resource/print' in endpoint:
                        alt_commands.append('system/resource')
                    elif 'ip/firewall/filter/print' in endpoint:
                        alt_commands.append('ip/firewall/filter')
                    
                    # Try alternatives
                    for alt in alt_commands:
                        try:
                            if status_placeholder:
                                status_placeholder.write(f"Trying alternative: /{alt}")
                            resource = api.get_resource('/' + alt)
                            result = resource.get()
                            if status_placeholder:
                                status_placeholder.write(f"✅ Data retrieved from alternative: /{alt}")
                            return result
                        except:
                            continue
                
                # If all attempts fail, raise the original error
                raise
        
        except Exception as e:
            if status_placeholder:
                status_placeholder.write(f"⚠️ Error with MikroTik API: {str(e)}")
                status_placeholder.write("Using mock data instead of live router data")
            else:
                st.warning(f"Error with MikroTik API: {str(e)}")
                st.warning("Using mock data instead of live router data")
    
    # If we reach here, use mock data
    if status_placeholder:
        status_placeholder.write(f"📊 Using mock data for: {endpoint}")
    
    if "/interface" in endpoint and "print" in endpoint:
        return [
            {"name": "ether1", "type": "ether", "mtu": 1500, "actual-mtu": 1500, "mac-address": "00:0C:29:45:2A:3B", "running": True},
            {"name": "ether2", "type": "ether", "mtu": 1500, "actual-mtu": 1500, "mac-address": "00:0C:29:45:2A:3C", "running": False},
            {"name": "bridge1", "type": "bridge", "mtu": 1500, "actual-mtu": 1500, "mac-address": "00:0C:29:45:2A:3D", "running": True}
        ]
    elif "/ip/address" in endpoint and "print" in endpoint:
        return [
            {"address": "192.168.88.1/24", "network": "192.168.88.0", "interface": "ether1", "disabled": False},
            {"address": "10.0.0.1/24", "network": "10.0.0.0", "interface": "bridge1", "disabled": False}
        ]
    elif "/ip/dhcp-server/lease" in endpoint:
        return [
            {"mac-address": "00:0C:29:45:2A:3B", "address": "192.168.88.100", "host-name": "client1", "status": "bound"},
            {"mac-address": "00:0C:29:45:2A:3C", "address": "192.168.88.101", "host-name": "client2", "status": "bound"}
        ]
    elif "/ip/firewall/filter" in endpoint:
        return [
            {"chain": "input", "action": "accept", "protocol": "icmp", "comment": "Allow ICMP"},
            {"chain": "input", "action": "accept", "connection-state": "established,related", "comment": "Allow established connections"},
            {"chain": "input", "action": "drop", "comment": "Drop everything else"}
        ]
    elif "/system/resource" in endpoint:
        return {
            "uptime": "1d2h3m4s",
            "cpu-load": 15,
            "free-memory": 128000,
            "total-memory": 256000,
            "free-hdd-space": 100000,
            "total-hdd-space": 200000
        }
    elif "/interface/wireless/registration-table" in endpoint:
        return [
            {"mac-address": "00:0C:29:45:2A:3B", "interface": "wlan1", "signal-strength": "-65", "tx-rate": "65Mbps", "rx-rate": "65Mbps"},
            {"mac-address": "00:0C:29:45:2A:3C", "interface": "wlan1", "signal-strength": "-70", "tx-rate": "54Mbps", "rx-rate": "54Mbps"}
        ]
    # Add more mock responses for other endpoints
    return {"status": "success", "message": f"Mock data for {endpoint} (API integration failed)"}

def generate_related_questions(query, results):
    """Generate related follow-up questions based on the original query and results"""
    try:
        # Build context from results
        endpoints_summary = "\n".join([
            f"- {result.payload.get('path')}: {result.payload.get('summary', 'No summary')}"
            for result in results[:3]
        ])
        
        # Add conversation history context
        history_context = ""
        if len(st.session_state.conversation_history) > 0:
            last_exchanges = st.session_state.conversation_history[-3:] if len(st.session_state.conversation_history) > 3 else st.session_state.conversation_history
            history_context = "Previous questions:\n" + "\n".join([f"- {exchange['query']}" for exchange in last_exchanges])
        
        prompt = f"""
        You are a MikroTik networking expert. Generate 3-5 logical follow-up questions based on this user's query.
        
        User Query: "{query}"
        
        Top relevant API endpoints:
        {endpoints_summary}
        
        {history_context}
        
        Create natural language follow-up questions a network engineer might ask next. 
        Make them specific, technical, and directly related to MikroTik router management.
        
        Return ONLY a JSON array of questions, with no other text.
        Example: ["How many DHCP leases are active?", "What's the CPU utilization?"]
        """
        
        # Use JSON-specific generation with lower temperature for more focused results
        try:
            questions = llm.generate_json(
                prompt=prompt,
                temperature=0.7
            )
            
            # Handle different response formats
            if isinstance(questions, list):
                return questions[:5]  # Limit to 5 questions maximum
            elif isinstance(questions, dict) and "questions" in questions:
                return questions["questions"][:5]
            else:
                # Try to find an array in the response
                for key, value in questions.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                        return value[:5]
            
            # Return default questions if format doesn't match expectations
            return [
                "What's the CPU utilization of the router?",
                "Show me active interface status",
                "List all DHCP leases",
                "What firewall rules are currently active?",
                "Show me the bandwidth usage statistics"
            ]
        except Exception as e:
            st.warning(f"Error generating related questions: {e}")
            # Fallback questions
            return [
                "Show me active interfaces",
                "List DHCP leases",
                "Check firewall rules",
                "View wireless clients",
                "What's the router's uptime?"
            ]
    except Exception as e:
        st.error(f"Error generating related questions: {e}")
        return [
            "Show me active interfaces",
            "List DHCP leases",
            "Check firewall rules",
            "View wireless clients"
        ]

def format_responses_to_natural_language_stream(query, endpoints_with_responses, result_placeholder):
    """Format multiple API responses into natural language with streaming output"""
    try:
        # Add conversation history context
        history_context = ""
        if len(st.session_state.conversation_history) > 0:
            last_exchanges = st.session_state.conversation_history[-2:] if len(st.session_state.conversation_history) > 2 else st.session_state.conversation_history
            history_context = "Recent conversation history:\n" + "\n".join([
                f"User: {exchange['query']}\nAssistant: {exchange['response'][:100]}..." 
                for exchange in last_exchanges
            ])
        
        # Prepare the prompt
        prompt = f"""
        You are a MikroTik networking expert. Convert these technical API responses to natural language for a network engineer.
        
        User Query: "{query}"
        
        {history_context}
        
        API Responses:
        """
        
        for endpoint_data in endpoints_with_responses:
            endpoint = endpoint_data["endpoint"]
            response = endpoint_data["response"]
            prompt += f"\nEndpoint: {endpoint['path']}\nResponse: {json.dumps(response, indent=2)}\n"
        
        prompt += """
        Provide a clear, consolidated answer that addresses the user's query completely using data from all endpoints.
        
        Format your response in these sections:
        1. A direct, concise answer to the question
        2. Relevant details from the API responses
        3. Any important observations or recommendations
        
        Use technical networking terminology appropriate for a network engineer.
        """
        
        # System prompt that emphasizes MikroTik expertise
        system_prompt = """
        You are a MikroTik networking expert with deep knowledge of RouterOS. 
        Provide precise, technical information using correct networking terminology.
        Be concise yet comprehensive, focusing on what's most relevant to the network engineer's query.
        """
        
        # Initialize the response text
        full_response = ""
        result_placeholder.markdown("## 📊 Result")
        response_placeholder = result_placeholder.empty()
        
        # Choose between reasoning and standard generation based on user setting
        if 'use_reasoning' in st.session_state and st.session_state.use_reasoning:
            try:
                # Display info about reasoning mode
                model_info = f"Using primary model: {llm.default_model}"
                response_placeholder.info(f"{model_info} with two-stage reasoning")
                
                stream_generator = llm.generate_with_reasoning(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    stream=True
                )
            except Exception as e:
                st.warning(f"Reasoning generation failed, falling back to standard: {str(e)}")
                stream_generator = llm.get_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    stream=True
                )
        else:
            # Display info about model being used
            model_info = f"Using primary model: {llm.default_model}"
            response_placeholder.info(model_info)
            
            # Generate response with streaming
            stream_generator = llm.get_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                stream=True
            )
        
        # Process the streaming response
        cursor_animation = ["▌", "▐", ""]
        cursor_idx = 0
        
        try:
            for chunk in stream_generator:
                full_response += chunk
                # Animate cursor for better user feedback
                cursor = cursor_animation[cursor_idx % len(cursor_animation)]
                response_placeholder.markdown(full_response + cursor)
                cursor_idx += 1
            
            # Final update without the cursor
            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Streaming error: {str(e)}")
            # Try to salvage whatever we have
            if full_response:
                response_placeholder.markdown(full_response)
            else:
                response_placeholder.error("Failed to generate a response. Please try again.")
                full_response = "Error generating response. Please try again."
        
        return full_response
    except Exception as e:
        # Fallback if formatting fails
        result_placeholder.error(f"Error formatting responses: {e}")
        results = []
        for endpoint_data in endpoints_with_responses:
            endpoint = endpoint_data["endpoint"]
            response = endpoint_data["response"]
            results.append(f"### Results from {endpoint['path']}:\n{json.dumps(response, indent=2)}")
        
        result_text = "\n\n".join(results)
        result_placeholder.markdown(result_text)
        return result_text

# Header
st.title("🔍 MikroTik API Finder")
st.markdown("Ask questions in natural language and get answers from the right MikroTik API endpoints.")

# Define custom CSS for status indicators
st.markdown("""
<style>
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}
.status-green {
    background-color: #28a745;
}
.status-red {
    background-color: #dc3545;
}
.status-gray {
    background-color: #6c757d;
}
.status-container {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Main content and sidebar layout
main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
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
    
    # Router connection settings
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
    
    st.markdown("---")
    
    # Related Questions (will be populated after first query)
    st.header("Related Questions")
    if 'related_questions' in st.session_state:
        for question in st.session_state.related_questions:
            if st.button(question, key=f"related_{hash(question)}"):
                st.session_state.query = question
                st.session_state.query_input = question
                st.session_state.submitted_query = question
                st.rerun()
    
    st.markdown("---")
    
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

with main_col:
    # Initialize session state for query
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Initialize session state for submission control
    if 'submitted_query' not in st.session_state:
        st.session_state.submitted_query = ""
        
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
    
    # Separately handle example and related question clicks
    query = st.session_state.submitted_query
    
    # Conversation history display
    if len(st.session_state.conversation_history) > 0:
        with st.expander("Conversation History", expanded=False):
            for i, exchange in enumerate(st.session_state.conversation_history):
                st.markdown(f"**You**: {exchange['query']}")
                st.markdown(f"**Assistant**: {exchange['response']}")
                st.markdown("---")
    
    # Process the query
    if st.session_state.submitted_query:
        # Create placeholder for streaming updates
        process_status = st.empty()
        endpoints_status = st.empty()
        api_status = st.empty()
        result_container = st.empty()
        tech_details_container = st.container()
        
        process_status.info("🔍 Finding relevant API endpoints...")
        
        # Record start time
        start_time = time.time()
        
        # Generate embedding for query
        query_embedding = model.encode(st.session_state.submitted_query)
        
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
            related_questions = generate_related_questions(st.session_state.submitted_query, search_results)
            st.session_state.related_questions = related_questions
            
            # Connect to router if not already connected
            if not st.session_state.use_mock_data and not st.session_state.mikrotik_connected:
                api_status.info("🔌 Connecting to MikroTik...")
                connect_to_mikrotik(router_ip, router_user, router_password, router_port, use_ssl)
            
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
            api_status.info("💬 Generating human-readable response...")
            consolidated_response = format_responses_to_natural_language_stream(
                st.session_state.submitted_query, 
                endpoints_with_responses,
                result_container
            )
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "query": st.session_state.submitted_query,
                "response": consolidated_response,
                "endpoints": [ep["endpoint"]["path"] for ep in endpoints_with_responses],
                "timestamp": time.time()
            })
            
            # Clear status messages
            api_status.empty()
            
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
        else:
            process_status.warning("No matching API endpoints found. Try a different query or lower the similarity threshold.")

# Cleanup on app exit
if hasattr(st, 'on_session_end'):
    st.on_session_end(safe_disconnect)

# Footer
st.markdown("---")

# Add search metrics in a subtle way
if 'last_search_time' not in st.session_state:
    st.session_state.last_search_time = None
    st.session_state.last_search_count = 0

if st.session_state.submitted_query and search_results:
    st.session_state.last_search_time = elapsed_time
    st.session_state.last_search_count = len(search_results)

# Display metrics in footer
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
with metrics_col1:
    if st.session_state.last_search_time:
        st.metric("Last search time", f"{st.session_state.last_search_time:.2f}s")
with metrics_col2:
    if st.session_state.last_search_count:
        st.metric("Endpoints found", st.session_state.last_search_count)
with metrics_col3:
    if len(st.session_state.conversation_history) > 0:
        st.metric("Conversation length", len(st.session_state.conversation_history))

st.markdown("This tool uses vector similarity to find the most relevant MikroTik API endpoints and consolidates information into natural language.")