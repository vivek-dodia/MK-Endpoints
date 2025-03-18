import streamlit as st
import json
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

@st.cache_resource
def load_resources(qdrant_url, qdrant_api_key):
    """Load and cache the embedding model and Qdrant client"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return model, client

def generate_related_questions(query, results, llm):
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

def format_responses_to_natural_language_stream(query, endpoints_with_responses, result_placeholder, llm):
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
        llm_prompt = f"""
        You are a MikroTik networking expert. Convert these technical API responses to natural language for a network engineer.
        
        User Query: "{query}"
        
        {history_context}
        
        API Responses:
        """
        
        for endpoint_data in endpoints_with_responses:
            endpoint = endpoint_data["endpoint"]
            response = endpoint_data["response"]
            llm_prompt += f"\nEndpoint: {endpoint['path']}\nResponse: {json.dumps(response, indent=2)}\n"
        
        llm_prompt += """
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
        
        # Create a clean result area with a heading and single placeholder
        result_placeholder.markdown("## 📊 Result")
        response_area = result_placeholder.container()
        
        # Start with subtle progress indication
        progress_placeholder = response_area.empty()
        response_placeholder = response_area.empty()
        
        # Initial message
        model_info = f"{'deepseek-chat' if 'deepseek' in llm.default_model else llm.default_model}"
        progress_placeholder.markdown(f"<div style='color: #6c757d; font-style: italic;'>Analyzing with {model_info}...</div>", unsafe_allow_html=True)
        
        # Choose between reasoning and standard generation
        use_reasoning = st.session_state.get('use_reasoning', True)
        
        # Start the LLM request immediately in the background
        if use_reasoning:
            progress_placeholder.markdown(f"<div style='color: #6c757d; font-style: italic;'>Processing with reasoning...</div>", unsafe_allow_html=True)
            stream_generator = llm.generate_with_reasoning(
                prompt=llm_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                stream=True
            )
        else:
            progress_placeholder.markdown(f"<div style='color: #6c757d; font-style: italic;'>Processing query...</div>", unsafe_allow_html=True)
            stream_generator = llm.get_completion(
                prompt=llm_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                stream=True
            )
        
        # Initialize the response text
        full_response = ""
        cursor = "▌"
        
        # Process the streaming response
        try:
            # Process the stream
            for chunk in stream_generator:
                if chunk:
                    # Clear progress message on first chunk
                    if not full_response:
                        progress_placeholder.empty()
                    
                    full_response += chunk
                    response_placeholder.markdown(full_response + cursor)
                
            # Final update without cursor
            response_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            # Error handling
            progress_placeholder.empty()
            if full_response:
                response_placeholder.markdown(full_response)
                return full_response
            else:
                response_placeholder.error(f"Error during streaming: {str(e)}")
                return f"Error during streaming: {str(e)}"
    
    except Exception as e:
        # Fallback if overall formatting fails
        result_placeholder.error(f"Error formatting responses: {e}")
        results = []
        for endpoint_data in endpoints_with_responses:
            endpoint = endpoint_data["endpoint"]
            response = endpoint_data["response"]
            results.append(f"### Results from {endpoint['path']}:\n{json.dumps(response, indent=2)}")
        
        result_text = "\n\n".join(results)
        result_placeholder.markdown(result_text)
        return result_text