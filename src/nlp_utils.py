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

# Removed the generate_related_questions function for a leaner application

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