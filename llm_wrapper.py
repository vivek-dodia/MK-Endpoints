"""
LLM wrapper module for MikroTik API Finder using direct DeepSeek API calls.
This avoids LiteLLM and directly uses the DeepSeek API.
"""

import os
import json
import requests
from typing import List, Dict, Any, Union, Generator
import time
import streamlit as st

class LLMWrapper:
    """
    A simplified wrapper for DeepSeek API interactions without LiteLLM.
    """
    
    def __init__(
        self, 
        default_model: str = "deepseek-chat",
        fallbacks: List[str] = None,
    ):
        """
        Initialize the DeepSeek API wrapper.
        
        Args:
            default_model: The default model to use (without provider prefix)
            fallbacks: Not used in this simplified version
        """
        self.default_model = default_model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            st.warning("⚠️ No DeepSeek API key found in environment variables. Please set DEEPSEEK_API_KEY.")
    
    def get_completion(
        self, 
        prompt: str, 
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Get a completion from DeepSeek API directly.
        
        Args:
            prompt: The user prompt
            model: The specific model to use (falls back to default_model)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Either string response or generator if streaming
        """
        model = model or self.default_model
        
        # DeepSeek API endpoint
        api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request data
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            if stream:
                # Streaming response
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    stream=True
                )
                
                # Check for error
                if response.status_code != 200:
                    error_msg = f"DeepSeek API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    raise Exception(error_msg)
                
                # Return generator for streaming
                return self._process_stream(response)
            else:
                # Regular response
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data
                )
                
                # Check for error
                if response.status_code != 200:
                    error_msg = f"DeepSeek API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    raise Exception(error_msg)
                
                # Parse response
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
                
        except Exception as e:
            st.error(f"DeepSeek API call failed: {str(e)}")
            raise Exception(f"DeepSeek API call failed: {str(e)}")
    
    def _process_stream(self, response):
        """Process streaming response from DeepSeek API"""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    
                    # Check for [DONE] marker
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        delta_content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if delta_content:
                            yield delta_content
                    except json.JSONDecodeError:
                        continue

    def generate_json(
        self, 
        prompt: str, 
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.2,
    ) -> Union[Dict[str, Any], List[str]]:
        """
        Get a JSON response from the DeepSeek API.
        
        Args:
            prompt: The user prompt
            model: The specific model to use
            system_prompt: Optional system prompt
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON object or fallback list
        """
        # Add JSON formatting to the prompt
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no other text."
        
        if system_prompt:
            system_prompt += " You MUST respond in valid JSON format with no additional text."
        else:
            system_prompt = "You MUST respond in valid JSON format with no additional text."
            
        try:
            # First attempt: Try to use native JSON mode where available
            response = self.get_completion(
                prompt=json_prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                stream=False,
            )
            
            # Clean up the response to ensure it's valid JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            return json.loads(response)
            
        except json.JSONDecodeError:
            # Handle the case where proper JSON wasn't returned
            st.warning("Failed to parse JSON response, trying alternative approach...")
            
            # Second attempt: Be more explicit about formatting
            fallback_prompt = (
                f"{prompt}\n\nYou MUST respond with ONLY a valid JSON object. "
                "No other text or explanation. No markdown formatting tags like ```json. "
                "Just the raw JSON object starting with {{."
            )
            
            try:
                response = self.get_completion(
                    prompt=fallback_prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    stream=False,
                )
                
                # One more attempt to clean it up
                if "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                # Final cleaning: find the first '{' and the last '}'
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    response = response[start:end+1]
                    
                return json.loads(response)
            except Exception as e:
                # Return default questions if all attempts fail
                st.warning(f"JSON generation failed: {str(e)}")
                return ["Show me active interfaces", "List DHCP leases", "Check firewall rules", "View wireless clients"]
        except Exception as e:
            # If JSON parsing keeps failing, return a basic list of default questions
            st.warning(f"JSON generation failed: {str(e)}")
            return ["Show me active interfaces", "List DHCP leases", "Check firewall rules", "View wireless clients"]

    def generate_with_reasoning(
        self,
        prompt: str,
        reasoning_model: str = None,
        response_model: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Two-stage generation with a reasoning step followed by a response generation step.
        
        Args:
            prompt: The user query
            reasoning_model: Model to use for reasoning (defaults to same as default_model)
            response_model: Model to use for final response (defaults to same as default_model)
            system_prompt: Optional system prompt for both stages
            temperature: Sampling temperature
            stream: Whether to stream the final response
            
        Returns:
            Final response string or generator if streaming
        """
        reasoning_model = reasoning_model or self.default_model
        response_model = response_model or self.default_model
        
        # First stage: Generate reasoning
        reasoning_prompt = f"""
        You are a reasoning assistant. Given the following query:
        
        ---
        {prompt}
        ---
        
        Think step by step about how to best approach and answer this query. 
        Consider what information is needed, what considerations are important, 
        and how to structure a comprehensive answer.
        
        Provide your reasoning process that would help formulate the best possible response.
        """
        
        try:
            reasoning = self.get_completion(
                prompt=reasoning_prompt,
                model=reasoning_model,
                system_prompt=system_prompt,
                temperature=temperature,
                stream=False
            )
            
            # Second stage: Generate final response using the reasoning
            response_prompt = f"""
            You are an expert assistant. A user has asked the following query:
            
            ---
            {prompt}
            ---
            
            Here is some careful reasoning about how to approach this query:
            
            {reasoning}
            
            Based on this reasoning, provide your best response to the user's query.
            Make your response clear, comprehensive, and directly addressing what the user asked.
            """
            
            # Get final response
            return self.get_completion(
                prompt=response_prompt,
                model=response_model,
                system_prompt=system_prompt,
                temperature=temperature,
                stream=stream
            )
        
        except Exception as e:
            # If reasoning fails, fall back to direct response
            st.warning(f"Reasoning step failed, falling back to direct response: {str(e)}")
            
            try:
                return self.get_completion(
                    prompt=prompt,
                    model=response_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    stream=stream
                )
            except Exception as fallback_error:
                st.error(f"Direct response also failed: {str(fallback_error)}")
                if stream:
                    def error_generator():
                        yield "Error generating response. Please check your API keys and model configuration."
                    return error_generator()
                else:
                    return "Error generating response. Please check your API keys and model configuration."