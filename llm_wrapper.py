"""
LLM wrapper module for MikroTik API Finder using direct API calls to Gemini and DeepSeek.
No dependencies on LiteLLM - just pure API calls.
"""

import os
import json
import time
import warnings
import requests
from typing import List, Dict, Any, Union, Generator, Optional
import streamlit as st

# Silence unnecessary warnings
warnings.filterwarnings("ignore", message="No secrets found")
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

class LLMWrapper:
    """
    A wrapper for LLM interactions using direct API calls.
    Primary: Gemini 2.0 Flash-Lite
    Fallback: DeepSeek Chat
    """
    
    def __init__(
        self, 
        default_model: str = "gemini-flash",
        fallbacks: List[str] = None,
    ):
        """
        Initialize the LLM wrapper for direct API calls.
        
        Args:
            default_model: The default model to use (without provider prefix)
            fallbacks: List of fallback models if the primary fails
        """
        self.default_model = default_model
        self.fallbacks = fallbacks or ["deepseek-chat"]
        
        # Load API keys from environment variables
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        # Track available models
        self.available_models = []
        
        # Check and validate API keys
        if self.google_api_key:
            print("Google API key found.")
            self.available_models.extend([
                "gemini-flash", 
                "gemini-pro"
            ])
        else:
            print("WARNING: No Google API key found. Gemini models won't be available.")
        
        if self.deepseek_api_key:
            print("DeepSeek API key found.")
            self.available_models.extend([
                "deepseek-chat", 
                "deepseek-coder"
            ])
        else:
            print("WARNING: No DeepSeek API key found. DeepSeek models won't be available.")
            
        print(f"Available models: {self.available_models}")
    
    def _call_gemini_api(self, messages, model="gemini-flash", temperature=0.7, max_tokens=1000, stream=False):
        """Make a direct call to the Gemini API"""
        # Map model names to Gemini API model names
        model_map = {
            "gemini-flash": "gemini-1.5-flash-latest",
            "gemini-pro": "gemini-1.5-pro-latest"
        }
        
        # Get the correct model name
        gemini_model = model_map.get(model, "gemini-1.5-flash-latest")
        
        # Gemini API endpoint
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System instruction: {msg['content']}"}]
                })
            else:
                gemini_messages.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [{"text": msg["content"]}]
                })
        
        # Prepare request data
        data = {
            "contents": gemini_messages,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.google_api_key
        }
        
        try:
            if stream:
                # For streaming, append ?alt=sse to the URL
                stream_url = f"{api_url}?alt=sse"
                response = requests.post(
                    stream_url,
                    headers=headers,
                    json=data,
                    stream=True
                )
                
                # Check for error
                if response.status_code != 200:
                    error_msg = f"Gemini API Error ({response.status_code}): {response.text}"
                    raise Exception(error_msg)
                
                # Return generator for streaming
                return self._process_gemini_stream(response)
            else:
                # Regular response
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data
                )
                
                # Check for error
                if response.status_code != 200:
                    error_msg = f"Gemini API Error ({response.status_code}): {response.text}"
                    raise Exception(error_msg)
                
                # Parse response
                response_data = response.json()
                
                # Extract text from response
                try:
                    text_content = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    return text_content
                except (KeyError, IndexError) as e:
                    raise Exception(f"Unexpected Gemini API response format: {e}")
                
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def _process_gemini_stream(self, response):
        """Process streaming response from Gemini API"""
        # Read SSE events from response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        json_data = json.loads(data)
                        if "candidates" in json_data and json_data["candidates"]:
                            candidate = json_data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        yield part["text"]
                    except json.JSONDecodeError:
                        continue
    
    def _call_deepseek_api(self, messages, model="deepseek-chat", temperature=0.7, max_tokens=1000, stream=False):
        """Direct call to DeepSeek API"""
        # Map model names to DeepSeek API model names
        model_map = {
            "deepseek-chat": "deepseek-chat",
            "deepseek-coder": "deepseek-coder"
        }
        
        # Get the correct model name
        deepseek_model = model_map.get(model, "deepseek-chat")
        
        # DeepSeek API endpoint
        api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Prepare request data
        data = {
            "model": deepseek_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
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
                    raise Exception(error_msg)
                
                # Return generator for streaming
                return self._process_deepseek_stream(response)
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
                    raise Exception(error_msg)
                
                # Parse response
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
                
        except Exception as e:
            raise Exception(f"DeepSeek API call failed: {str(e)}")
    
    def _process_deepseek_stream(self, response):
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
        Get a completion from the LLM using direct API calls.
        
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
        try:
            model = model or self.default_model
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Use the appropriate API based on the model
            if model.startswith("gemini"):
                # Use Gemini API
                if not self.google_api_key:
                    raise Exception("No Google API key found. Cannot use Gemini models.")
                    
                return self._call_gemini_api(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            elif model.startswith("deepseek"):
                # Use DeepSeek API
                if not self.deepseek_api_key:
                    raise Exception("No DeepSeek API key found. Cannot use DeepSeek models.")
                    
                return self._call_deepseek_api(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            else:
                raise Exception(f"Unknown model: {model}")
                
        except Exception as e:
            st.warning(f"Primary model {model} failed: {str(e)}")
            
            # Try fallback models
            if model == self.default_model and self.fallbacks:
                for fallback in self.fallbacks:
                    try:
                        st.info(f"Trying fallback model: {fallback}")
                        
                        if fallback.startswith("gemini"):
                            # Use Gemini API
                            if not self.google_api_key:
                                continue  # Skip if no API key
                                
                            return self._call_gemini_api(
                                messages=messages,
                                model=fallback,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=stream
                            )
                        elif fallback.startswith("deepseek"):
                            # Use DeepSeek API
                            if not self.deepseek_api_key:
                                continue  # Skip if no API key
                                
                            return self._call_deepseek_api(
                                messages=messages,
                                model=fallback,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=stream
                            )
                        else:
                            st.warning(f"Unknown fallback model: {fallback}")
                            continue
                            
                    except Exception as fallback_error:
                        st.warning(f"Fallback {fallback} failed: {str(fallback_error)}")
                        continue
            
            # If all attempts fail, raise the error
            st.error(f"All LLM completion attempts failed. Last error: {str(e)}")
            raise Exception(f"All LLM completion attempts failed. Last error: {str(e)}")

    def generate_json(
        self, 
        prompt: str, 
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.2,
    ) -> Union[Dict[str, Any], List[str]]:
        """
        Get a JSON response from the LLM.
        
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
            # Try to generate reasoning with primary model
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