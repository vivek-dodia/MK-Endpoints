import os
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Enrich MikroTik API endpoints with detailed documentation")
    parser.add_argument("--extracted", default="output/extracted_endpoints.json", help="Path to extracted endpoints file")
    parser.add_argument("--output", default="output/enriched_endpoints.json", help="Path to output file")
    parser.add_argument("--checkpoint", default="output/enrichment_checkpoint.json", help="Path to checkpoint file")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of endpoints to process per batch")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds")
    parser.add_argument("--family", help="Only process endpoints in the specified API family")
    parser.add_argument("--provider", default="google", choices=["google", "deepseek"], 
                      help="LLM provider to use (google or deepseek)")
    parser.add_argument("--model", help="Model to use for enrichment. Default depends on --provider")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed API calls")
    parser.add_argument("--generate-multi-queries", action="store_true", help="Generate multi-endpoint queries after enrichment")
    parser.add_argument("--start-index", type=int, default=0, help="Start processing from this index")
    parser.add_argument("--end-index", type=int, default=-1, help="Stop processing at this index")
    return parser.parse_args()

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Initialize Google Gemini client (will only be used if provider=google)
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    print("Warning: google-generativeai package not installed. Run 'pip install google-generativeai' to use Gemini models.")

def get_client(provider, model_name=None):
    """Get the appropriate client based on provider."""
    if provider == "google":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return None  # We'll use the genai module directly
    elif provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_default_model(provider):
    """Get the default model name based on provider."""
    if provider == "google":
        return "gemini-2.0-flash"
    elif provider == "deepseek":
        return "deepseek-reasoner"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def extract_api_family(path):
    """Extract the API family from the endpoint path."""
    # Remove leading slash if present
    clean_path = path.lstrip('/')
    
    # Get the first segment of the path
    segments = clean_path.split('/')
    if not segments:
        return "general"
    
    # Handle special cases for better family grouping
    main_segment = segments[0]
    
    # If there's a second segment, it might help refine the family
    if len(segments) > 1:
        sub_segment = segments[1]
        # Some MikroTik API families are two levels deep (e.g., ip/firewall, interface/bridge)
        if main_segment in ["ip", "interface", "routing", "system"]:
            if sub_segment:
                return f"{main_segment}-{sub_segment}"
    
    return main_segment

def identify_related_endpoints(endpoint, all_endpoints):
    """Identify endpoints that are closely related to this one."""
    path = endpoint["path"]
    family = extract_api_family(path)
    
    # Find endpoints in the same family
    same_family_endpoints = []
    for ep in all_endpoints:
        if ep["path"] != path and extract_api_family(ep["path"]) == family:
            same_family_endpoints.append({
                "path": ep["path"],
                "relation": "same-family",
                "joint_use_case": f"Used together for comprehensive {family} management"
            })
    
    # Find parent/child relationships
    path_parts = path.strip('/').split('/')
    parent_endpoints = []
    if len(path_parts) > 1:
        # This is a potential parent path (less specific)
        potential_parent = '/' + '/'.join(path_parts[:-1])
        for ep in all_endpoints:
            if ep["path"] == potential_parent:
                parent_endpoints.append({
                    "path": ep["path"],
                    "relation": "parent",
                    "joint_use_case": f"Use {ep['path']} to get overview, then {path} for details"
                })
    
    # Find child relationships (more specific)
    child_endpoints = []
    for ep in all_endpoints:
        ep_path = ep["path"]
        if ep_path != path and ep_path.startswith(path + "/"):
            child_endpoints.append({
                "path": ep_path,
                "relation": "child",
                "joint_use_case": f"Use {path} first, then {ep_path} for specific operations"
            })
    
    # Find complementary endpoints (different family but often used together)
    # We'll identify these using common patterns in MikroTik API
    complementary_mappings = {
        "interface": ["bridge", "ethernet", "wireless", "ip-address"],
        "ip-address": ["interface", "ip-route", "ip-dns"],
        "ip-arp": ["interface", "ip-address", "bridge-host"],
        "ip-route": ["interface", "ip-address", "routing-bgp"],
        "bridge": ["interface", "bridge-port", "bridge-vlan"],
        "firewall": ["ip-firewall", "ip-address", "interface"],
        "system": ["resource", "identity", "clock"],
        "wireless": ["interface-wireless", "wireless-security"],
        "snmp": ["system", "ip-address"],
        "dhcp-server": ["ip-address", "ip-dhcp-client"],
        "certificate": ["system-identity", "ip-service"]
    }
    
    related_families = complementary_mappings.get(family, [])
    
    # Add endpoints from complementary families
    complementary_endpoints = []
    for ep in all_endpoints:
        ep_family = extract_api_family(ep["path"])
        if ep_family in related_families:
            use_case = "Used together for cross-functional management"
            
            # Customize joint use cases based on specific family relationships
            if family == "interface" and ep_family == "ip-address":
                use_case = "View interfaces and their assigned IP addresses together"
            elif family == "bridge" and ep_family == "interface":
                use_case = "View bridge status along with associated interfaces"
            elif family == "ip-arp" and ep_family == "interface":
                use_case = "Match ARP entries with interface information"
                
            complementary_endpoints.append({
                "path": ep["path"],
                "relation": "complementary",
                "joint_use_case": use_case
            })
    
    # Limit to a reasonable number of related endpoints
    same_family_endpoints = same_family_endpoints[:5]  # Top 5 same family
    parent_endpoints = parent_endpoints[:2]  # Top 2 parents
    child_endpoints = child_endpoints[:3]  # Top 3 children
    complementary_endpoints = complementary_endpoints[:5]  # Top 5 complementary
    
    return {
        "same_family": same_family_endpoints,
        "parent": parent_endpoints,
        "child": child_endpoints,
        "complementary": complementary_endpoints
    }

def create_enrichment_prompt(endpoint, all_endpoints):
    """Create a prompt for the LLM to enrich an endpoint with distinctive characteristics."""
    path = endpoint["path"]
    parameters = endpoint.get("parameters", [])
    operation_id = endpoint.get("operation_id", "")
    responses = endpoint.get("responses", {})
    
    # Extract the API family from the path
    api_family = extract_api_family(path)
    
    # Get related endpoints
    related_endpoints = identify_related_endpoints(endpoint, all_endpoints)
    
    # Format related endpoints for prompt
    same_family_info = "\n".join([f"- {ep['path']} ({ep['joint_use_case']})" for ep in related_endpoints["same_family"][:3]])
    parent_info = "\n".join([f"- {ep['path']} ({ep['joint_use_case']})" for ep in related_endpoints["parent"][:2]])
    child_info = "\n".join([f"- {ep['path']} ({ep['joint_use_case']})" for ep in related_endpoints["child"][:2]])
    complementary_info = "\n".join([f"- {ep['path']} ({ep['joint_use_case']})" for ep in related_endpoints["complementary"][:3]])
    
    related_info = f"""
    Similar endpoints in the same family:
    {same_family_info if same_family_info else "None identified"}
    
    Parent endpoints:
    {parent_info if parent_info else "None identified"}
    
    Child endpoints:
    {child_info if child_info else "None identified"}
    
    Complementary endpoints that are often used together with this one:
    {complementary_info if complementary_info else "None identified"}
    """
    
    # Format example response structure based on the responses field
    response_info = ""
    if responses and "200" in responses:
        response_schema = responses["200"].get("content", {}).get("application/json", {}).get("schema", {})
        response_info = f"Response structure: {json.dumps(response_schema, indent=2)}"
    
    prompt = f"""
    You are a MikroTik networking expert. Create comprehensive and DISTINCTIVE documentation for this API endpoint.
    
    API Endpoint: {path}
    API Family: {api_family}
    HTTP Method: GET
    Operation ID: {operation_id}
    Parameters: {json.dumps(parameters, indent=2)}
    {response_info}
    
    Related Endpoints Information:
    {related_info}
    
    IMPORTANT: Your documentation MUST clearly differentiate this endpoint from others, especially those in different API families. Avoid generic networking terms that could apply to many endpoints.
    
    Based on MikroTik's networking architecture, provide:
    
    1. A concise summary (1 sentence) that specifically identifies what UNIQUE information this endpoint provides
    2. A detailed description (2-3 sentences) explaining EXACTLY what information this endpoint returns and how it DIFFERS from similar endpoints
    3. 4-5 common use cases that are SPECIFIC to this endpoint (concrete scenarios where a network engineer would use THIS endpoint rather than others)
    4. 5-7 DISTINCTIVE networking tags or keywords that represent this endpoint's UNIQUE function (avoid generic terms like "configuration" or "management" unless qualified with specifics)
    5. Description for each parameter
    6. Example value for each parameter (realistic values a network engineer might use)
    7. A list of "related_endpoints" that are commonly used with this endpoint (based on the information provided above), including a brief explanation of why they're related
    8. A list of "queries" containing 3-4 example natural language questions that would specifically need this endpoint
    9. A simplified example response that this endpoint might return (JSON format)
    
    Always include the API family ("{api_family}") in the tags to help distinguish this endpoint from others.
    
    Respond in JSON format with these fields: 
    - summary
    - description
    - use_cases (array)
    - tags (array)
    - parameter_descriptions (object with parameter names as keys)
    - parameter_examples (object with parameter names as keys and example values)
    - related_endpoints (array of objects with "path", "reason" and "joint_use_case" properties)
    - queries (array of example questions)
    - response_example (a simplified JSON example of what the endpoint might return)
    """
    return prompt

def validate_enrichment(enriched_endpoint):
    """Validate the enriched endpoint data quality."""
    required_fields = ["summary", "description", "use_cases", "tags", 
                       "parameter_descriptions", "related_endpoints", "queries"]
    
    # Check all required fields exist
    missing_fields = [field for field in required_fields if field not in enriched_endpoint]
    if missing_fields:
        print(f"Warning: Missing fields for {enriched_endpoint['path']}: {', '.join(missing_fields)}")
        
    # Check for minimum content in text fields
    if len(enriched_endpoint.get("summary", "")) < 10:
        print(f"Warning: Summary too short for {enriched_endpoint['path']}")
    
    # Check for minimum number of use cases
    if len(enriched_endpoint.get("use_cases", [])) < 2:
        print(f"Warning: Too few use cases for {enriched_endpoint['path']}")
    
    # Check for minimum number of tags
    if len(enriched_endpoint.get("tags", [])) < 3:
        print(f"Warning: Too few tags for {enriched_endpoint['path']}")
    
    # Check for minimum related endpoints
    if len(enriched_endpoint.get("related_endpoints", [])) < 1:
        print(f"Warning: No related endpoints for {enriched_endpoint['path']}")
    
    # Check for example queries
    if len(enriched_endpoint.get("queries", [])) < 2:
        print(f"Warning: Too few example queries for {enriched_endpoint['path']}")
    
    return len(missing_fields) == 0

def enrich_endpoint_with_gemini(endpoint, all_endpoints, model_name, max_retries=3):
    """Enrich a single endpoint using Google's Gemini model."""
    prompt = create_enrichment_prompt(endpoint, all_endpoints)
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff for retry timing
            if attempt > 0:
                backoff_time = 2 ** attempt 
                print(f"Retrying {endpoint['path']} in {backoff_time} seconds...")
                time.sleep(backoff_time)
            
            # Set up generation config
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            
            # Create model and generate content
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            response = model.generate_content(prompt)
            
            # Extract text response
            response_text = response.text
            
            # The model might return the JSON with markdown code blocks, clean that up
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON response
            enrichment = json.loads(response_text)
            
            # Add API family explicitly
            api_family = extract_api_family(endpoint["path"])
            enrichment["api_family"] = api_family
            
            # Make sure the API family is in the tags
            if "tags" in enrichment and api_family not in enrichment["tags"]:
                enrichment["tags"].append(api_family)
            
            # Combine original endpoint data with enrichment
            enriched_endpoint = {
                **endpoint,
                **enrichment
            }
            
            # Validate the enriched endpoint
            is_valid = validate_enrichment(enriched_endpoint)
            if not is_valid:
                print(f"Warning: Enrichment validation failed for {endpoint['path']}, but continuing with available data")
            
            return enriched_endpoint
        except Exception as e:
            print(f"Error enriching {endpoint['path']} with Gemini (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Failed to enrich {endpoint['path']} after {max_retries} attempts")
    
    # If all retries fail, return original endpoint with empty enrichment
    api_family = extract_api_family(endpoint["path"])
    return {
        **endpoint,
        "summary": f"GET endpoint for {endpoint['path']}",
        "description": f"Retrieves information from {endpoint['path']}",
        "use_cases": ["General network management"],
        "tags": ["mikrotik", "api", "get", api_family],
        "parameter_descriptions": {},
        "parameter_examples": {},
        "api_family": api_family,
        "related_endpoints": [],
        "queries": [f"Get information from {endpoint['path']}"],
        "response_example": {"status": "success", "data": []}
    }

def enrich_endpoint_with_deepseek(client, endpoint, all_endpoints, model_name, max_retries=3):
    """Enrich a single endpoint using DeepSeek's model."""
    prompt = create_enrichment_prompt(endpoint, all_endpoints)
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff for retry timing
            if attempt > 0:
                backoff_time = 2 ** attempt 
                print(f"Retrying {endpoint['path']} in {backoff_time} seconds...")
                time.sleep(backoff_time)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a MikroTik networking expert who provides detailed, accurate API documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.1
            )
            
            # Extract text response
            response_text = response.choices[0].message.content
            
            # The model might return the JSON with markdown code blocks, clean that up
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON response
            enrichment = json.loads(response_text)
            
            # Add API family explicitly
            api_family = extract_api_family(endpoint["path"])
            enrichment["api_family"] = api_family
            
            # Make sure the API family is in the tags
            if "tags" in enrichment and api_family not in enrichment["tags"]:
                enrichment["tags"].append(api_family)
            
            # Combine original endpoint data with enrichment
            enriched_endpoint = {
                **endpoint,
                **enrichment
            }
            
            # Validate the enriched endpoint
            is_valid = validate_enrichment(enriched_endpoint)
            if not is_valid:
                print(f"Warning: Enrichment validation failed for {endpoint['path']}, but continuing with available data")
            
            return enriched_endpoint
        except Exception as e:
            print(f"Error enriching {endpoint['path']} with DeepSeek (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Failed to enrich {endpoint['path']} after {max_retries} attempts")
    
    # If all retries fail, return original endpoint with empty enrichment
    api_family = extract_api_family(endpoint["path"])
    return {
        **endpoint,
        "summary": f"GET endpoint for {endpoint['path']}",
        "description": f"Retrieves information from {endpoint['path']}",
        "use_cases": ["General network management"],
        "tags": ["mikrotik", "api", "get", api_family],
        "parameter_descriptions": {},
        "parameter_examples": {},
        "api_family": api_family,
        "related_endpoints": [],
        "queries": [f"Get information from {endpoint['path']}"],
        "response_example": {"status": "success", "data": []}
    }

def enrich_endpoint(endpoint, all_endpoints, provider="google", model_name=None, client=None, max_retries=3):
    """Enrich a single endpoint using the specified provider and model."""
    if provider == "google":
        return enrich_endpoint_with_gemini(endpoint, all_endpoints, model_name, max_retries)
    elif provider == "deepseek":
        return enrich_endpoint_with_deepseek(client, endpoint, all_endpoints, model_name, max_retries)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def save_checkpoint(enriched_endpoints, checkpoint_file, output_file):
    """Save current progress to both checkpoint and final output files."""
    # Save to checkpoint file
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_endpoints, f, indent=2)
    
    # Also save to final output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_endpoints, f, indent=2)
    
    print(f"Saved {len(enriched_endpoints)} endpoints to checkpoint and output files")

def estimate_completion_time(start_time, processed_count, total_count):
    """Estimate completion time based on current progress."""
    if processed_count == 0:
        return "unknown"
    
    elapsed = time.time() - start_time
    items_per_sec = processed_count / elapsed
    remaining_items = total_count - processed_count
    
    # Handle division by zero
    if items_per_sec <= 0:
        return "calculating..."
        
    remaining_seconds = remaining_items / items_per_sec
    
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    eta_time = datetime.now() + timedelta(seconds=remaining_seconds)
    eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S")
    
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s (ETA: {eta_str})"

def generate_multi_endpoint_queries(enriched_endpoints, provider, model_name, client=None, 
                                  output_file="output/multi_endpoint_queries.json"):
    """Generate sample multi-endpoint queries for testing the correlation logic."""
    print("\nGenerating multi-endpoint query examples...")
    
    # Group endpoints by family
    endpoints_by_family = {}
    for ep in enriched_endpoints:
        family = ep.get("api_family", "unknown")
        if family not in endpoints_by_family:
            endpoints_by_family[family] = []
        endpoints_by_family[family].append(ep)
    
    multi_endpoint_queries = []
    
    # Get largest families (those with most endpoints)
    largest_families = sorted(
        [(family, len(endpoints)) for family, endpoints in endpoints_by_family.items()],
        key=lambda x: x[1], 
        reverse=True
    )[:10]  # Top 10 families
    
    print(f"Generating queries for top {len(largest_families)} API families...")
    
    # Generate queries for each major family
    for family, count in largest_families:
        if count < 3:  # Need at least 3 endpoints for meaningful multi-endpoint queries
            continue
            
        endpoints = endpoints_by_family[family]
        sample_endpoints = random.sample(endpoints, min(5, len(endpoints)))
        
        endpoint_info = "\n".join([
            f"- {ep['path']}: {ep.get('summary', 'No summary available')}" 
            for ep in sample_endpoints
        ])
        
        # Also include some complementary endpoints from other families
        complementary_endpoints = []
        for ep in sample_endpoints:
            for related in ep.get("related_endpoints", []):
                if related.get("relation") == "complementary":
                    complementary_endpoints.append(related["path"])
        
        complementary_info = ""
        if complementary_endpoints:
            complementary_info = "Related endpoints from other families:\n" + "\n".join([
                f"- {path}" for path in complementary_endpoints[:5]
            ])
        
        prompt = f"""
        You are a MikroTik networking expert. Generate 5 natural language queries about {family} that would require information from multiple API endpoints to answer completely.
        
        Example: "Show me all DHCP leases and which interfaces they're connected to" (requires both DHCP and interface endpoints)
        
        Available endpoints in the {family} family:
        {endpoint_info}
        
        {complementary_info}
        
        For each query, specify which endpoints would be needed to answer it completely, and explain briefly how the data would be combined.
        
        Return a JSON array of objects with:
        - "query": The natural language question a network engineer might ask
        - "required_endpoints": Array of endpoint paths needed to answer this query  
        - "correlation_logic": Brief explanation of how data from these endpoints would be combined
        """
        
        try:
            print(f"Generating queries for {family} family...")
            
            # Handle different providers
            if provider == "google":
                # Set up generation config
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
                
                # Create model and generate content
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                response = model.generate_content(prompt)
                response_text = response.text
            elif provider == "deepseek":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a MikroTik networking expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
            
            # Clean up response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            family_queries = json.loads(response_text)
            
            # Add family information
            for query in family_queries:
                query["family"] = family
            
            multi_endpoint_queries.extend(family_queries)
            print(f"Added {len(family_queries)} queries for {family} family")
            
            # Add small delay between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating queries for {family}: {e}")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(multi_endpoint_queries, f, indent=2)
        
    print(f"Generated {len(multi_endpoint_queries)} multi-endpoint queries saved to {output_file}")
    return multi_endpoint_queries

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set up file paths
    extracted_file = args.extracted
    enriched_file = args.output
    checkpoint_file = args.checkpoint
    batch_size = args.batch_size
    delay = args.delay
    provider = args.provider
    max_retries = args.max_retries
    
    # Get default model name for the provider
    model_name = args.model if args.model else get_default_model(provider)
    
    print(f"Starting endpoint enrichment process with DISTINCTIVE family-specific documentation")
    print(f"Using provider: {provider}, model: {model_name}")
    
    # Initialize client based on provider
    client = get_client(provider, model_name)
    
    # Create output directory if it doesn't exist
    Path(os.path.dirname(enriched_file)).mkdir(parents=True, exist_ok=True)
    
    # Check if extracted endpoints file exists
    if not os.path.exists(extracted_file):
        print(f"ERROR: Extracted endpoints file not found at {extracted_file}")
        return
    
    # Load extracted endpoints
    print(f"Loading extracted endpoints from {extracted_file}")
    with open(extracted_file, 'r', encoding='utf-8') as f:
        all_endpoints = json.load(f)
    
    print(f"Loaded {len(all_endpoints)} endpoints to enrich")
    
    # Initialize enriched endpoints list
    enriched_endpoints = []
    
    # Check if checkpoint exists and load it
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            enriched_endpoints = json.load(f)
        
        # Find which endpoints are already processed
        processed_paths = {ep["path"] for ep in enriched_endpoints}
        endpoints_to_process = [ep for ep in all_endpoints if ep["path"] not in processed_paths]
        
        print(f"Loaded checkpoint with {len(enriched_endpoints)} already processed endpoints")
        print(f"Remaining endpoints to process: {len(endpoints_to_process)}")
    else:
        endpoints_to_process = all_endpoints
        print(f"No checkpoint found. Processing all {len(all_endpoints)} endpoints")
    
    # Filter by family if specified
    if args.family:
        endpoints_to_process = [ep for ep in endpoints_to_process 
                              if extract_api_family(ep["path"]) == args.family]
        print(f"Filtered to {len(endpoints_to_process)} endpoints in the '{args.family}' family")
    
    # Apply start and end index if specified
    if args.end_index > 0:
        endpoints_to_process = endpoints_to_process[args.start_index:args.end_index]
        print(f"Processing endpoints from index {args.start_index} to {args.end_index} ({len(endpoints_to_process)} endpoints)")
    elif args.start_index > 0:
        endpoints_to_process = endpoints_to_process[args.start_index:]
        print(f"Processing endpoints from index {args.start_index} ({len(endpoints_to_process)} endpoints)")
    
    # Exit if no endpoints to process
    if not endpoints_to_process:
        print("No endpoints to process. Exiting.")
        
        # Generate multi-endpoint queries if requested
        if args.generate_multi_queries and enriched_endpoints:
            generate_multi_endpoint_queries(enriched_endpoints, provider, model_name, client,
                                          f"output/multi_endpoint_queries.json")
        return
    
    # Track start time for ETA calculation
    start_time = time.time()
    total_to_process = len(endpoints_to_process)
    
    # Process endpoints in batches
    batch_size = min(batch_size, len(endpoints_to_process))
    for i in range(0, len(endpoints_to_process), batch_size):
        batch = endpoints_to_process[i:i+batch_size]
        
        # Calculate progress and ETA
        processed_so_far = len(enriched_endpoints)
        progress_pct = processed_so_far / (processed_so_far + total_to_process) * 100
        eta = estimate_completion_time(start_time, i, total_to_process)
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(endpoints_to_process)-1)//batch_size + 1}")
        print(f"Progress: {processed_so_far}/{processed_so_far + total_to_process} endpoints ({progress_pct:.1f}%)")
        print(f"Estimated time remaining: {eta}")
        
        for endpoint in tqdm(batch, desc="Enriching endpoints"):
            enriched = enrich_endpoint(endpoint, all_endpoints, provider, model_name, client, max_retries)
            enriched_endpoints.append(enriched)
            time.sleep(delay)  # Add delay to avoid rate limits
        
        # Save checkpoint after each batch
        save_checkpoint(enriched_endpoints, checkpoint_file, enriched_file)
    
    print(f"\nEnrichment completed successfully!")
    print(f"Enriched {len(enriched_endpoints)} endpoints")
    print(f"Final output saved to {enriched_file}")
    
    # Generate multi-endpoint queries if requested
    if args.generate_multi_queries:
        generate_multi_endpoint_queries(enriched_endpoints, provider, model_name, client,
                                      f"output/multi_endpoint_queries.json")

if __name__ == "__main__":
    main()