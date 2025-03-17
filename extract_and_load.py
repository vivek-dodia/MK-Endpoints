import os
import yaml
import json
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configure OpenAI client for DeepSeek
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# Qdrant configuration
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "mikrotik-GET"  # Custom collection name as requested

# Initialize directories
Path("data").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

def load_openapi_spec(file_path):
    """Load the OpenAPI spec from YAML file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def extract_get_endpoints(openapi_spec):
    """Extract all GET endpoints from the OpenAPI spec."""
    get_endpoints = []
    
    # Debug: print the first few keys of the spec
    print(f"First 5 keys in the YAML: {list(openapi_spec.keys())[:5]}")
    
    # If it's a complete OpenAPI spec, paths might be under a 'paths' key
    paths_obj = openapi_spec.get('paths', openapi_spec)
    
    # Iterate through each path and look for GET methods
    for path, path_item in paths_obj.items():
        if not isinstance(path_item, dict):
            continue
            
        # Debug the path structure for the first few paths
        if len(get_endpoints) < 2 and 'get' in path_item:
            print(f"Example path structure for '{path}': {path_item.keys()}")
        
        # Check for 'get' method
        if 'get' in path_item:
            get_method = path_item['get']
            
            # Extract relevant information
            endpoint_info = {
                "path": path,
                "method": "GET",
                "operation_id": get_method.get("operationId", f"GET_{path.replace('/', '_')}"),
                "parameters": get_method.get("parameters", []),
                "responses": get_method.get("responses", {}),
                # Store original spec for reference
                "original_spec": get_method
            }
            get_endpoints.append(endpoint_info)
    
    return get_endpoints

def create_enrichment_prompt(endpoint):
    """Create a prompt for the LLM to enrich an endpoint."""
    path = endpoint["path"]
    parameters = endpoint.get("parameters", [])
    operation_id = endpoint.get("operation_id", "")
    
    prompt = f"""
    You are a MikroTik networking expert. Create comprehensive documentation for this API endpoint.
    
    API Endpoint: {path}
    HTTP Method: GET
    Operation ID: {operation_id}
    Parameters: {json.dumps(parameters, indent=2)}
    
    Based on MikroTik's networking architecture, provide:
    
    1. A concise summary (1 sentence)
    2. A detailed description (2-3 sentences explaining what information this endpoint returns)
    3. 4-5 common use cases for this endpoint (specific scenarios where a network engineer would use it)
    4. 5-7 relevant networking tags or keywords
    5. Description for each parameter
    
    Respond in JSON format with these fields: summary, description, use_cases (array), 
    tags (array), parameter_descriptions (object with parameter names as keys)
    """
    return prompt

def enrich_endpoint(endpoint, max_retries=3):
    """Enrich a single endpoint using DeepSeek R1 model."""
    prompt = create_enrichment_prompt(endpoint)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
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
            
            # Combine original endpoint data with enrichment
            enriched_endpoint = {
                **endpoint,
                **enrichment
            }
            
            return enriched_endpoint
        except Exception as e:
            print(f"Error enriching {endpoint['path']} (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)  # Wait before retry
    
    # If all retries fail, return original endpoint with empty enrichment
    print(f"Failed to enrich {endpoint['path']} after {max_retries} attempts")
    return {
        **endpoint,
        "summary": f"GET endpoint for {endpoint['path']}",
        "description": f"Retrieves information from {endpoint['path']}",
        "use_cases": ["General network management"],
        "tags": ["mikrotik", "api", "get"],
        "parameter_descriptions": {}
    }

def create_text_representation(endpoint):
    """Create a text representation of an endpoint for embedding."""
    path = endpoint["path"]
    summary = endpoint.get("summary", "")
    description = endpoint.get("description", "")
    use_cases = " ".join(endpoint.get("use_cases", []))
    tags = " ".join(endpoint.get("tags", []))
    
    return f"{path} {summary} {description} {use_cases} {tags}"

def setup_qdrant_collection(client, collection_name, vector_size):
    """Setup or validate Qdrant collection."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Collection {collection_name} already exists")
        return
    
    print(f"Creating collection {collection_name}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

def main():
    print("Starting MikroTik GET endpoint extraction and enrichment")
    
    # 1. Load and extract GET endpoints
    print("Loading OpenAPI spec from data/mikrotik_openapi.yaml")
    try:
        openapi_spec = load_openapi_spec("data/mikrotik_openapi.yaml")
        print(f"YAML loaded successfully, got {len(openapi_spec)} top-level keys")
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return
    
    print("Extracting GET endpoints")
    get_endpoints = extract_get_endpoints(openapi_spec)
    print(f"Found {len(get_endpoints)} GET endpoints")
    
    if len(get_endpoints) == 0:
        print("WARNING: No GET endpoints found. Please check your YAML structure.")
        print("Exiting process.")
        return
    
    # Save extracted endpoints
    print("Saving extracted endpoints to output/extracted_endpoints.json")
    with open("output/extracted_endpoints.json", "w", encoding="utf-8") as f:
        json.dump(get_endpoints, f, indent=2)
    
    # 2. Enrich endpoints
    print("Enriching endpoints with DeepSeek R1")
    enriched_endpoints = []
    
    # Process in batches with saving checkpoints
    batch_size = 10
    checkpoint_file = "output/enrichment_checkpoint.json"
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            enriched_endpoints = json.load(f)
        processed_paths = {ep["path"] for ep in enriched_endpoints}
        endpoints_to_process = [ep for ep in get_endpoints if ep["path"] not in processed_paths]
        print(f"Loaded checkpoint with {len(enriched_endpoints)} already processed endpoints")
        print(f"Remaining endpoints to process: {len(endpoints_to_process)}")
    else:
        endpoints_to_process = get_endpoints
    
    # Process remaining endpoints
    for i in tqdm(range(0, len(endpoints_to_process), batch_size), desc="Enriching endpoints (batches)"):
        batch = endpoints_to_process[i:i+batch_size]
        
        for endpoint in tqdm(batch, desc=f"Batch {i//batch_size+1}", leave=False):
            enriched = enrich_endpoint(endpoint)
            enriched_endpoints.append(enriched)
            time.sleep(1)  # Add delay to avoid rate limits
        
        # Save checkpoint after each batch
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_endpoints, f, indent=2)
    
    # Save final enriched endpoints
    print("Saving enriched endpoints to output/enriched_endpoints.json")
    with open("output/enriched_endpoints.json", "w", encoding="utf-8") as f:
        json.dump(enriched_endpoints, f, indent=2)
    
    # 3. Generate embeddings
    print("Generating embeddings")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create text representations and generate embeddings
    texts = [create_text_representation(endpoint) for endpoint in enriched_endpoints]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 4. Store in Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Setup collection
    vector_size = len(embeddings[0])
    setup_qdrant_collection(qdrant_client, COLLECTION_NAME, vector_size)
    
    # Prepare points for upload
    points = []
    for i, (endpoint, embedding) in enumerate(zip(enriched_endpoints, embeddings)):
        # Remove embedding from payload to save space
        payload = {k: v for k, v in endpoint.items() if k != "embedding"}
        
        point = models.PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload=payload
        )
        points.append(point)
    
    # Upload points in batches
    upload_batch_size = 100
    for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading to Qdrant"):
        batch = points[i:i+upload_batch_size]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
    
    print(f"Successfully uploaded {len(points)} endpoints to Qdrant collection '{COLLECTION_NAME}'")
    print("Process completed successfully!")

if __name__ == "__main__":
    main()