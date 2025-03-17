import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Load enriched endpoints to Qdrant vector database")
    parser.add_argument("--input", default="output/enriched_endpoints.json", help="Path to enriched endpoints file")
    parser.add_argument("--collection", default="mikrotik-GET", help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Upload batch size")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model for embeddings")
    parser.add_argument("--recreate", action="store_true", help="Force recreation of collection if it exists")
    return parser.parse_args()

def create_text_representation(endpoint):
    """Create a rich text representation of an endpoint for embedding."""
    path = endpoint["path"]
    api_family = endpoint.get("api_family", "")
    summary = endpoint.get("summary", "")
    description = endpoint.get("description", "")
    use_cases = " ".join(endpoint.get("use_cases", []))
    tags = " ".join(endpoint.get("tags", []))
    queries = " ".join(endpoint.get("queries", []))
    
    # Build a rich representation that incorporates all the enriched data
    # The more context we include, the better the vector search will perform
    text = f"""
    PATH: {path}
    FAMILY: {api_family}
    SUMMARY: {summary}
    DESCRIPTION: {description}
    USE CASES: {use_cases}
    TAGS: {tags}
    QUERIES: {queries}
    """
    return text

def setup_qdrant_collection(client, collection_name, vector_size, recreate=False):
    """Setup or validate Qdrant collection."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Collection {collection_name} already exists")
        if recreate:
            print(f"Recreating collection {collection_name}...")
            client.delete_collection(collection_name=collection_name)
            create_collection(client, collection_name, vector_size)
    else:
        create_collection(client, collection_name, vector_size)

def create_collection(client, collection_name, vector_size):
    """Create a new Qdrant collection with optimized settings."""
    print(f"Creating collection {collection_name} with vector size {vector_size}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,  # Index immediately for small collections
        )
    )
    
    print("Creating payload indexes for faster filtering...")
    # Create a payload index on the api_family field for faster filtering
    client.create_payload_index(
        collection_name=collection_name,
        field_name="api_family",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
    # Create a payload index on the path field
    client.create_payload_index(
        collection_name=collection_name,
        field_name="path",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
    # Create a payload index on the tags field
    client.create_payload_index(
        collection_name=collection_name,
        field_name="tags",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    input_file = args.input
    collection_name = args.collection
    batch_size = args.batch_size
    model_name = args.model
    recreate = args.recreate
    
    # Check Qdrant credentials
    QDRANT_URL = os.environ.get("QDRANT_URL")
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return
    
    print(f"Loading enriched endpoints from {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Enriched endpoints file not found at {input_file}")
        return
    
    # Load enriched endpoints
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            enriched_endpoints = json.load(f)
        
        print(f"Loaded {len(enriched_endpoints)} enriched endpoints")
    except Exception as e:
        print(f"ERROR: Failed to load enriched endpoints: {e}")
        return
    
    # Initialize sentence transformer model
    print(f"Loading sentence transformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Make sure you have installed sentence-transformers: pip install sentence-transformers")
        return
    
    # Create text representations for embedding
    print("Creating text representations for embedding...")
    texts = [create_text_representation(endpoint) for endpoint in tqdm(enriched_endpoints)]
    
    # Generate embeddings
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Test connection
        client.get_collections()
    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant: {e}")
        return
    
    # Setup collection
    vector_size = len(embeddings[0])
    setup_qdrant_collection(client, collection_name, vector_size, recreate)
    
    # Prepare points for upload
    print("Preparing points for upload...")
    points = []
    for i, (endpoint, embedding) in enumerate(zip(enriched_endpoints, embeddings)):
        # Use path hash as ID for deterministic IDs
        point_id = abs(hash(endpoint["path"])) % (2**63 - 1)
        
        point = models.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=endpoint
        )
        points.append(point)
    
    # Upload points in batches
    print(f"Uploading {len(points)} points to Qdrant in batches of {batch_size}...")
    for i in tqdm(range(0, len(points), batch_size)):
        batch = points[i:i+batch_size]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
        except Exception as e:
            print(f"ERROR uploading batch {i//batch_size + 1}: {e}")
            continue
    
    # Count points in collection to verify
    try:
        count_result = client.count(collection_name=collection_name)
        print(f"Uploaded {count_result.count} points to Qdrant collection '{collection_name}'")
        
        # Run a test query
        test_query = "Show interfaces status"
        test_embedding = model.encode(test_query)
        test_results = client.search(
            collection_name=collection_name,
            query_vector=test_embedding.tolist(),
            limit=1
        )
        if test_results:
            print("\nTest query successful!")
            print(f"Test query: '{test_query}'")
            print(f"Top result: {test_results[0].payload.get('path')} (Score: {test_results[0].score:.4f})")
            print(f"Summary: {test_results[0].payload.get('summary', 'No summary')}")
        else:
            print("\nTest query returned no results. Please check your collection.")
    except Exception as e:
        print(f"ERROR verifying upload: {e}")
    
    print("\nProcess completed successfully!")
    print(f"Collection '{collection_name}' is now ready for querying.")
    print("\nNext steps:")
    print("1. Create a search interface (e.g., using Streamlit)")
    print("2. Implement MikroTik API integration")
    print("3. Build natural language response formatting")

if __name__ == "__main__":
    main()