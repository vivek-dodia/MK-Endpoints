# MikroTik API Finder

A natural language interface to find and query relevant MikroTik API endpoints, using vector search and LLM technologies.

## Overview

This tool allows network engineers to interact with MikroTik routers using natural language. It:

1. Translates queries into relevant MikroTik API endpoints using vector search
2. Executes the API calls and retrieves data
3. Presents the results in a human-readable format using LLM processing
4. Suggests related follow-up questions

## Setup Instructions

### Prerequisites

- Python 3.9+ (Python 3.11 recommended)
- A MikroTik router with API access for real data (optional)
- API keys for Qdrant and Google Gemini (or other LLM providers)

### Installation

Follow these steps to set up the application with the correct dependencies:

1. **Create a virtual environment**:

```bash
# Create a new virtual environment
python -m venv venv_mikrotik --prompt="mikrotik-api"

# Activate the virtual environment
# On Windows:
venv_mikrotik\Scripts\activate
# On macOS/Linux:
source venv_mikrotik/bin/activate
```

2. **Update base packages**:

```bash
pip install -U pip setuptools wheel
```

3. **Install dependencies**:

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

4. **Configure environment variables**:

Create a `.env` file in the project root with your API keys and configuration:

```
# LLM API Keys
GOOGLE_API_KEY=your_google_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Default model to use
DEFAULT_MODEL=gemini-2.0-pro-exp-02-05

# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=mikrotik_endpoints

# MikroTik Router settings (optional)
test_harlem_mikrotik=router_ip_address
test_harlem_mikrotik_user=router_username
test_harlem_mikrotik_pw=router_password
```

### Running the Application

Start the application with Streamlit:

```bash
streamlit run app.py
```

The web interface will be available at http://localhost:8501 by default.

## Usage

1. Enter your query in natural language (e.g., "Show me all interfaces and their IP addresses")
2. Click "Search" to execute the query
3. Review the results in the main panel
4. Explore related questions or technical details through the UI

## Troubleshooting

### Dependency Issues

If you encounter dependency conflicts, try the following:

1. Ensure you're using the exact package versions specified in requirements.txt
2. If an error mentions NumPy, try reinstalling with: `pip install --force-reinstall numpy==1.24.3`
3. For issues with sentence-transformers, try: `pip install --force-reinstall sentence-transformers==2.2.2`

### Connection Issues

If you have trouble connecting to your MikroTik router:

1. Verify that the router IP, username, and password are correct
2. Check that the API port (usually 8728) is open and accessible
3. Try enabling the "Use Mock Data" option to test without a router

## Architecture

The application consists of several components:

- **Streamlit UI**: Provides the web interface
- **Qdrant Vector Store**: Stores embeddings of API endpoint descriptions
- **Sentence Transformers**: Generates embeddings for natural language queries
- **RouterOS API**: Communicates with MikroTik routers
- **LLM Integration**: Uses Gemini or other LLMs for text processing

## Contributing

Contributions to improve the application are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.