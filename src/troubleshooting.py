import streamlit as st
import json
import re
from src.api_utils import call_mikrotik_api

def identify_troubleshooting_intent(query, llm):
    """Determine if query is a troubleshooting request and extract key elements"""
    # Use LLM to classify and extract components
    prompt = f"""
    Analyze this network troubleshooting query: "{query}"
    
    Identify:
    1. The primary service or feature involved (DHCP, firewall, routing, etc.)
    2. The symptom or issue observed (not working, slow, disconnected, etc.)
    3. Any specific devices or interfaces mentioned
    4. Recent actions that might relate to the issue
    
    Return your analysis as a structured JSON object with these fields:
    {{
        "service": "primary service name",
        "symptom": "main issue",
        "devices": ["device1", "device2"],
        "interfaces": ["interface1", "interface2"],
        "recent_actions": ["action1", "action2"],
        "is_troubleshooting": true or false,
        "original_query": "the original query"
    }}
    """
    
    try:
        result = llm.generate_json(prompt=prompt, temperature=0.3)
        # Add the original query to the result
        if isinstance(result, dict):
            result["original_query"] = query
            
            # If is_troubleshooting is not in the result, try to infer it
            if "is_troubleshooting" not in result:
                result["is_troubleshooting"] = (
                    "not working" in query.lower() or 
                    "problem" in query.lower() or 
                    "issue" in query.lower() or 
                    "doesn't" in query.lower() or 
                    "isn't" in query.lower() or 
                    "fail" in query.lower() or
                    "trouble" in query.lower()
                )
        return result
    except Exception as e:
        st.warning(f"Error identifying troubleshooting intent: {e}")
        # Fallback with basic detection
        is_troubleshooting = (
            "not working" in query.lower() or 
            "problem" in query.lower() or 
            "issue" in query.lower() or 
            "doesn't" in query.lower() or 
            "isn't" in query.lower() or 
            "fail" in query.lower() or
            "trouble" in query.lower()
        )
        
        # Simple keyword extraction for service type
        service_types = {
            "dhcp": ["dhcp", "ip address", "lease", "dynamic ip"],
            "firewall": ["firewall", "filter", "block", "allow", "nat"],
            "wireless": ["wifi", "wireless", "wlan", "signal", "connect"],
            "routing": ["route", "routing", "gateway", "path", "next hop"],
            "interface": ["interface", "ether", "port", "link"]
        }
        
        detected_service = None
        for service, keywords in service_types.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_service = service
                break
        
        return {
            "service": detected_service or "unknown",
            "symptom": "not working",
            "devices": [],
            "interfaces": [],
            "recent_actions": [],
            "is_troubleshooting": is_troubleshooting,
            "original_query": query
        }

def orchestrate_dhcp_troubleshooting(issue_details, status_placeholder=None):
    """Run a comprehensive DHCP troubleshooting check"""
    if status_placeholder:
        status_placeholder.write("🔍 Starting DHCP troubleshooting...")
    
    endpoints_to_check = [
        "/ip/dhcp-server/print",
        "/ip/dhcp-server/network/print",
        "/ip/dhcp-server/lease/print",
        "/ip/firewall/filter/print",
        "/interface/print",
        "/ip/address/print",
        "/system/logging/print"
    ]
    
    # Container for results
    all_results = {}
    
    # Execute all queries
    for endpoint in endpoints_to_check:
        if status_placeholder:
            status_placeholder.write(f"Checking {endpoint}...")
        
        try:
            result = call_mikrotik_api(endpoint, status_placeholder=status_placeholder)
            all_results[endpoint] = result
        except Exception as e:
            if status_placeholder:
                status_placeholder.write(f"⚠️ Error querying {endpoint}: {str(e)}")
            all_results[endpoint] = {"error": str(e)}
    
    # Add a special check for DHCP logs
    try:
        if status_placeholder:
            status_placeholder.write("Checking DHCP-related logs...")
        
        # Get DHCP logs specifically
        log_result = call_mikrotik_api("/log/print", parameters={"?topics": "dhcp"}, status_placeholder=status_placeholder)
        all_results["/log/print?topics=dhcp"] = log_result
    except Exception as e:
        if status_placeholder:
            status_placeholder.write(f"⚠️ Error querying DHCP logs: {str(e)}")
        all_results["/log/print?topics=dhcp"] = {"error": str(e)}
    
    if status_placeholder:
        status_placeholder.write("✅ Completed data collection for DHCP troubleshooting")
    
    return all_results

def analyze_dhcp_troubleshooting(all_results, issue_details, llm):
    """Analyze results from multiple endpoints to identify DHCP issues"""
    # Format all results for the LLM
    formatted_results = "\n\n".join([
        f"Endpoint: {endpoint}\nData: {json.dumps(data, indent=2)}"
        for endpoint, data in all_results.items()
    ])
    
    prompt = f"""
    You are a MikroTik network troubleshooting expert. A user reports:
    "{issue_details['original_query']}"
    
    Based on this issue description and the following router data, identify the most likely
    causes of the DHCP problem, from most to least probable.
    
    Router Data:
    {formatted_results}
    
    For your analysis:
    1. First check if the DHCP server is properly configured (IP ranges, networks, interfaces).
    2. Verify the DHCP server is running on the correct interface.
    3. Check if there are firewall rules that might block DHCP traffic (UDP 67/68).
    4. Examine if the interface where client connects has an IP address.
    5. Look for conflicting IP addresses or overlapping networks.
    6. Check if there are any DHCP lease requests in the logs.
    7. Verify if the network has the correct gateway and DNS settings.
    
    For each potential issue, provide:
    - The specific problematic configuration or setting
    - How to verify if this is causing the problem
    - The exact steps to fix the issue
    
    Structure your answer as a prioritized list of possible causes with specific evidence.
    """
    
    # Use the reasoning mode for complex analysis
    analysis = llm.generate_with_reasoning(
        prompt=prompt,
        temperature=0.7,
        stream=False
    )
    
    return analysis

def render_dhcp_troubleshooting_diagram(all_results):
    """Create a visual representation of the DHCP process with problem areas highlighted"""
    # Analyze results to identify problem areas
    problem_areas = []
    
    # Check if DHCP server exists and is enabled
    dhcp_servers = all_results.get('/ip/dhcp-server/print', [])
    if not dhcp_servers:
        problem_areas.append("dhcp-server")
    else:
        for server in dhcp_servers:
            if server.get('disabled', False):
                problem_areas.append("dhcp-server")
                break
    
    # Check if interface has IP address
    interfaces = all_results.get('/interface/print', [])
    ip_addresses = all_results.get('/ip/address/print', [])
    
    # Map interfaces to their IP addresses
    interface_ips = {}
    for ip in ip_addresses:
        interface_name = ip.get('interface', '')
        if interface_name:
            interface_ips[interface_name] = ip.get('address', '')
    
    # Check if any active interfaces are missing IPs
    for interface in interfaces:
        if interface.get('running', False) and interface.get('name') not in interface_ips:
            problem_areas.append("interface")
            break
    
    # Check DHCP network configuration
    dhcp_networks = all_results.get('/ip/dhcp-server/network/print', [])
    if not dhcp_networks:
        problem_areas.append("dhcp-network")
    
    # Check for firewall rules that might block DHCP
    firewall_rules = all_results.get('/ip/firewall/filter/print', [])
    dhcp_blocked = False
    for rule in firewall_rules:
        if (rule.get('chain') == 'input' and 
            rule.get('action') == 'drop' and 
            (rule.get('protocol') == 'udp' or rule.get('protocol') is None)):
            # Check if rule might block DHCP ports
            dst_port = rule.get('dst-port', '')
            if dst_port and ('67' in dst_port or '68' in dst_port):
                dhcp_blocked = True
                problem_areas.append("firewall")
                break
    
    # Generate HTML for the diagram
    html = """
    <div class="dhcp-flow">
        <div class="component client">Client Device</div>
        <div class="arrow">DHCP Discover →</div>
    """
    
    # Add interface component (potentially problematic)
    if "interface" in problem_areas:
        html += '<div class="component interface problem">Router Interface<br><small>⚠️ Missing IP address</small></div>'
    else:
        html += '<div class="component interface">Router Interface</div>'
    
    # Add firewall component (potentially problematic)
    if "firewall" in problem_areas:
        html += '<div class="arrow">→</div><div class="component firewall problem">Firewall Rules<br><small>⚠️ May block DHCP</small></div>'
    else:
        html += '<div class="arrow">→</div><div class="component firewall">Firewall Rules</div>'
    
    # Add DHCP server component (potentially problematic)
    if "dhcp-server" in problem_areas:
        html += '<div class="arrow">→</div><div class="component dhcp-server problem">DHCP Server<br><small>⚠️ Not configured/disabled</small></div>'
    elif "dhcp-network" in problem_areas:
        html += '<div class="arrow">→</div><div class="component dhcp-server problem">DHCP Server<br><small>⚠️ Network pool issues</small></div>'
    else:
        html += '<div class="arrow">→</div><div class="component dhcp-server">DHCP Server</div>'
    
    # Complete the flow
    html += '<div class="arrow">← DHCP Offer</div>'
    
    # Close the container and add styling
    html += """
    </div>
    <style>
    .dhcp-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 20px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    .component {
        padding: 15px;
        border-radius: 8px;
        background: #f0f2f6;
        text-align: center;
        min-width: 120px;
    }
    .problem {
        background: #ffdddd;
        border: 1px solid #ff5555;
    }
    .arrow {
        margin: 0 5px;
        color: #666;
        font-weight: bold;
    }
    </style>
    """
    
    return html

def get_dhcp_recommendations(all_results, issue_details, llm):
    """Generate DHCP configuration recommendations"""
    # Extract key information for generating recommendations
    dhcp_servers = all_results.get('/ip/dhcp-server/print', [])
    dhcp_networks = all_results.get('/ip/dhcp-server/network/print', [])
    interfaces = all_results.get('/interface/print', [])
    ip_addresses = all_results.get('/ip/address/print', [])
    leases = all_results.get('/ip/dhcp-server/lease/print', [])
    
    # Prepare data for the LLM
    prompt = f"""
    You are a MikroTik network expert. Generate specific configuration recommendations 
    for solving DHCP issues based on this data.
    
    User issue: "{issue_details['original_query']}"
    
    DHCP Servers: {json.dumps(dhcp_servers, indent=2)}
    DHCP Networks: {json.dumps(dhcp_networks, indent=2)}
    Interfaces: {json.dumps(interfaces, indent=2)}
    IP Addresses: {json.dumps(ip_addresses, indent=2)}
    Current Leases: {json.dumps(leases, indent=2)}
    
    Provide 3-5 specific MikroTik terminal commands that would fix common DHCP configuration issues.
    For each command:
    1. Explain what the command does
    2. Show the exact command syntax to enter in RouterOS terminal
    3. Explain any parameters that need to be adjusted
    
    Format your response with clear headings and code blocks for commands.
    """
    
    # Generate recommendations
    recommendations = llm.get_completion(
        prompt=prompt,
        temperature=0.5,
        stream=False
    )
    
    return recommendations

def is_troubleshooting_query(query):
    """Quick check if a query is likely a troubleshooting request"""
    troubleshooting_keywords = [
        "not working", "problem", "issue", "doesn't", "isn't", 
        "fail", "trouble", "broken", "error", "help", "wrong",
        "can't", "unable", "no connection"
    ]
    
    return any(keyword in query.lower() for keyword in troubleshooting_keywords)