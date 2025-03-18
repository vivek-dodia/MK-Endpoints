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
    1. The primary service or feature involved (DHCP, firewall, routing, interface, wireless, VPN, etc.)
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
            "interface": ["interface", "ether", "port", "link"],
            "vpn": ["vpn", "tunnel", "ipsec", "l2tp", "pptp", "sstp"],
            "dns": ["dns", "domain", "resolve", "name server"],
            "qos": ["qos", "traffic", "shaping", "queue", "bandwidth", "priority"],
            "bridge": ["bridge", "spanning", "switch", "vlan"]
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

def orchestrate_troubleshooting(issue_details, status_placeholder=None):
    """Run a comprehensive network troubleshooting check across multiple endpoint families"""
    if status_placeholder:
        status_placeholder.write("🔍 Starting comprehensive network troubleshooting...")
    
    # Common/core endpoints to always check for network status
    core_endpoints = [
        "/system/resource/print",  # System health
        "/interface/print",        # Interface status
        "/ip/address/print",       # IP addressing
        "/ip/route/print",         # Routing information
        "/system/logging/print"    # System logs
    ]
    
    # Service-specific endpoints based on detected issue
    service_specific_endpoints = {
        "dhcp": [
            "/ip/dhcp-server/print",
            "/ip/dhcp-server/network/print",
            "/ip/dhcp-server/lease/print"
        ],
        "firewall": [
            "/ip/firewall/filter/print",
            "/ip/firewall/nat/print",
            "/ip/firewall/mangle/print",
            "/ip/firewall/connection/print"
        ],
        "wireless": [
            "/interface/wireless/print",
            "/interface/wireless/registration-table/print",
            "/interface/wireless/access-list/print"
        ],
        "routing": [
            "/ip/route/print",
            "/routing/ospf/interface/print",
            "/routing/bgp/peer/print",
            "/ip/neighbor/print"
        ],
        "interface": [
            "/interface/ethernet/print",
            "/interface/bridge/print", 
            "/interface/bridge/port/print",
            "/interface/monitor-traffic"
        ],
        "dns": [
            "/ip/dns/print",
            "/ip/dns/static/print",
            "/ip/dns/cache/print"
        ],
        "vpn": [
            "/interface/ovpn-server/print",
            "/ip/ipsec/policy/print",
            "/interface/sstp-server/print"
        ],
        "bridge": [
            "/interface/bridge/print",
            "/interface/bridge/port/print",
            "/interface/bridge/vlan/print"
        ],
        "qos": [
            "/queue/simple/print",
            "/queue/tree/print"
        ]
    }
    
    # Determine which endpoints to check based on the issue
    service = issue_details.get("service", "unknown").lower()
    
    # Always include core endpoints
    endpoints_to_check = core_endpoints.copy()
    
    # Add service-specific endpoints
    if service in service_specific_endpoints:
        endpoints_to_check.extend(service_specific_endpoints[service])
        if status_placeholder:
            status_placeholder.write(f"Adding {service}-specific checks...")
    
    # Add related services based on symptoms
    symptom = issue_details.get("symptom", "").lower()
    
    # If connectivity issue, check firewall and routing
    if "connect" in symptom or "reach" in symptom or "access" in symptom:
        if "firewall" not in service:
            endpoints_to_check.extend(service_specific_endpoints["firewall"])
        if "routing" not in service:
            endpoints_to_check.extend(service_specific_endpoints["routing"])
    
    # If speed/performance issue, check QoS
    if "slow" in symptom or "speed" in symptom or "performance" in symptom:
        endpoints_to_check.extend(service_specific_endpoints["qos"])
    
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
    
    # Add specific logs based on service
    try:
        if status_placeholder:
            status_placeholder.write(f"Checking {service}-related logs...")
        
        topic_map = {
            "dhcp": "dhcp",
            "wireless": "wireless",
            "firewall": "firewall",
            "dns": "dns",
            "routing": "routing",
            "vpn": "pptp,l2tp,ipsec",
            "interface": "interface"
        }
        
        log_topic = topic_map.get(service, "info,error,warning")
        log_result = call_mikrotik_api("/log/print", parameters={"?topics": log_topic}, status_placeholder=status_placeholder)
        all_results[f"/log/print?topics={log_topic}"] = log_result
    except Exception as e:
        if status_placeholder:
            status_placeholder.write(f"⚠️ Error querying logs: {str(e)}")
        all_results[f"/log/print?topics={log_topic}"] = {"error": str(e)}
    
    if status_placeholder:
        status_placeholder.write("✅ Completed comprehensive data collection")
    
    return all_results

# Keep DHCP-specific functions for backward compatibility and specialized analysis
def orchestrate_dhcp_troubleshooting(issue_details, status_placeholder=None):
    """Run a comprehensive DHCP troubleshooting check"""
    # For backward compatibility, use the general function with DHCP specifics
    issue_details["service"] = "dhcp"
    return orchestrate_troubleshooting(issue_details, status_placeholder)

def analyze_network_troubleshooting(all_results, issue_details, llm):
    """Analyze results from multiple endpoints to identify network issues"""
    service = issue_details.get("service", "unknown").lower()
    
    # Prepare system resource information for context
    system_resources = all_results.get("/system/resource/print", [{}])
    if isinstance(system_resources, list) and len(system_resources) > 0:
        system_resource = system_resources[0]
    else:
        system_resource = {}
    
    router_context = f"""
    Router Information:
    - Uptime: {system_resource.get('uptime', 'Unknown')}
    - CPU Load: {system_resource.get('cpu-load', 'Unknown')}%
    - Free Memory: {system_resource.get('free-memory', 'Unknown')} / {system_resource.get('total-memory', 'Unknown')}
    - RouterOS Version: {system_resource.get('version', 'Unknown')}
    """
    
    # Format key results for the LLM to make the prompt more manageable
    key_endpoints = []
    
    # System information is always important
    if "/system/resource/print" in all_results:
        key_endpoints.append("/system/resource/print")
    
    # Get service-specific key endpoints
    service_key_endpoint_map = {
        "dhcp": ["/ip/dhcp-server/print", "/ip/dhcp-server/network/print", "/ip/dhcp-server/lease/print"],
        "firewall": ["/ip/firewall/filter/print", "/ip/firewall/nat/print"],
        "wireless": ["/interface/wireless/print", "/interface/wireless/registration-table/print"],
        "routing": ["/ip/route/print"],
        "interface": ["/interface/print", "/ip/address/print"],
        "dns": ["/ip/dns/print"],
        "bridge": ["/interface/bridge/print", "/interface/bridge/port/print"],
        "vpn": ["/interface/ovpn-server/print", "/ip/ipsec/policy/print"]
    }
    
    # Add relevant key endpoints based on service
    if service in service_key_endpoint_map:
        for endpoint in service_key_endpoint_map[service]:
            if endpoint in all_results and endpoint not in key_endpoints:
                key_endpoints.append(endpoint)
    
    # Always include interfaces and IP addresses
    for important_endpoint in ["/interface/print", "/ip/address/print", f"/log/print?topics={service}"]:
        if important_endpoint in all_results and important_endpoint not in key_endpoints:
            key_endpoints.append(important_endpoint)
    
    # Format the key results
    formatted_key_results = "\n\n".join([
        f"Endpoint: {endpoint}\nData: {json.dumps(all_results[endpoint], indent=2)}"
        for endpoint in key_endpoints if endpoint in all_results
    ])
    
    # Create a list of all available endpoints to let the LLM know what data is available
    all_available_endpoints = "\n".join([f"- {endpoint}" for endpoint in all_results.keys()])
    
    # Create service-specific analysis prompts
    service_specific_analysis = {
        "dhcp": """
        For DHCP issues:
        1. Check if the DHCP server is properly configured (IP ranges, networks, interfaces)
        2. Verify the DHCP server is running on the correct interface
        3. Check if there are firewall rules that might block DHCP traffic (UDP 67/68)
        4. Examine if the interface where client connects has an IP address
        5. Look for conflicting IP addresses or overlapping networks
        6. Check if there are any DHCP lease requests in the logs
        """,
        "firewall": """
        For firewall issues:
        1. Check for overly restrictive firewall rules that might block legitimate traffic
        2. Look for missing allow rules for established/related connections
        3. Verify NAT rules are correctly configured for internet access
        4. Check if the connection tracking table is full
        5. Look for any recent changes to firewall rules in logs
        """,
        "interface": """
        For interface issues:
        1. Check if the interface is enabled and running
        2. Verify physical connectivity (link status)
        3. Check if the interface has the correct IP address assigned
        4. Look for duplex/speed mismatches
        5. Check interface error counters for signs of physical problems
        """,
        "wireless": """
        For wireless issues:
        1. Check if the wireless interface is enabled and running
        2. Verify wireless configuration (SSID, security, channel)
        3. Check signal strength for connected clients
        4. Look for interference sources or channel congestion
        5. Check if access lists might be blocking clients
        """,
        "routing": """
        For routing issues:
        1. Verify default gateway configuration
        2. Check if there are multiple routes to the same destination
        3. Look for missing routes to needed networks
        4. Check if dynamic routing protocols are properly configured
        5. Verify that routing protocols have established neighborships
        """,
        "dns": """
        For DNS issues:
        1. Check if DNS servers are configured and reachable
        2. Verify that the router can resolve common domain names
        3. Look for DNS cache issues
        4. Check if clients are using the router as their DNS server
        5. Verify that ISP DNS servers are responsive
        """,
        "vpn": """
        For VPN issues:
        1. Check if the VPN server is running
        2. Verify authentication settings
        3. Check if required ports are open in the firewall
        4. Look for incorrect encryption/authentication settings
        5. Verify that IP pools are correctly configured
        """,
        "bridge": """
        For bridge issues:
        1. Check if the bridge interface is enabled and running
        2. Verify that ports are added to the correct bridge
        3. Check for STP (Spanning Tree Protocol) issues
        4. Look for VLAN configuration problems
        5. Verify that the bridge has an IP address if needed
        """
    }
    
    # Get the appropriate analysis guidelines
    analysis_guidelines = service_specific_analysis.get(service, """
    For general network issues:
    1. Check overall system health (CPU, memory, disk usage)
    2. Verify interface status and connectivity
    3. Check IP addressing and subnet configuration
    4. Look for routing problems or missing routes
    5. Check firewall rules that might impact connectivity
    6. Review recent logs for errors or warnings
    """)
    
    # Prepare the main prompt
    prompt = f"""
    You are a MikroTik network troubleshooting expert. A user reports:
    "{issue_details['original_query']}"
    
    {router_context}
    
    Based on this issue description and the following router data, identify the most likely
    causes of the network problem, from most to least probable.
    
    Key Router Data:
    {formatted_key_results}
    
    All available endpoints for reference:
    {all_available_endpoints}
    
    {analysis_guidelines}
    
    For each potential issue, provide:
    - The specific problematic configuration or setting
    - How to verify if this is causing the problem
    - The exact steps to fix the issue
    
    Structure your answer as a prioritized list of possible causes with specific evidence.
    Include applicable RouterOS commands to implement the fixes.
    """
    
    # Use reasoning mode for complex analysis
    analysis = llm.generate_with_reasoning(
        prompt=prompt,
        temperature=0.7,
        stream=False
    )
    
    return analysis

# For backward compatibility, maintain the DHCP-specific function
def analyze_dhcp_troubleshooting(all_results, issue_details, llm):
    """Analyze results from multiple endpoints to identify DHCP issues"""
    issue_details["service"] = "dhcp"
    return analyze_network_troubleshooting(all_results, issue_details, llm)

def render_network_troubleshooting_diagram(all_results, issue_details):
    """Create a visual representation of the network with problem areas highlighted"""
    service = issue_details.get("service", "unknown").lower()
    
    # Use specialized diagrams for specific services
    if service == "dhcp":
        return render_dhcp_troubleshooting_diagram(all_results)
    
    # For other services, create a general network diagram
    # Analyze results to identify problem areas
    problem_areas = []
    
    # Check system resources
    system_resources = all_results.get("/system/resource/print", [{}])
    system_resource = system_resources[0] if isinstance(system_resources, list) and len(system_resources) > 0 else {}
    
    # Safe conversion of numeric values
    try:
        cpu_load = float(system_resource.get('cpu-load', 0))
    except (ValueError, TypeError):
        cpu_load = 0
        
    try:
        free_memory = float(system_resource.get('free-memory', 0))
        total_memory = float(system_resource.get('total-memory', 100000))
    except (ValueError, TypeError):
        free_memory = 0
        total_memory = 100000
    
    if cpu_load > 80:
        problem_areas.append("system-cpu")
        
    if free_memory < (total_memory * 0.2):
        problem_areas.append("system-memory")
    
    # Check interfaces
    interfaces = all_results.get('/interface/print', [])
    interface_status = {}
    
    for interface in interfaces:
        name = interface.get('name', '')
        if name:
            interface_status[name] = {
                'running': interface.get('running', False),
                'disabled': interface.get('disabled', False)
            }
            if not interface.get('running', False) and not interface.get('disabled', True):
                problem_areas.append("interface-down")
    
    # Check IP addresses
    ip_addresses = all_results.get('/ip/address/print', [])
    interface_ips = {}
    
    for ip in ip_addresses:
        interface_name = ip.get('interface', '')
        if interface_name:
            interface_ips[interface_name] = ip.get('address', '')
    
    for name, status in interface_status.items():
        if status['running'] and name not in interface_ips and not name.startswith('bridge'):
            problem_areas.append("interface-no-ip")
    
    # Check routes
    routes = all_results.get('/ip/route/print', [])
    has_default_route = False
    
    for route in routes:
        if route.get('dst-address', '') == '0.0.0.0/0':
            has_default_route = True
            if not route.get('active', False):
                problem_areas.append("routing-default-inactive")
    
    if not has_default_route:
        problem_areas.append("routing-no-default")
    
    # Check firewall rules if present
    firewall_rules = all_results.get('/ip/firewall/filter/print', [])
    has_allow_established = False
    
    for rule in firewall_rules:
        if (rule.get('chain', '') == 'input' and 
            rule.get('action', '') == 'accept' and 
            'established' in str(rule.get('connection-state', ''))):
            has_allow_established = True
    
    if not has_allow_established and firewall_rules:
        problem_areas.append("firewall-no-established")
    
    # Generate HTML for the general network diagram
    html = """
    <div class="network-diagram">
        <div class="diagram-title">Network Status Overview</div>
        <div class="diagram-grid">
    """
    
    # System component
    if "system-cpu" in problem_areas or "system-memory" in problem_areas:
        system_issues = []
        if "system-cpu" in problem_areas:
            system_issues.append("High CPU usage")
        if "system-memory" in problem_areas:
            system_issues.append("Low memory")
        
        html += f"""
        <div class="component system problem">
            <div class="component-icon">🖥️</div>
            <div class="component-title">System Resources</div>
            <div class="component-status">⚠️ {', '.join(system_issues)}</div>
        </div>
        """
    else:
        html += """
        <div class="component system">
            <div class="component-icon">🖥️</div>
            <div class="component-title">System Resources</div>
            <div class="component-status">✅ Normal</div>
        </div>
        """
    
    # Interface component
    if "interface-down" in problem_areas or "interface-no-ip" in problem_areas:
        interface_issues = []
        if "interface-down" in problem_areas:
            interface_issues.append("Interface(s) down")
        if "interface-no-ip" in problem_areas:
            interface_issues.append("Missing IP address(es)")
        
        html += f"""
        <div class="component interface problem">
            <div class="component-icon">🔌</div>
            <div class="component-title">Interfaces</div>
            <div class="component-status">⚠️ {', '.join(interface_issues)}</div>
        </div>
        """
    else:
        html += """
        <div class="component interface">
            <div class="component-icon">🔌</div>
            <div class="component-title">Interfaces</div>
            <div class="component-status">✅ Normal</div>
        </div>
        """
    
    # Routing component
    if "routing-no-default" in problem_areas or "routing-default-inactive" in problem_areas:
        routing_issues = []
        if "routing-no-default" in problem_areas:
            routing_issues.append("No default route")
        if "routing-default-inactive" in problem_areas:
            routing_issues.append("Default route inactive")
        
        html += f"""
        <div class="component routing problem">
            <div class="component-icon">🛣️</div>
            <div class="component-title">Routing</div>
            <div class="component-status">⚠️ {', '.join(routing_issues)}</div>
        </div>
        """
    else:
        html += """
        <div class="component routing">
            <div class="component-icon">🛣️</div>
            <div class="component-title">Routing</div>
            <div class="component-status">✅ Normal</div>
        </div>
        """
    
    # Firewall component
    if "firewall-no-established" in problem_areas:
        html += """
        <div class="component firewall problem">
            <div class="component-icon">🔒</div>
            <div class="component-title">Firewall</div>
            <div class="component-status">⚠️ Missing established rule</div>
        </div>
        """
    else:
        html += """
        <div class="component firewall">
            <div class="component-icon">🔒</div>
            <div class="component-title">Firewall</div>
            <div class="component-status">✅ Normal</div>
        </div>
        """
    
    # Add service-specific component
    service_icons = {
        "dhcp": "🏷️",
        "dns": "📋",
        "wireless": "📶",
        "vpn": "🔑",
        "bridge": "🌉"
    }
    
    service_icon = service_icons.get(service, "⚙️")
    service_name = service.upper() if service else "Unknown"
    
    # Check for service-specific issues
    service_specific_issues = []
    
    if service == "dhcp":
        dhcp_servers = all_results.get('/ip/dhcp-server/print', [])
        dhcp_networks = all_results.get('/ip/dhcp-server/network/print', [])
        
        if not dhcp_servers:
            service_specific_issues.append("No DHCP servers configured")
        elif all(server.get('disabled', False) for server in dhcp_servers):
            service_specific_issues.append("All DHCP servers disabled")
        
        if not dhcp_networks:
            service_specific_issues.append("No DHCP networks configured")
    
    elif service == "dns":
        dns_settings = all_results.get('/ip/dns/print', [{}])
        dns_setting = dns_settings[0] if isinstance(dns_settings, list) and len(dns_settings) > 0 else {}
        
        if not dns_setting.get('servers', []):
            service_specific_issues.append("No DNS servers configured")
        if not dns_setting.get('allow-remote-requests', False):
            service_specific_issues.append("Remote DNS requests disabled")
    
    elif service == "wireless":
        wireless_interfaces = all_results.get('/interface/wireless/print', [])
        
        if not wireless_interfaces:
            service_specific_issues.append("No wireless interfaces")
        elif all(iface.get('disabled', False) for iface in wireless_interfaces):
            service_specific_issues.append("All wireless interfaces disabled")
    
    # Add the service component to the diagram
    if service_specific_issues:
        html += f"""
        <div class="component service problem">
            <div class="component-icon">{service_icon}</div>
            <div class="component-title">{service_name} Service</div>
            <div class="component-status">⚠️ {', '.join(service_specific_issues)}</div>
        </div>
        """
    else:
        html += f"""
        <div class="component service">
            <div class="component-icon">{service_icon}</div>
            <div class="component-title">{service_name} Service</div>
            <div class="component-status">Status unknown</div>
        </div>
        """
    
    # Close grid and container
    html += """
        </div>
    </div>
    """
    
    # Add styling
    html += """
    <style>
    .network-diagram {
        font-family: Arial, sans-serif;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .diagram-title {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .diagram-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }
    .component {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: all 0.3s ease;
    }
    .component.problem {
        background: #ffdddd;
        border: 1px solid #ff5555;
    }
    .component-icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    .component-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .component-status {
        font-size: 12px;
        text-align: center;
    }
    </style>
    """
    
    return html

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
            dst_port = str(rule.get('dst-port', ''))
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

def get_network_recommendations(all_results, issue_details, llm):
    """Generate network configuration recommendations based on the service type"""
    service = issue_details.get("service", "unknown").lower()
    
    # If it's DHCP, use the specialized function
    if service == "dhcp":
        return get_dhcp_recommendations(all_results, issue_details, llm)
    
    # Extract key information based on service type
    key_data = {}
    
    # Always include system resource data
    system_resources = all_results.get('/system/resource/print', [{}])
    key_data["system_resources"] = system_resources[0] if isinstance(system_resources, list) and len(system_resources) > 0 else {}
    
    # Always include interfaces and IP addresses
    key_data["interfaces"] = all_results.get('/interface/print', [])
    key_data["ip_addresses"] = all_results.get('/ip/address/print', [])
    
    # Add service-specific data
    if service == "firewall":
        key_data["firewall_filter"] = all_results.get('/ip/firewall/filter/print', [])
        key_data["firewall_nat"] = all_results.get('/ip/firewall/nat/print', [])
    elif service == "routing":
        key_data["routes"] = all_results.get('/ip/route/print', [])
        key_data["ospf_interfaces"] = all_results.get('/routing/ospf/interface/print', [])
        key_data["bgp_peers"] = all_results.get('/routing/bgp/peer/print', [])
    elif service == "wireless":
        key_data["wireless_interfaces"] = all_results.get('/interface/wireless/print', [])
        key_data["wireless_registrations"] = all_results.get('/interface/wireless/registration-table/print', [])
    elif service == "dns":
        key_data["dns_settings"] = all_results.get('/ip/dns/print', [])
        key_data["dns_cache"] = all_results.get('/ip/dns/cache/print', [])
    elif service == "vpn":
        key_data["ovpn_servers"] = all_results.get('/interface/ovpn-server/print', [])
        key_data["ipsec_policies"] = all_results.get('/ip/ipsec/policy/print', [])
    elif service == "bridge":
        key_data["bridges"] = all_results.get('/interface/bridge/print', [])
        key_data["bridge_ports"] = all_results.get('/interface/bridge/port/print', [])
    
    # Prepare service-specific prompts
    service_specific_prompts = {
        "firewall": """
        For firewall configuration, focus on:
        1. Ensuring established/related traffic is allowed
        2. Properly configuring NAT rules for internet access
        3. Creating rules that balance security and accessibility
        4. Addressing any overly restrictive rules
        """,
        "routing": """
        For routing configuration, focus on:
        1. Ensuring a proper default gateway is configured
        2. Configuring static routes for specific networks if needed
        3. Setting up dynamic routing protocols correctly
        4. Addressing routing loops or conflicting routes
        """,
        "wireless": """
        For wireless configuration, focus on:
        1. Optimizing channel and frequency settings
        2. Security settings (WPA2/WPA3, authentication methods)
        3. Signal strength and coverage issues
        4. Client isolation or access control if needed
        """,
        "dns": """
        For DNS configuration, focus on:
        1. Configuring appropriate DNS servers
        2. Setting up DNS caching for performance
        3. Configuring static DNS entries if needed
        4. Enabling remote requests if serving DNS to clients
        """,
        "vpn": """
        For VPN configuration, focus on:
        1. Server settings (authentication, encryption)
        2. Client access and IP pool configuration
        3. Firewall rules to allow VPN traffic
        4. Routing configuration for VPN clients
        """,
        "bridge": """
        For bridge configuration, focus on:
        1. Correctly adding interfaces to bridges
        2. VLAN configuration if applicable
        3. STP settings to prevent loops
        4. IP addressing for bridge interfaces
        """
    }
    
    # Get the appropriate prompt guidelines
    prompt_guidelines = service_specific_prompts.get(service, """
    For general network configuration, focus on:
    1. Interface settings and connectivity
    2. IP addressing and subnet design
    3. Routing configuration
    4. Basic firewall security
    """)
    
    # Prepare data for the LLM
    prompt = f"""
    You are a MikroTik network expert. Generate specific configuration recommendations 
    for solving {service} issues based on this data.
    
    User issue: "{issue_details['original_query']}"
    
    Current Configuration Data:
    {json.dumps(key_data, indent=2)}
    
    {prompt_guidelines}
    
    Provide 3-5 specific MikroTik terminal commands that would fix common {service} configuration issues.
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
        "can't", "unable", "no connection", "slow", "intermittent"
    ]
    
    return any(keyword in query.lower() for keyword in troubleshooting_keywords)