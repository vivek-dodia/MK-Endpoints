import re
import streamlit as st
import routeros_api
import os

# MikroTik Router settings (import from environment)
MIKROTIK_IP = os.environ.get("test_harlem_mikrotik", "127.0.0.1")
MIKROTIK_USER = os.environ.get("test_harlem_mikrotik_user", "")
MIKROTIK_PW = os.environ.get("test_harlem_mikrotik_pw", "")

def is_id_endpoint(path):
    """Check if the endpoint contains an {id} parameter in the path"""
    # Look for path segments with {xyz} pattern
    return bool(re.search(r'/\{[^/]+\}', path))

def connect_to_mikrotik(ip, username, password, port=8728, ssl=False):
    """Connect to MikroTik router using RouterOS API"""
    try:
        connection = routeros_api.RouterOsApiPool(
            host=ip,
            username=username,
            password=password,
            port=port if not ssl else 8729,
            use_ssl=ssl,
            plaintext_login=True
        )
        api = connection.get_api()
        st.session_state.mikrotik_api = api
        st.session_state.mikrotik_connected = True
        return api
    
    except routeros_api.exceptions.RouterOsApiConnectionError as e:
        st.error(f"Connection failed: {str(e)}")
        st.session_state.mikrotik_connected = False
        return None
    
    except routeros_api.exceptions.RouterOsApiError as e:
        st.error(f"API error: {str(e)}")
        st.session_state.mikrotik_connected = False
        return None
    
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
        st.session_state.mikrotik_connected = False
        return None

def safe_disconnect():
    """Safely disconnect from MikroTik router"""
    if st.session_state.mikrotik_api:
        try:
            st.session_state.mikrotik_api.disconnect()
            st.session_state.mikrotik_api = None
            st.session_state.mikrotik_connected = False
        except:
            pass

def call_mikrotik_api(endpoint, parameters=None, router_ip=None, status_placeholder=None):
    """Call the MikroTik API with proper error handling and fallbacks"""
    # Update status if placeholder provided
    if status_placeholder:
        status_placeholder.write(f"Querying: {endpoint}")
    
    # Use the actual router if credentials are available
    if MIKROTIK_USER and MIKROTIK_PW and not st.session_state.get('use_mock_data', False):
        try:
            # Ensure we have a connection
            api = st.session_state.mikrotik_api
            if not api:
                if status_placeholder:
                    status_placeholder.write(f"Establishing connection to router...")
                api = connect_to_mikrotik(MIKROTIK_IP, MIKROTIK_USER, MIKROTIK_PW)
                if not api:
                    raise Exception("Could not establish connection")
            
            # Map endpoint path to actual RouterOS API path
            # This is needed because the OpenAPI spec may use different paths
            command = endpoint.rstrip('/')
                
            # Fix common command path issues
            command = command.replace('/print', '')  # Remove print suffix
            
            # Remove leading slash if present
            if command.startswith('/'):
                command = command[1:]
                
            # Get the appropriate resource
            try:
                if status_placeholder:
                    status_placeholder.write(f"Accessing resource: /{command}")
                resource = api.get_resource('/' + command)
                result = resource.get()
                if status_placeholder:
                    status_placeholder.write(f"✅ Data retrieved from: {endpoint}")
                return result
            except Exception as e:
                if status_placeholder:
                    status_placeholder.write(f"⚠️ Error accessing {command}: {str(e)}")
                
                # Try to handle special cases or path mappings
                if "no such command" in str(e).lower():
                    # Try common variants
                    alt_commands = []
                    
                    # Handle special cases
                    if 'interface/print' in endpoint:
                        alt_commands.append('interface')
                    elif 'ip/address/print' in endpoint:
                        alt_commands.append('ip/address')
                    elif 'system/resource/print' in endpoint:
                        alt_commands.append('system/resource')
                    elif 'ip/firewall/filter/print' in endpoint:
                        alt_commands.append('ip/firewall/filter')
                    
                    # Try alternatives
                    for alt in alt_commands:
                        try:
                            if status_placeholder:
                                status_placeholder.write(f"Trying alternative: /{alt}")
                            resource = api.get_resource('/' + alt)
                            result = resource.get()
                            if status_placeholder:
                                status_placeholder.write(f"✅ Data retrieved from alternative: /{alt}")
                            return result
                        except:
                            continue
                
                # If all attempts fail, raise the original error
                raise
        
        except Exception as e:
            if status_placeholder:
                status_placeholder.write(f"⚠️ Error with MikroTik API: {str(e)}")
                status_placeholder.write("Using mock data instead of live router data")
            else:
                st.warning(f"Error with MikroTik API: {str(e)}")
                st.warning("Using mock data instead of live router data")
    
    # If we reach here, use mock data
    if status_placeholder:
        status_placeholder.write(f"📊 Using mock data for: {endpoint}")
    
    # Return appropriate mock data based on endpoint
    return get_mock_data(endpoint)

def get_mock_data(endpoint):
    """Return mock data for various endpoints"""
    if "/interface" in endpoint and "print" in endpoint:
        return [
            {"name": "ether1", "type": "ether", "mtu": 1500, "actual-mtu": 1500, "mac-address": "00:0C:29:45:2A:3B", "running": True},
            {"name": "ether2", "type": "ether", "mtu": 1500, "actual-mtu": 1500, "mac-address": "00:0C:29:45:2A:3C", "running": False},
            {"name": "bridge1", "type": "bridge", "mtu": 1500, "actual-mtu": 1500, "mac-address": "00:0C:29:45:2A:3D", "running": True}
        ]
    elif "/ip/address" in endpoint and "print" in endpoint:
        return [
            {"address": "192.168.88.1/24", "network": "192.168.88.0", "interface": "ether1", "disabled": False},
            {"address": "10.0.0.1/24", "network": "10.0.0.0", "interface": "bridge1", "disabled": False}
        ]
    elif "/ip/dhcp-server/lease" in endpoint:
        return [
            {"mac-address": "00:0C:29:45:2A:3B", "address": "192.168.88.100", "host-name": "client1", "status": "bound"},
            {"mac-address": "00:0C:29:45:2A:3C", "address": "192.168.88.101", "host-name": "client2", "status": "bound"}
        ]
    elif "/ip/firewall/filter" in endpoint:
        return [
            {"chain": "input", "action": "accept", "protocol": "icmp", "comment": "Allow ICMP"},
            {"chain": "input", "action": "accept", "connection-state": "established,related", "comment": "Allow established connections"},
            {"chain": "input", "action": "drop", "comment": "Drop everything else"}
        ]
    elif "/system/resource" in endpoint:
        return {
            "uptime": "1d2h3m4s",
            "cpu-load": 15,
            "free-memory": 128000,
            "total-memory": 256000,
            "free-hdd-space": 100000,
            "total-hdd-space": 200000
        }
    elif "/interface/wireless/registration-table" in endpoint:
        return [
            {"mac-address": "00:0C:29:45:2A:3B", "interface": "wlan1", "signal-strength": "-65", "tx-rate": "65Mbps", "rx-rate": "65Mbps"},
            {"mac-address": "00:0C:29:45:2A:3C", "interface": "wlan1", "signal-strength": "-70", "tx-rate": "54Mbps", "rx-rate": "54Mbps"}
        ]
    # Default fallback
    return {"status": "success", "message": f"Mock data for {endpoint} (API integration failed)"}