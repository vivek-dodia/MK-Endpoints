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
    """Call the MikroTik API with proper error handling"""
    # Update status if placeholder provided
    if status_placeholder:
        status_placeholder.write(f"Querying: {endpoint}")
    
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
        else:
            st.error(f"Error with MikroTik API: {str(e)}")
        
        # Return an empty result or raise error based on severity
        if "permission denied" in str(e).lower():
            st.error("Permission denied. Please check your credentials.")
            return []
        elif "connection timed out" in str(e).lower():
            st.error("Connection timed out. Please check if the router is reachable.")
            return []
        else:
            # For other types of errors, re-raise to be handled by caller
            raise