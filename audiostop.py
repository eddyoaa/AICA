from pythonosc.udp_client import SimpleUDPClient

# SuperCollider OSC server details
SC_IP = "127.0.0.1"  # IP address of the SuperCollider server
SC_PORT = 57123       # Port number to send OSC messages

# Create an OSC client
client = SimpleUDPClient(SC_IP, SC_PORT)

# Execute the stop and reset function
if __name__ == "__main__":
    stop_and_reset()
