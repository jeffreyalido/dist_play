import socket
import sys

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually need to send data
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    finally:
        s.close()
    return IP

def save_ip(ip_address, file_path):
    with open(file_path, 'w') as file:
        file.write(ip_address)

if __name__ == "__main__":
    ip_address = get_ip()
    file_path = sys.argv[1]  # Pass file path as a command-line argument
    save_ip(ip_address, file_path)
