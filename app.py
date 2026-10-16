#!/usr/bin/env python3
"""Zero-Dependency Local HTTP & API Server for CarePing OS v3.0 Quantum."""
import http.server
import socketserver
import webbrowser
import os

PORT = 8050
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class CarePingHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.path = '/forgotten_patient_dashboard.html'
        return super().do_GET()

if __name__ == '__main__':
    url = f'http://localhost:{PORT}'
    print('==============================================================================')
    print(f'⚡ CarePing OS v3.0 QUANTUM Local Server Online at: {url}')
    print('==============================================================================')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    with socketserver.TCPServer(('', PORT), CarePingHandler) as httpd:
        httpd.serve_forever()
