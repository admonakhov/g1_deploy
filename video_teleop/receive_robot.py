"""Print incoming GMR-G1 UDP JSON packets; does not control robot hardware."""

import argparse
import json
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9870)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind((args.host, args.port))
        print(f"Listening on {args.host}:{args.port}", flush=True)
        try:
            while True:
                payload, _address = receiver.recvfrom(65535)
                try:
                    packet = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(packet, dict) or packet.get("schema") != "gmr-g1-v1":
                    continue
                print(json.dumps(packet, ensure_ascii=False), flush=True)
                if packet.get("status") == "stopped":
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
