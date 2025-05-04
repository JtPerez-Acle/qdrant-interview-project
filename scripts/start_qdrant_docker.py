#!/usr/bin/env python
"""Script to start Qdrant in Docker for Contexto-Crusher."""

import argparse
import os
import subprocess
import sys
import time
import requests


def check_docker_installed():
    """Check if Docker is installed and running."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_docker_permissions():
    """Check if the user has permissions to use Docker."""
    try:
        # Try to run a simple Docker command that requires daemon access
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_qdrant_running(port=6333):
    """Check if Qdrant is already running on the specified port."""
    try:
        # Try the root endpoint for health check
        response = requests.get(f"http://localhost:{port}/")
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass
    return False


def start_qdrant_container(port=6333, volume_path=None, container_name="qdrant-contexto"):
    """Start Qdrant in Docker.

    Args:
        port: Port to expose Qdrant on
        volume_path: Path to store Qdrant data (for persistence)
        container_name: Name for the Docker container

    Returns:
        True if successful, False otherwise
    """
    # Check if container already exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if container_name in result.stdout:
        print(f"Container '{container_name}' already exists.")

        # Check if it's running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )

        if container_name in result.stdout:
            print(f"Container '{container_name}' is already running.")
            return True
        else:
            print(f"Starting existing container '{container_name}'...")
            subprocess.run(["docker", "start", container_name], check=True)
            return True

    # Build the docker run command
    cmd = ["docker", "run", "-d", "--name", container_name, "-p", f"{port}:{port}"]

    # Add volume if specified
    if volume_path:
        # Create the directory if it doesn't exist
        os.makedirs(volume_path, exist_ok=True)

        # Add volume mapping
        cmd.extend(["-v", f"{volume_path}:/qdrant/storage"])

    # Add the image name
    cmd.append("qdrant/qdrant")

    # Run the command
    try:
        print(f"Starting Qdrant in Docker with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Wait for Qdrant to be ready
        print("Waiting for Qdrant to be ready...")
        max_retries = 30  # Increased from 10 to 30
        retry_delay = 5   # Increased from 2 to 5 seconds

        for i in range(max_retries):
            if check_qdrant_running(port):
                print(f"Qdrant is now running on http://localhost:{port}")
                return True

            # Show container logs if we're having trouble
            if i > 0 and i % 5 == 0:
                print("\nChecking container logs:")
                try:
                    logs = subprocess.run(
                        ["docker", "logs", container_name, "--tail", "10"],
                        capture_output=True, text=True, check=False
                    )
                    if logs.stdout:
                        print("Container logs:")
                        print(logs.stdout)
                except Exception as e:
                    print(f"Error getting container logs: {e}")

            print(f"Waiting for Qdrant to start (attempt {i+1}/{max_retries}, {retry_delay}s delay)...")
            time.sleep(retry_delay)

        print("\nTimed out waiting for Qdrant to start.")
        print("Possible issues:")
        print("1. Docker might not have enough resources (memory/CPU)")
        print("2. Network port 6333 might be blocked or in use")
        print("3. The container might be failing to initialize properly")

        # Show final container logs
        try:
            logs = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True, text=True, check=False
            )
            if logs.stdout:
                print("\nFull container logs:")
                print(logs.stdout)
        except Exception as e:
            print(f"Error getting container logs: {e}")

        return False

    except subprocess.CalledProcessError as e:
        print(f"Error starting Qdrant container: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start Qdrant in Docker for Contexto-Crusher.")
    parser.add_argument("--port", type=int, default=6333, help="Port to expose Qdrant on (default: 6333)")
    parser.add_argument("--volume", type=str, help="Path to store Qdrant data (for persistence)")
    parser.add_argument("--name", type=str, default="qdrant-contexto", help="Name for the Docker container")

    args = parser.parse_args()

    # Check if Docker is installed
    if not check_docker_installed():
        print("❌ Docker is not installed. Please install Docker and try again.")
        print("   Visit https://docs.docker.com/get-docker/ for installation instructions.")
        sys.exit(1)

    # Check if user has permissions to use Docker
    if not check_docker_permissions():
        print("❌ Permission denied when trying to access Docker.")
        print("\nTo fix this issue, you have several options:")
        print("1. Run the script with sudo:")
        print("   sudo python scripts/start_qdrant_docker.py")
        print("\n2. Add your user to the docker group (recommended):")
        print("   sudo usermod -aG docker $USER")
        print("   Then log out and log back in for the changes to take effect.")
        print("\n3. Use rootless Docker:")
        print("   https://docs.docker.com/engine/security/rootless/")
        sys.exit(1)

    # Check if Qdrant is already running
    if check_qdrant_running(args.port):
        print(f"Qdrant is already running on http://localhost:{args.port}")
        sys.exit(0)

    # Start Qdrant
    if start_qdrant_container(args.port, args.volume, args.name):
        print("\nQdrant is now ready to use!")
        print(f"To use it with build_index.py, run:")
        print(f"python scripts/build_index.py --download --use-docker --qdrant-url http://localhost:{args.port}")
        sys.exit(0)
    else:
        print("Failed to start Qdrant in Docker.")
        sys.exit(1)


if __name__ == "__main__":
    main()
