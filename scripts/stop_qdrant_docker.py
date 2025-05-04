#!/usr/bin/env python
"""Script to stop and optionally remove the Qdrant Docker container."""

import argparse
import subprocess
import sys


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


def stop_qdrant_container(container_name="qdrant-contexto", remove=False):
    """Stop the Qdrant Docker container.

    Args:
        container_name: Name of the Docker container
        remove: Whether to remove the container after stopping

    Returns:
        True if successful, False otherwise
    """
    # Check if container exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if container_name not in result.stdout:
        print(f"Container '{container_name}' does not exist.")
        return True

    # Check if it's running
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if container_name in result.stdout:
        print(f"Stopping container '{container_name}'...")
        try:
            subprocess.run(["docker", "stop", container_name], check=True)
            print(f"Container '{container_name}' stopped.")
        except subprocess.CalledProcessError as e:
            print(f"Error stopping container: {e}")
            return False
    else:
        print(f"Container '{container_name}' is not running.")

    # Remove the container if requested
    if remove:
        print(f"Removing container '{container_name}'...")
        try:
            subprocess.run(["docker", "rm", container_name], check=True)
            print(f"Container '{container_name}' removed.")
        except subprocess.CalledProcessError as e:
            print(f"Error removing container: {e}")
            return False

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Stop and optionally remove the Qdrant Docker container.")
    parser.add_argument("--name", type=str, default="qdrant-contexto", help="Name of the Docker container")
    parser.add_argument("--remove", action="store_true", help="Remove the container after stopping")

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
        print("   sudo python scripts/stop_qdrant_docker.py")
        print("\n2. Add your user to the docker group (recommended):")
        print("   sudo usermod -aG docker $USER")
        print("   Then log out and log back in for the changes to take effect.")
        print("\n3. Use rootless Docker:")
        print("   https://docs.docker.com/engine/security/rootless/")
        sys.exit(1)

    # Stop the container
    if stop_qdrant_container(args.name, args.remove):
        print("Done.")
        sys.exit(0)
    else:
        print("Failed to stop Qdrant container.")
        sys.exit(1)


if __name__ == "__main__":
    main()
