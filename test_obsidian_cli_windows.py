import subprocess
import sys
import platform

def test_obsidian_cli_windows():
    print(f"Testing Obsidian CLI integration on {platform.system()}...\n")
    
    # Test 1: Search Vault
    print("Test 1: obsidian search (Windows Shell Mode)")
    try:
        # Notice we are using a single string and shell=True which is often required on Windows
        result = subprocess.run(
            "obsidian search query=a format=text limit=5", 
            capture_output=True, 
            text=True, 
            check=True,
            shell=True  # Critial for Windows PATH resolution
        )
        print("✅ SUCCESS: Found files:")
        lines = result.stdout.strip().split('\n')
        for line in lines[:3]:  # Show first 3
            if "obsidian.md/download" not in line:
                print(f"  - {line}")
        print("  ...")
    except FileNotFoundError:
        print("❌ ERROR: The 'obsidian' command was not found.")
        print("Make sure the Obsidian CLI is installed and accessible in your Windows PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Command failed with exit code {e.returncode}")
        print(f"Error output: \n{e.stderr}")
        sys.exit(1)
        
    print("-" * 40)
    
    # Test 2: Status check
    print("Test 2: obsidian status check (Windows Shell Mode)")
    try:
        result = subprocess.run(
            "obsidian --help", 
            capture_output=True, 
            text=True, 
            check=True,
            shell=True  # Critial for Windows PATH resolution
        )
        print("✅ SUCCESS: Obsidian CLI is responding.")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        
    print("\nIf both tests passed, your Obsidian CLI is working correctly for the agent on Windows!")

if __name__ == "__main__":
    test_obsidian_cli_windows()
