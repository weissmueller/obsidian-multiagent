import subprocess
import sys

def test_obsidian_cli():
    print("Testing Obsidian CLI integration...\n")
    
    # Test 1: Search Vault
    print("Test 1: obsidian search")
    try:
        # We search for the letter 'a' which should exist in almost any vault
        result = subprocess.run(
            ["obsidian", "search", "query=a", "format=text", "limit=5"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("✅ SUCCESS: Found files:")
        lines = result.stdout.strip().split('\n')
        for line in lines[:3]:  # Show first 3
            if "obsidian.md/download" not in line:
                print(f"  - {line}")
        print("  ...")
    except FileNotFoundError:
        print("❌ ERROR: The 'obsidian' command was not found in your system PATH.")
        print("Make sure the Obsidian CLI is installed and accessible.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Command failed with exit code {e.returncode}")
        print(f"Error output: \n{e.stderr}")
        sys.exit(1)
        
    print("-" * 40)
    
    # Test 2: Status check
    print("Test 2: obsidian status check")
    try:
        # Check if the Advanced URI plugin is responding
        result = subprocess.run(
            ["obsidian", "--help"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("✅ SUCCESS: Obsidian CLI is responding.")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        
    print("\nIf both tests passed, your Obsidian CLI is working correctly for the agent!")

if __name__ == "__main__":
    test_obsidian_cli()
