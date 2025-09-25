#!/usr/bin/env python3
"""
Script to automatically check and update all packages in requirements.txt to their latest versions.
"""

import subprocess
import sys
import re
from typing import Dict, List, Tuple
import json

def get_latest_version(package_name: str) -> str:
    """Get the latest version of a package from PyPI.
    
    Args:
        package_name (str): Name of the package
        
    Returns:
        str: Latest version string or empty string if not found
    """
    try:
        # Use pip index versions to get the latest version
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Parse the output to get the latest version
            output = result.stdout.strip()
            if output:
                # Extract version from first line: "package_name (version)"
                match = re.search(r'\(([^)]+)\)', output.split('\n')[0])
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"Error getting version for {package_name}: {e}")
        
    return ""

def parse_requirements(file_path: str) -> List[Tuple[str, str, str]]:
    """Parse requirements.txt file.
    
    Args:
        file_path (str): Path to requirements.txt
        
    Returns:
        List[Tuple[str, str, str]]: List of (package_name, current_version, line_content)
    """
    packages = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '==' in line:
                    # Parse package==version
                    parts = line.split('==')
                    if len(parts) == 2:
                        package_name = parts[0].strip()
                        current_version = parts[1].strip()
                        packages.append((package_name, current_version, line))
                        
    except Exception as e:
        print(f"Error reading requirements file: {e}")
        
    return packages

def update_requirements_file(file_path: str, updates: Dict[str, str]) -> None:
    """Update requirements.txt with new versions.
    
    Args:
        file_path (str): Path to requirements.txt
        updates (Dict[str, str]): Dictionary of package_name -> new_version
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Update each package version
        for package_name, new_version in updates.items():
            # Use regex to replace the version
            pattern = rf'^({re.escape(package_name)}==)[^\s]+(.*)$'
            replacement = rf'\g<1>{new_version}\g<2>'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(content)
            
        print(f"✅ Updated requirements.txt with {len(updates)} package updates")
        
    except Exception as e:
        print(f"❌ Error updating requirements file: {e}")

def main():
    """Main function to check and update all package versions."""
    requirements_file = "requirements.txt"
    
    print("🔍 Checking current packages in requirements.txt...")
    packages = parse_requirements(requirements_file)
    
    if not packages:
        print("❌ No packages found in requirements.txt")
        return
    
    print(f"📦 Found {len(packages)} packages to check")
    print("=" * 80)
    
    updates = {}
    unchanged = []
    errors = []
    
    for package_name, current_version, _ in packages:
        print(f"🔎 Checking {package_name}...", end=" ", flush=True)
        
        latest_version = get_latest_version(package_name)
        
        if latest_version:
            if latest_version != current_version:
                print(f"📈 {current_version} → {latest_version}")
                updates[package_name] = latest_version
            else:
                print(f"✅ {current_version} (already latest)")
                unchanged.append(package_name)
        else:
            print(f"❌ Could not get version")
            errors.append(package_name)
    
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   🔄 Updates available: {len(updates)}")
    print(f"   ✅ Already latest: {len(unchanged)}")
    print(f"   ❌ Errors: {len(errors)}")
    
    if updates:
        print(f"\n📋 Updates to be applied:")
        for package, version in updates.items():
            current = next(v for p, v, _ in packages if p == package)
            print(f"   • {package}: {current} → {version}")
        
        # Ask for confirmation
        response = input(f"\n❓ Apply {len(updates)} updates? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            update_requirements_file(requirements_file, updates)
            print("\n🎉 All updates applied successfully!")
            print("💡 Run 'pip install -r requirements.txt' to install updated packages")
        else:
            print("❌ Updates cancelled")
    else:
        print("\n🎉 All packages are already at their latest versions!")
    
    if errors:
        print(f"\n⚠️  Could not check versions for: {', '.join(errors)}")

if __name__ == "__main__":
    main()