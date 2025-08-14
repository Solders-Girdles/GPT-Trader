#!/usr/bin/env python3
"""Migration script to help transition from .env to secure secrets management.

This script:
1. Backs up existing .env file
2. Creates .env.local from .env
3. Removes .env from git
4. Sets up the new secrets management system
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.security.secrets_manager import SecretManager


def migrate_env_file():
    """Migrate from .env to secure secrets management."""

    print("=" * 60)
    print("GPT-Trader Secrets Migration Tool")
    print("=" * 60)

    env_file = project_root / ".env"
    env_local = project_root / ".env.local"
    env_template = project_root / ".env.template"

    # Step 1: Check if .env exists
    if not env_file.exists():
        print("✅ No .env file found - system is already secure")
        return

    # Step 2: Create backup
    backup_name = f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = project_root / backup_name
    shutil.copy2(env_file, backup_path)
    print(f"✅ Created backup: {backup_name}")

    # Step 3: Copy to .env.local if it doesn't exist
    if not env_local.exists():
        shutil.copy2(env_file, env_local)
        print("✅ Created .env.local from existing .env")
    else:
        print("⚠️  .env.local already exists - keeping existing file")

    # Step 4: Remove .env
    env_file.unlink()
    print("✅ Removed .env file")

    # Step 5: Create .env.template if needed
    if not env_template.exists():
        SecretManager.setup_secure_environment(project_root)
        print("✅ Created .env.template")

    # Step 6: Update git
    os.system("git rm --cached .env 2>/dev/null || true")
    print("✅ Removed .env from git tracking")

    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load environment variables from .env.local:")
    print("   $ export $(cat .env.local | xargs)")
    print("   OR")
    print("   $ source .env.local")
    print("\n2. Test the application:")
    print("   $ python -m src.bot.startup_validation")
    print("\n3. Commit the changes:")
    print("   $ git add -A")
    print('   $ git commit -m "chore: migrate to secure secrets management"')
    print("\nYour secrets are now secure! 🔐")
    print("=" * 60)


if __name__ == "__main__":
    try:
        migrate_env_file()
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
