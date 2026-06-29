#!/usr/bin/env bash
set -euo pipefail

# ---
# Generate a read-only deploy key for the sliced-print-preview repo.
# After running this, add the public key to GitHub repo settings.
# ---

KEY_PATH="${HOME}/.ssh/sliced-print-preview-deploy-key"
REPO="widget-/sliced-print-preview"

if [ -f "$KEY_PATH" ]; then
  echo "⚠️  Key already exists at ${KEY_PATH}"
  echo "   To regenerate, delete it first: rm ${KEY_PATH} ${KEY_PATH}.pub"
  exit 1
fi

echo "→ Generating Ed25519 deploy key…"
ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "deploy:${REPO}"

echo ""
echo "✅ Key generated!"
echo ""
echo "─────────────────────────────────────────────"
echo "NEXT STEP: Add the public key to GitHub"
echo "─────────────────────────────────────────────"
echo ""
echo "1. Open this URL in your browser:"
echo "   https://github.com/${REPO}/settings/keys/new"
echo ""
echo "2. Copy and paste this public key:"
echo ""
cat "${KEY_PATH}.pub"
echo ""
echo "3. Check 'Allow read access' (NOT write access)"
echo "4. Click 'Add key'"
echo ""
echo "─────────────────────────────────────────────"
echo "AFTER THAT: Run the Ansible playbook:"
echo "─────────────────────────────────────────────"
echo ""
echo "   cd deploy/ansible"
echo "   ansible-playbook -i inventory.yml playbook.yml"
echo ""
echo "Make sure your SSH key for the VM is in your agent:"
echo "   ssh-add ~/.ssh/id_ed25519   # or wherever your OCI key is"
echo "   ssh o                       # should work without a password flag"
echo ""
