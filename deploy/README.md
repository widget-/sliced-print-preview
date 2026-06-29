# Deploy: sliced-print-preview on Oracle Cloud ARM VM

All the files you need to provision the Oracle Cloud free-tier ARM VM and deploy
the 3D print preview web app.

## What you'll end up with

```
┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐
│  Internet     │───▶│  nginx:443   │───▶│  Bun backend:3000     │
│  (HTTPS)      │    │  (certbot)   │    │  - serves frontend    │
│               │    │              │    │  - runs OrcaSlicer    │
│               │    │              │    │  - calls gcode2segbin │
└──────────────┘    └──────────────┘    └───────────────────────┘
```

## Prerequisites

| Requirement | How to check |
|-------------|-------------|
| `ssh o` works | `ssh o whoami` |
| Ansible ≥ 2.15 | `ansible --version` |
| Ansible on Nix | `nix-shell -p ansible` |
| Your OCI SSH key loaded | `ssh-add -l` |
| GitHub SSH access | `ssh -T git@github.com` |

## Step-by-step

### 1. Generate a deploy key

```bash
cd deploy
chmod +x setup-deploy-key.sh
./setup-deploy-key.sh
```

This creates `~/.ssh/sliced-print-preview-deploy-key` and prints the public key.
Follow the on-screen instructions to add it as a read-only deploy key on GitHub:

https://github.com/widget-/sliced-print-preview/settings/keys/new

Check **"Allow read access"** (NOT write).

### 2. Run the Ansible playbook

```bash
cd deploy/ansible
ansible-playbook -i inventory.yml playbook.yml
```

This will take **15–30 minutes** on the first run (Rust build, frontend build,
downloads). Subsequent runs are incremental.

### 3. Set up TLS (after you have a domain)

If you have a domain pointed at the VM, re-run the playbook with the domain:

```bash
cd deploy/ansible
ansible-playbook -i inventory.yml playbook.yml \
  -e domain=preview.yourdomain.com \
  -e admin_email=you@yourdomain.com
```

Certbot will obtain a Let's Encrypt certificate and configure nginx to redirect
HTTP → HTTPS automatically.

### 4. Verify it works

Visit the domain (or the VM's public IP) in a browser. You should see the
upload page where you can drop an STL file and get a preview.

## Files

| File | Purpose |
|------|---------|
| `setup-deploy-key.sh` | Generate a GitHub deploy key |
| `ansible/inventory.yml` | Host definition (target: `o`) |
| `ansible/playbook.yml` | Full provisioning playbook |
| `ansible/files/sliced-print-preview.service` | systemd unit for the backend |
| `ansible/files/nginx.conf.j2` | nginx reverse proxy config (Jinja2) |

## What the playbook does

On the target VM (`o`):

1. **System packages** — curl, git, nginx, certbot, fuse, build-essential
2. **Bun** — JavaScript runtime (from bun.sh install script)
3. **Rust** — via rustup (minimal profile)
4. **OrcaSlicer** — downloads the ARM64 AppImage, extracts it to `/opt/orca-slicer`
5. **Deploy key** — copies your GitHub deploy key to the VM, configures SSH
6. **Clone repo** — `git clone` using the deploy key
7. **Build Rust CLI** — `cargo build --release` for `gcode-to-segbin`
8. **Build frontend** — `bun install && bun run build`
9. **Build backend** — `bun install && bun run build`
10. **Environment** — creates `.env` with all paths
11. **systemd service** — enables the backend to start on boot
12. **nginx** — reverse proxy with file upload limits and timeouts
13. **Certbot** — obtains TLS certificate (if domain is configured)

## Manual steps if you skip Ansible

If you prefer to set things up manually on the VM:

```bash
# 1. Install Bun
curl -fsSL https://bun.sh/install | bash

# 2. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# 3. Install OrcaSlicer
cd /tmp
wget https://github.com/SoftFever/OrcaSlicer/releases/download/v2.4.1/OrcaSlicer_Linux_AppImage_Ubuntu2404_aarch64_V2.4.1.AppImage
chmod +x OrcaSlicer_Linux_AppImage_Ubuntu2404_aarch64_V2.4.1.AppImage
./OrcaSlicer_Linux_AppImage_Ubuntu2404_aarch64_V2.4.1.AppImage --appimage-extract
sudo mv squashfs-root /opt/orca-slicer/
sudo ln -s /opt/orca-slicer/squashfs-root/AppRun /usr/local/bin/orca-slicer

# 4. Clone repo (needs deploy key set up first)
git clone git@github.com:widget-/sliced-print-preview.git
cd sliced-print-preview/packages/gcode-to-segbin && cargo build --release
cd ../frontend && bun install && bun run build
cd ../backend && bun install && bun run build

# 5. Create .env
cat > .env << 'ENVEOF'
PORT=3000
UPLOAD_DIR=/tmp/print-preview-uploads
OUTPUT_DIR=/tmp/print-preview-output
ORCA_SLICER_BIN=/usr/local/bin/orca-slicer
ORCA_RESOURCES_DIR=/opt/orca-slicer/squashfs-root/resources
GCODE_TO_SEGBIN_BIN=/home/ubuntu/sliced-print-preview/packages/gcode-to-segbin/target/release/gcode-to-segbin
NODE_ENV=production
ENVEOF

# 5b. Create upload/output directories
mkdir -p /tmp/print-preview-uploads /tmp/print-preview-output

# 6. Create systemd service (see files/sliced-print-preview.service)
sudo cp sliced-print-preview.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sliced-print-preview

# 7. Set up nginx (see files/nginx.conf.j2)
```
