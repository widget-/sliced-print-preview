{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "sliced-print-preview";

  buildInputs = with pkgs; [
    # --- Frontend ---
    nodejs_22
    bun

    # --- Rust pipeline ---
    cargo
    rustc

    # --- Deployment ---
    ansible

    # --- Utilities ---
    git
  ];

  shellHook = ''
    echo "sliced-print-preview dev environment"
    echo "  Bun $(bun --version)"
    echo "  Rust $(cargo --version 2>/dev/null | head -1 || echo 'not found')"
  '';
}
