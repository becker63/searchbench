{
  description = "searchbench development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    small = {
      url = "github:pallets/itsdangerous";
      flake = false;
    };

    medium = {
      url = "github:pallets/click";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      small,
      medium,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        loopVizFeh = pkgs.writeShellApplication {
          name = "loop-viz-feh";
          runtimeInputs = [ pkgs.uv pkgs.graphviz pkgs.xdg-utils pkgs.python3 ];
          text = ''
            set -euo pipefail
            tmpdir="$(mktemp -d)"
            repair="$tmpdir/repair.png"
            opt="$tmpdir/optimization.png"

            uv run harness/tools/loop_viz.py --format png --machine repair --output "$repair"
            uv run harness/tools/loop_viz.py --format png --machine optimization --output "$opt"

            echo "Rendered: $repair"
            echo "Rendered: $opt"

            opener="xdg-open"
            if command -v setsid >/dev/null 2>&1; then
              setsid "$opener" "$repair" </dev/null >/dev/null 2>&1 &
              setsid "$opener" "$opt" </dev/null >/dev/null 2>&1 &
            else
              "$opener" "$repair" </dev/null >/dev/null 2>&1 &
              "$opener" "$opt" </dev/null >/dev/null 2>&1 &
            fi
            echo "xdg-open launch attempted for both images. If windows did not appear, run: xdg-open \"$repair\" \"$opt\""
            echo "Files remain in $tmpdir"
          '';
        };
      in
      {
        packages = {
          inherit loopVizFeh;
          default = loopVizFeh;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python3
            pkgs.uv
            pkgs.graphviz
            pkgs.xdg-utils
            loopVizFeh
          ];

          shellHook = ''
            export TEST_REPO_SMALL="${small}"
            export TEST_REPO_MEDIUM="${medium}"

            echo "Test repos:"
            echo "  small  -> ${small}"
            echo "  medium -> ${medium}"
          '';
        };
      }
    );
}
