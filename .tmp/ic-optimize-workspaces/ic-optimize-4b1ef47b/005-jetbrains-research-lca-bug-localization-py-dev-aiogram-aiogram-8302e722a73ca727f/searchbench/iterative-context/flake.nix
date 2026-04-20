{
  description = "Iterative-context development environment";

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
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python3
            pkgs.uv
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
