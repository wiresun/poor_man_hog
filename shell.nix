let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/refs/tags/25.11.tar.gz";
    sha256 = "1zn1lsafn62sz6azx6j735fh4vwwghj8cc9x91g5sx2nrg23ap9k";
  }) { };
in
pkgs.callPackage (
  {
    mkShell,
    python3,
    basedpyright,
    ruff,
  }:
  mkShell {
    strictDeps = true;
    nativeBuildInputs = [
      basedpyright
      ruff
      (python3.withPackages (python-pkgs: with python-pkgs; [
        numpy
        pillow
        scikit-image
      ]))
    ];
  }
) { }
