let
  pkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/11cb3517b3af6af300dd6c055aeda73c9bf52c48.tar.gz";
    sha256 = "sha256:1915r28xc4znrh2vf4rrjnxldw2imysz819gzhk9qlrkqanmfsxd";
  }) {};
  # pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;
  rPacks = with pkgs.rPackages; [
    data_table
    tidyverse
  ];
  myR = pkgs.rWrapper.override{ packages = rPacks; };
  myRStudio = pkgs.rstudioWrapper.override { packages = rPacks; };
  myPy = pkgs.python312.withPackages (pykgs: with pykgs; [
    numpyro
  ]);
in with pkgs; {
  busData = stdenv.mkDerivation {
    name = "bus-data";
    version = "0.1.0.0";
    src = if pkgs.lib.inNixShell then null else nix;
    buildInputs = [
      myR
      myRStudio
      myPy
    ];
  };
}
