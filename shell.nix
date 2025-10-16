let
  pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;
  rPacks = with pkgs.rPackages; [
    data_table
    tidyverse
  ];
  myR = pkgs.rWrapper.override{ packages = rPacks; };
  myRStudio = pkgs.rstudioWrapper.override { packages = rPacks; };
in with pkgs; {
  busData = stdenv.mkDerivation {
    name = "bus-data";
    version = "0.1.0.0";
    src = if pkgs.lib.inNixShell then null else nix;
    buildInputs = [
      myR
      myRStudio
    ];
  };
}
