let
  pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;
in with pkgs; {
  busData = stdenv.mkDerivation {
    name = "bus-data";
    version = "0.1.0.0";
    src = if pkgs.lib.inNixShell then null else nix;
    buildInputs = with rPackages; [
      R
      data_table
      tidyverse
    ];
  };
}
