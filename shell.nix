{ pkgs ? (import <nixpkgs> {}).pkgs }: with pkgs;
  let
    venvDir = "./.venv";
    opencvWithGtk = python311Packages.opencv4.override (old : { enableGtk2 = true; });
  in mkShell {
    buildInputs = [
      python311Packages.virtualenv # run virtualenv .
      python311Packages.pip

      # Those are dependencies that we would like to use from nixpkgs, which will
      # add them to PYTHONPATH and thus make them accessible from within the venv.
      opencvWithGtk
      python311Packages.flatbuffers
      python311Packages.numpy
      python311Packages.pygame
      python311Packages.tkinter
      python311Packages.imutils
      python311Packages.pyotp

      # In this particular example, in order to compile any binary extensions they may
      # require, the Python modules listed in the hypothetical requirements.txt need
      # the following packages to be installed locally:
      taglib
      openssl
      git
      libxml2
      libxslt
      libzip
      zlib
    ];

    # This is very close to how venvShellHook is implemented, but
    # adapted to use 'virtualenv'
    shellHook = ''
      # Fixes libstdc++ issues and libgl.so issues
      export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc
      ]}

      SOURCE_DATE_EPOCH=$(date +%s)

      if [ -d "${venvDir}" ]; then
        echo "Skipping venv creation, '${venvDir}' already exists"
      else
        echo "Creating new venv environment in path: '${venvDir}'"
        # Note that the module venv was only introduced in python 3, so for 2.7
        # this needs to be replaced with a call to virtualenv
        ${python311Packages.python.interpreter} -m venv "${venvDir}"
      fi

      # Under some circumstances it might be necessary to add your virtual
      # environment to PYTHONPATH, which you can do here too;
      # PYTHONPATH=$PWD/${venvDir}/${python311Packages.python.sitePackages}/:$PYTHONPATH

      source "${venvDir}/bin/activate"

      # As in the previous example, this is optional.
      # pip install -r requirements.txt
      pip install validate_email Py3DNS mediapipe
    '';
  }
