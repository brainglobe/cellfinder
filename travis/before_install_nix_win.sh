export MINICONDA_PATH=$HOME/miniconda;
export MINICONDA_PATH_WIN=`cygpath --windows $MINICONDA_PATH`;
export MINICONDA_SUB_PATH=$MINICONDA_PATH/Scripts;
export MINICONDA_LIB_BIN_PATH=$MINICONDA_PATH/Library/bin;

choco install openssl.light;
