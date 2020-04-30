export MINICONDA_PATH=$HOME/miniconda;
export MINICONDA_PATH_WIN=`cygpath --windows $MINICONDA_PATH`;
export MINICONDA_SUB_PATH=$MINICONDA_PATH/Scripts;
export MINICONDA_LIB_BIN_PATH=$MINICONDA_PATH/Library/bin;

choco install openssl.light;

echo "folder $MINICONDA_SUB_PATH does not exist"
echo "installing miniconda for windows";
choco install miniconda3 --params="'/JustMe /AddToPath:1 /D:$MINICONDA_PATH_WIN'";

export PATH="$MINICONDA_PATH:$MINICONDA_SUB_PATH:$MINICONDA_LIB_BIN_PATH:$PATH";

# begin checking miniconda existance
echo "checking if folder $MINICONDA_SUB_PATH exists"
if [[ -d $MINICONDA_SUB_PATH ]]; then
    echo "folder $MINICONDA_SUB_PATH exists"
else
    echo "folder $MINICONDA_SUB_PATH does not exist"
fi;
# end checking miniconda existance

source $MINICONDA_PATH/etc/profile.d/conda.sh;
hash -r;
python --version
conda config --set always_yes yes --set changeps1 no;
conda update -q conda;
conda info -a
