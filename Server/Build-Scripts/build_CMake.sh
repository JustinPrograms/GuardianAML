cd ~/builds
mkdir -p cmake-src
cd cmake-src

# Grab a 3.28.6 tarball 
wget https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6.tar.gz
tar -xzf cmake-3.28.6.tar.gz
cd cmake-3.28.6

./bootstrap --prefix=/usr/local
make -j"$(nproc)"
sudo make install

cmake --version
