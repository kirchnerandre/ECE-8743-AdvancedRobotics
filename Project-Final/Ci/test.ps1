
pushd ./
cd build
ctest --output-on-failure --build-config Debug
popd
