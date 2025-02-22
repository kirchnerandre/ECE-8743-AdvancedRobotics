
pushd ./
cd build/Tests
ctest --output-on-failure --build-config Debug
popd
