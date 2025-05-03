$location = Get-Location

cd build/Tests

ctest --output-on-failure

Set-Location $location
