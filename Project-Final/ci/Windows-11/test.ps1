$location = Get-Location

cd build/Tests

ctest

Set-Location $location
