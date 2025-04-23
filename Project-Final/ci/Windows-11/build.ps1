if (-not (Test-Path -path "build"))
{
    mkdir ./build
}

$location = Get-Location

cd ./build

cmake ../

Set-Location $location
