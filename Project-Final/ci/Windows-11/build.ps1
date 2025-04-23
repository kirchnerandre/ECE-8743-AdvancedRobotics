if (-not (Test-Path -path "build"))
{
    mkdir ./build
}

$location = Get-Location

cd ./build

cmake ../

if (-not (Test-Path -path "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"))
{
    Write-Error "MSBuild.exe not found"
}
else
{
    $env:path = "$env:path; C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin"

    MSBuild ProjectFinal.sln
}

Set-Location $location
