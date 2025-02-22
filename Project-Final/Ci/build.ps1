
if (Test-Path ./build)
{

}
else
{
    mkdir ./build
}

pushd ./

cd ./build
cmake ..
&"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\msbuild" ProjectFinal.sln

popd
