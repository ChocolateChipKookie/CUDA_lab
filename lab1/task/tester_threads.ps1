$program_temp1 = "nvprof --log-file logs\text"
$program_temp2 = ".log ..\test\x64\Release\test.exe"

For($i = 1; $i -le 1024; ++$i){
	Invoke-Expression "$program_temp1$i$program_temp2 50000 $i"
}