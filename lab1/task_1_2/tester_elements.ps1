$program_temp1 = "nvprof --log-file logs_elements\text"
$program_temp2 = ".log ..\test\x64\Release\test.exe"

For($i = 1000; $i -le 200000; $i = $i + 1000){
	Invoke-Expression "$program_temp1$i$program_temp2 $i 256"
}