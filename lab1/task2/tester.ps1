$program_temp1 = "nvprof --log-file logs\text"
$program_temp2 = ".log ..\test.exe"

For($i = 1; $i -le 1024; $i = $i + 10){
	Invoke-Expression "$program_temp1$i$program_temp2 $i"
}