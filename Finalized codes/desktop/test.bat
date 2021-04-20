@echo off
cmd /k mysql -u praveen -p1234 -h 192.168.8.105 && use mynewdb; && select * from (select 'id' ,'Module_name', 'Module_code', 'Lecturer_name', 'Venue', 'Date', 'Starting_time',  'Ending_time'union all (select * from lecturelist)) resulting_set into outfile "D:\\lecturelist.csv" fields terminated by ',' lines terminated by'\n';



