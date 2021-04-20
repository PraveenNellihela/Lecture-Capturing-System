import schedule 
import time
import pandas
import calendar
import csv
import datetime
import subprocess


def job(duration_minutes):
    arg1 = duration_minutes
    print ("Capturing Audio and video........")
    print(duration_minutes)

    item = subprocess.Popen(["Capture_Audio_video.bat", str(arg1)],
                            shell=True, stdout=subprocess.PIPE)
    for line in item.stdout:
        print (line)  

df = pandas.DataFrame.from_csv('client1.csv',index_col='ID1')
print(df)

##import csv
f = open('client1.csv')
csv_f = csv.reader(f)

    
my_date=datetime.date.today()
datetoday = calendar.day_name[my_date.weekday()]
print (datetoday)

for row in csv_f:
    string_date = row[4]
    starttime = row [5]
    endtime = row[6]



    if string_date =="Monday":        
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().monday.at(starttime).do(job, duration_minutes)
        
        
        
    elif string_date == "Tuesday":
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().tuesday.at(starttime).do(job, duration_minutes)
        
    
    elif string_date == "Wednesday":
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().wednesday.at(starttime).do(job, duration_minutes)
        
        
    elif string_date == "Thursday":
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().thursday.at(starttime).do(job, duration_minutes)
        
        
    elif string_date == "Friday":
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().friday.at(starttime).do(job, duration_minutes)
        
    elif string_date == "Saturday":
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().saturday.at(starttime).do(job, duration_minutes)
        

    elif string_date == "Sunday":
        duration = datetime.datetime.strptime(endtime,'%H:%M')-datetime.datetime.strptime(starttime,'%H:%M')
        duration_minutes = int(duration.total_seconds()/60)
        print (duration_minutes)
        schedule.every().sunday.at(starttime).do(job, duration_minutes)
        
while True:
    schedule.run_pending()
    time.sleep(0.3)
##    
##    
##    
####schedule.every(10).minutes.do(job)
####schedule.every().hour.do(job)
####schedule.every().day.at("10:30").do(job)
####schedule.every().monday.do(job)
###schedule.every().at("20:45").do(job)
##
##
