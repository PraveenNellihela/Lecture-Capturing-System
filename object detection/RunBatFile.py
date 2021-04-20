import schedule 
import time
import pandas
import calendar
import csv
import datetime
import subprocess


def job(end_time):
    print('doing job')
    arg1 = end_time
    filepath = r'C:\tensorflow1\models\research\object_detection\scheduledAutoRun.bat'
    subprocess.Popen([filepath, arg1],shell=True)
    print ("Im working....")
    print(end_time)
    
    
    

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
##    print('string date is; '+str(string_date)+' and end_time is; '+str(endtime))
##    print(string_date)
######    print(endtime)


    if string_date =="Monday":
        end_time = endtime
        schedule.every().monday.at(starttime).do(job, end_time)

    elif string_date == "Tuesday":
        end_time = endtime
        schedule.every().tuesday.at(starttime).do(job, end_time)
        
    elif string_date == "Wednesday":
        end_time = endtime
        schedule.every().wednesday.at(starttime).do(job, end_time)
        
    elif string_date == "Thursday":
        end_time = endtime
        schedule.every().thursday.at(starttime).do(job, end_time)
        
    elif string_date == "Friday":
        end_time = endtime
        schedule.every().friday.at(starttime).do(job, end_time)

    elif string_date == "Saturday":
        end_time = endtime
        schedule.every().saturday.at(starttime).do(job, end_time)
        print('string date is; '+str(string_date)+' and end_time is; '+str(endtime))
        
    elif string_date == "Sunday":
        end_time = endtime
        schedule.every().sunday.at(starttime).do(job, end_time)
        print('string date is; '+str(string_date)+' and end_time is; '+str(endtime))

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
