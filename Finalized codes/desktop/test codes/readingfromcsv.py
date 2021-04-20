import pandas
import calendar
import schedule
import csv
import datetime
import time
 
# Load csv
df = pandas.DataFrame.from_csv('client1.csv',index_col='ID1')
print(df)

##import csv
f = open('client1.csv')
csv_f = csv.reader(f)
##print(f)

date1 = df['Date']
print (date1)


start_time = df['Start time']
print (start_time)


def job():
    print('JOB')


my_date=datetime.date.today()
datetoday = calendar.day_name[my_date.weekday()]

systemtime = datetime.datetime.now().time()
currenttime = systemtime.strftime("%H:%M")
print (currenttime)
while (1):
    for row in csv_f:
        
        string_date = row[4]
        starttime = row[5]
        #print starttime

        if string_date == datetoday :
            print (string_date)
            print (starttime)
        
      
           
            
        
##            if currenttime ==starttime:
##                print ('GOT IT')
##        

