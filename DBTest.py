import mysql.connector

server = "cst04.ddns.net"
login = "mirror"
database = "datatable"

#connect to mysql server
MirrorDB = mysql.connector.connect(
    host=server,
    user=login,
    passwd=login,
    database=database
)

#print(MirrorDB)

cursor = MirrorDB.cursor()

cursor.execute("SHOW TABLES")

for x in cursor:
    print(x)

#print(cursor.arraysize)