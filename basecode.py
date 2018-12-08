import csv


database=[]
with open('D:\\Work\\Neibhourhood roughness set\\voilin\\testFile.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        database.append(row)

for x in database:
    print(x)
csvFile.close()

#--------------------------------------------------------------

delta=[]
for x in range(len(database)):
    delta.append([])
    for y in database[x]:
        delta[x].append([])


print("\n-------------\n")

for y in range(1,len(database[1])):
    for x in range(1,len(database)):
        for xn in range(1,len(database)):
            if(database[x][y]==database[xn][y]):
                delta[xn][y].append(x)


#---------------------------------------

delta_set=delta
for x in range(len(delta_set)):
    for y in range(len(delta_set[x])):
        delta_set[x][y]=set(delta_set[x][y])


for x in delta_set:
    print(x)

#-----------------------------------------


#
#
# c=0
# for y in range(1,len(database[1])):
#
#     for x in range(1,len(database)):
#          for xn in range(1,len(database)):
#              if(database[x-1][y-1]==database[xn-1][y-1]):
#                  delta[y-1].append(xn)
#
