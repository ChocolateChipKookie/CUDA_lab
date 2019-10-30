import os
from matplotlib import pyplot as plt

'''Preprocessing'''
path = '.\\logs_elements'
results = {}
for f in os.listdir(path):
    index = int(f[4: f.find('.')])

    with open(path + '\\' + f) as file:
        for line in file:
            if 'vectorAdd' in line:
                duration = float(line[line.find('%') + 2: line.find('us')])
                results[index] = duration
                break

'''Averages'''
'''Printing in order'''
xs = sorted([t for t in results])
ys = [results[x] / x for x in xs]
times = open('times_average.txt', 'w')

zipped = [x for x in zip(xs, ys)]

for t in zipped:
    times.write(t.__str__() + '\n')

'''Printing sorted'''
zipped.sort(key=lambda x: x[1])
sorted_out = open('sorted_output_average.txt', 'w')

for t in zipped:
    sorted_out.write(t.__str__() + '\n')

'''Plot'''
plt.yscale("log")
plt.plot(xs, ys, 'bo')
plt.savefig('plot_average.png')

'''Normal'''
# Setting new Y's
ys = [results[x] for x in xs]
'''Printing normal'''
times = open('times.txt', 'w')
zipped = [x for x in zip(xs, ys)]

for t in zipped:
    times.write(t.__str__() + '\n')

'''Printing sorted'''
zipped.sort(key=lambda x: x[1])
sorted_out = open('sorted_output.txt', 'w')

for t in zipped:
    sorted_out.write(t.__str__() + '\n')

plt.clf()
plt.plot(xs, ys, 'bo')
plt.savefig('plot_times.png')
