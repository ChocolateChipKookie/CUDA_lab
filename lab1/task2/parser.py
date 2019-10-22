import os
from matplotlib import pyplot as plt

'''Preprocessing'''
path = '.\\logs'
results = {}
ls = []
for f in os.listdir(path):
    index = int(f[4: f.find('.')])

    with open(path + '\\' + f) as file:
        for i, line in enumerate(file):
            if i is 4:
                duration = float(line[line.find('%') + 2 : line.find('us')])
                results[index] = duration
                ls.append((index, duration))
                break

'''Printing of sorted out results'''
ls.sort(key=lambda x: x[1])
sorted_out = open('sorted_output.txt', 'w')
for t in ls:
    sorted_out.write(t.__str__() + '\n')

'''Printing of times in order'''

xs = sorted([t for t in results])
ys = [results[x] for x in xs]
times = open('times.txt', 'w')

for t in zip(xs, ys):
    times.write(t.__str__() + '\n')

'''Plot'''
plt.plot(xs, ys, 'bo')
plt.savefig('plot.png')

'''Plot without first element'''
plt.clf()
xs = xs[1:]
ys = ys[1:]
plt.plot(xs, ys, 'bo')
plt.savefig('plot_without_first.png')
