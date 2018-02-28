# -*- encoding:utf8 -*-
import sys
import random
import math
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf-8')

import os

class Preprocess():
    def __init__(self, path, header):
        self.feature = []
        self.output = []
        self.idx = 0
        files = os.listdir(path)
        with open('Header', 'r') as f:
            if path == './data/day':
                head = unicode(f.read()).split('\n')[1].split(' ')
            else:
                head = unicode(f.read()).split('\n')[0].split(' ')
        header = [head.index(word) for word in header]
        for each_file in files:
            is_header = 0
            with open(path+'/'+each_file, 'r') as c:
                cfile = c.read().split('\n')
                for line in cfile:
                    if is_header == 0:
                        is_header = 1
                        continue
                    tmp_line = line.split(',')

                    try:
                        month = int(tmp_line[1].split('-')[1])
                        if month == 12 or month < 3:
                            self.output.append([0, 0, 1, ])
                        elif month < 6:
                            self.output.append([1, 0, 0, ])
                        elif month < 9:
                            self.output.append([0, 1, 0, ])
                        else:
                            self.output.append([1, 0, 0, ])
                    except:
                        pass

                    line_data = []
                    for h_idx in header:
                        try:
                            if h_idx == 1:
                                line_data.append(float(tmp_line[h_idx].split(' ')[1].split(':')[0]))
                            elif tmp_line[h_idx] == '':
                                line_data.append(-1.0)
                            else:
                                line_data.append(float(tmp_line[h_idx]))
                        except:
                            pass
                    if line_data != []:
                        self.feature.append(line_data)
        # for t in zip(self.feature, self.output):
        #     print t
        # print self.feature

    def normalization(self):
        mean = [0] * len(self.feature[0])
        minimum = self.feature[0]
        maximum = self.feature[1]
        for line in self.feature:
            mean = [mean[i]+line[i]/float(len(self.feature)) for i in range(len(line))]
            for idx in range(len(line)):
                if minimum[idx] > line[idx]:
                    minimum[idx] = line[idx]
                if maximum[idx] < line[idx]:
                    maximum[idx] = line[idx]
        min_max = [maximum[i] - minimum[i] for i in range(len(maximum))]
        data = []
        for line in self.feature:
            data.append([(line[i]-mean[i])/min_max[i] for i in range(len(line))])
        self.feature = data
        

    def next_batch(self, batch_size, randomly):
        if randomly:
            idx = random.sample(range(len(self.feature)), batch_size)
            batch_feature = [self.feature[m_idx] for m_idx in idx]
            batch_output = [self.output[m_idx] for m_idx in idx]
        else:
            batch_feature = self.feature[self.idx:self.idx+batch_size]
            batch_output = self.output[self.idx:self.idx + batch_size]
            self.idx += batch_size
        return batch_feature, batch_output

    def logarithmatic(self):
        for idx1 in range(len(self.feature)):
            for idx2 in range(len(self.feature[0])):
                try:
                    self.feature[idx1][idx2] = math.log(self.feature[idx1][idx2], 100)
                except:
                    self.feature[idx1][idx2] = 0

    def visualization(self, idx):
        data = []
        for d in self.feature[0]:
            data.append([d])
        for line in self.feature[1:]:
            for idx in range(len(line)):
                data[idx].append(line[idx])
            # data = [data[idx].append(d[idx]) for idx in range(len(d))]
        plt.hist(data[idx], bins=20)
        plt.show()

# data = Preprocess('./data', [u'시간', u'기온', u'강수량', u'습도'])
# data.logarithmatic()
# data.normalization()
# data.visualization(1)
# print data.next_batch(10)
# print random.sample(range(6), 3)