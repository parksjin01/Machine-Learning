# -*- encoding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os

class Preprocess():
    def __init__(self, path, header):
        self.feature = []
        self.output = []
        files = os.listdir(path)
        with open('Header', 'r') as f:
            head = unicode(f.read()).split(' ')
        header = [head.index(word) for word in header]
        print header
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
                            self.output.append([1, 0, 0, 0])
                        elif month < 6:
                            self.output.append([0, 1, 0, 0])
                        elif month < 9:
                            self.output.append([0, 0, 1, 0])
                        else:
                            self.output.append([0, 0, 0, 1])
                    except:
                        print tmp_line
                        pass

                    line_data = []
                    for h_idx in header:
                        try:
                            if h_idx == 1:
                                line_data.append(float(tmp_line[h_idx].split(':')))
                            elif tmp_line[h_idx] == '':
                                line_data.append(-1.0)
                            else:
                                line_data.append(float(tmp_line[h_idx]))
                        except:
                            print tmp_line, h_idx
                    if line_data != []:
                        self.feature.append(line_data)
        # for t in zip(self.feature, self.output):
        #     print t
        # print self.feature
        print len(self.feature), len(self.output)

Preprocess('./data', [u'기온', u'강수량', u'습도'])