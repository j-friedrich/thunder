import os
import glob
import json
import types
from numpy import ndarray, frombuffer, dtype, int16, float, array, sum, mean, std, size
from thunder.rdds.data import Data


class Series(Data):
    """A series backed by an RDD of (tuple,array) pairs
    where the tuple is an identifier for each record,
    and the value is an array indexed by a
    common, fixed list (e.g. a time series)"""

    def __init__(self, rdd, index=None):
        super(Series, self).__init__(rdd)
        if index is not None:
            self.index = index
        else:
            record = self.rdd.first()
            self.index = range(0, len(record[1]))

    @staticmethod
    def _check_type(record):
        key = record[0]
        value = record[1]
        if not isinstance(key, tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(value, ndarray):
            raise Exception('Values must be ndarrays')
        else:
            if value.ndim != 1:
                raise Exception('Values must be 1d arrays')

    def between(self, left, right, inclusive=True):
        if inclusive:
            crit = lambda x: left <= x <= right
        else:
            crit = lambda x: left < x < right
        return self.select(crit)

    def select(self, crit):
        """Select subset of values that match a given index criterion"""
        index = self.index

        if not isinstance(crit, types.FunctionType):
            if isinstance(crit, list):
                critlist = set(crit)
            else:
                critlist = {crit}
            crit = lambda x: x in critlist

        newindex = [i for i in index if crit(i)]
        if len(newindex) == 0:
            raise Exception("No indices found matching criterion")
        if newindex == index:
            return self

        rdd = self.rdd.mapValues(lambda x: array([y[0] for y in zip(array(x), index) if crit(y[1])]))

        return Series(rdd, index=newindex)

    def seriesSum(self):
        return self.seriesStat('sum')

    def seriesMean(self):
        return self.seriesStat('mean')

    def seriesStdev(self):
        return self.seriesStat('stdev')

    def seriesStat(self, stat):
        STATS = {
            'sum': sum,
            'mean': mean,
            'stdev': std,
            'max': max,
            'min': min,
            'count': size
        }
        func = STATS[stat]
        rdd = self.rdd.mapValues(lambda x: func(x))
        return Series(rdd, index=[stat])

    def seriesStats(self):
        rdd = self.rdd.mapValues(lambda x: array([x.size, mean(x), std(x), max(x), min(x)]))
        return Series(rdd, index=['size', 'mean', 'std', 'max', 'min'])


class SeriesLoader(object):

    def __init__(self, nkeys, nvalues, keytype='int16', valuetype='int16', minPartitions=None):
        self.nkeys = nkeys
        self.nvalues = nvalues
        self.keytype = keytype
        self.valuetype = valuetype
        self.minPartitions = minPartitions

    def fromText(self, datafile, sc):

        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.txt')))
            datafile = ''.join([files[x] + ',' for x in range(0, len(files))])[0:-1]

        def parse(line, nkeys):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys:])
            keys = tuple(int(x) for x in vec[:nkeys])
            return keys, ts

        lines = sc.textFile(datafile, self.minPartitions)
        nkeys = self.nkeys
        data = lines.map(lambda x: parse(x, nkeys))

        return Series(data)

    def fromBinary(self, datafile, sc):

        if os.path.isdir(datafile):
            datafile = os.path.join(datafile, '*.bin')

        recordsize = self.nvalues + self.nkeys
        recordsize *= dtype(FORMATS[self.valuetype]).itemsize

        lines = sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                              'org.apache.hadoop.io.LongWritable',
                                              'org.apache.hadoop.io.BytesWritable',
                                              conf={'recordLength': str(recordsize)})

        valuetype = self.valuetype
        nkeys = self.nkeys
        parsed = lines.map(lambda (k, v): (k, frombuffer(v, valuetype)))
        data = parsed.map(lambda (k, v): (tuple(v[0:nkeys].astype(int)), v[nkeys:].astype(float)))

        return Series(data)

    @staticmethod
    def loadConf(datafile):
        if os.path.isdir(datafile):
            basepath = datafile
        else:
            basepath = os.path.dirname(datafile)
        try:
            f = open(os.path.join(basepath, 'conf.json'), 'r')
            params = json.load(f)
        except IOError:
            params = None
        return params


FORMATS = {
    'int16': int16,
    'float': float
}