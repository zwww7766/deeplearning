import  UnitLine

f = lambda x:x

def get_train_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_line_unit():
    lu = UnitLine.LineUnit(1,f)

    input_vecs, labels = get_train_dataset()
    lu.train(input_vecs, labels, 10 ,0.01)
    return lu


if __name__ =='__main__':
    lu_ = train_line_unit()
    print lu_
    print 'Work 3.4 years, monthly salary = %.2f' % lu_.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % lu_.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % lu_.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % lu_.predict([6.3])