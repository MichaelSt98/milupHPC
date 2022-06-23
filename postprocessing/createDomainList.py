
DIM = 3
maxLevel = 21
curveType = 0

KEY_MAX = (1 << 64) - 1

#ranges = [0, (4 << 60) + (1 << 54), (6 << 60) + (2 << 54), KEY_MAX]
ranges = [0, 1152921504606845000, 1646833077180417398, 2799754581787262398, KEY_MAX]



def key2char(key):
    path = []
    for i in range(maxLevel):
        path.append(key >> (maxLevel*DIM - DIM*(i+1)) & 7)
    #print(path)
    return path

def key2proc(key):
    for proc in range(len(ranges)):
        if ranges[proc] <= key < ranges[proc + 1]:
            return proc


def isDomainListNode(key, maxLevel, level, type):

    p1 = 0
    p2 = 0
    #if type == 0:
    p1 = key2proc(key)
    #print("p2 key: {}".format((key | ~(KEY_MAX << DIM * (maxLevel - level)))& 0xffffffffffffffff))
    #p2 = key2proc((key | ~(KEY_MAX << DIM * (maxLevel - level))) & 0xffffffffffffffff)
    p2 = key2proc((key | (KEY_MAX >> DIM * level + 1)) & 0xffffffffffffffff)
    #else:
    #    ...

    if p1 != p2:
        #print("p1 = {} != p2 = {}".format(p1, p2))
        return True
    else:
        #print("p1 = {} == p2 = {}".format(p1, p2))
        return False

def createDomainList(domainListKeys, domainListLevels):

    shiftValue = 1
    toShift = 63
    keyMax = (shiftValue << toShift) - 1  # 1 << 63 not working!

    key2test = 0
    level = 1

    while key2test <= keyMax:

        if isDomainListNode(key2test & (KEY_MAX << (DIM * (maxLevel - level + 1))), maxLevel, level-1, curveType):
            domainListKeys.append(key2test)
            domainListLevels.append(level)
            print("append key: {} (level: {}, path: {})".format(key2test, level, key2char(key2test)))
            if isDomainListNode(key2test, maxLevel, level, curveType):
                level += 1
            else:
                key2test = key2test + (1 << DIM * (maxLevel - level))
                # new
                while key2test >> (DIM * (maxLevel - level)) & 7 == 0:
                    level -= 1
                level -= 1
                # end: new
        else:
            level -= 1


if __name__ == '__main__':

    domainListKeys = []
    domainListLevels = []

    createDomainList(domainListKeys, domainListLevels)

    counter = [0 for i in range(maxLevel + 1)]
    for i in range(len(domainListLevels)):
        counter[domainListLevels[i]] += 1

    for i in range(maxLevel):
        print("# level {}: {}".format(i, counter[i]))

    print("# domain list: {}".format(len(domainListLevels)))

    for i in range(len(ranges)):
        print("range[{}] = {} = {}".format(i, ranges[i], key2char(ranges[i])))

    #(2)[INFO ] range[2] = #|1|3|3|3|2|5|6|3|7|2|5|3|1|5|4|7|4|2|5|6|6|
    #(2)[INFO ] range[3] = #|2|3|3|3|2|5|6|3|7|2|5|3|1|5|4|7|3|6|6|7|6|
    #(2)[INFO ] range[4] = #|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|7|

    if counter[1] != 8:
        for i in range(len(domainListLevels)):
            if domainListLevels[i] == 1:
                print("level: 1: key = {} = {}".format(domainListKeys[i], key2char(domainListKeys[i])))

    print(key2char((1<<63) - 1))