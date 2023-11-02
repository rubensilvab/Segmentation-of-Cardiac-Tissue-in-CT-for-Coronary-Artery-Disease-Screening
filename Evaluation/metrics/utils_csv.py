import csv


def readCsv(csvpath, header=True):
    with open(csvpath) as f:
        reader = csv.reader(f)
        lines = list(reader)

    if header:
        return lines[0], lines[1:]
    else:
        return lines


def writeCsv(csvpath, lines, header=None):
    with open(csvpath, mode='w', newline='') as f:
        writer = csv.writer(f)

        if not header == None:
            writer.writerow(header)
        writer.writerows(lines)


def readCsvDataLists(csvpath, targetstr=['filename', 'class']):
    header, lines = readCsv(csvpath)
    if isinstance(targetstr, str):
        targetstr = [targetstr]
    targetind = [header.index(t) for t in targetstr]
    datalists = [[] for _ in targetstr]
    for l in lines:
        for i, t in enumerate(targetind):
            datalists[i].append(l[t])

    if len(targetstr) == 1:
        return datalists[0]
    else:
        return datalists
