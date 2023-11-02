import os
import nrrd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, ndimage
import multiprocessing
from functools import partial

import utils_csv


def calcDiceJaccard(mask1, mask2):
    union = ((mask1 + mask2) > 0).sum()
    inter = ((mask1 + mask2) == 2).sum()

    return inter / union, 2 * inter / (mask1.sum() + mask2.sum())


def calcdistMetrics_MemErrAux(c1, edge2, pixelspacing):
    edge2 = edge2 == 1
    d1 = np.zeros(c1.shape[0])
    dt = 0
    for i1, c in enumerate(c1):
        dtest = -1
        d = max(0, (dt - 10) - (dt - 10) % 10)
        while dtest < 0 or dtest >= d:
            d += 10
            c = np.array(c)
            cmin = np.amax(np.concatenate((c[:, None] - d, np.zeros(np.array(c[:, None].shape))), axis=1),
                           axis=1).astype(int)
            cmax = np.amin(np.concatenate((c[:, None] + d, np.array(edge2.shape)[:, None] - 1), axis=1), axis=1).astype(
                int)
            c2t = np.array(
                np.where(edge2[cmin[0]:cmax[0] + 1, cmin[1]:cmax[1] + 1, cmin[2]:cmax[2] + 1])).transpose()
            if c2t.size > 0:
                c2t += cmin
                dt = (c2t - c) ** 2
                dtest = np.sum(dt, axis=1)
                dtest = np.amin(np.sqrt(dtest))
                dt = np.sum(dt * pixelspacing, axis=1)
                dt = np.amin(np.sqrt(dt))
            else:
                dtest = d
        d1[i1] = dt
    return d1


def calcdistMetrics(edge1, edge2, pixelspacing):
    c1 = np.array(np.where(edge1 == 1)).transpose()
    c2 = np.array(np.where(edge2 == 1)).transpose()
    """try:
        d = cdist(c1, c2)
        d1 = np.amin(d, axis=0)
        d2 = np.amin(d, axis=1)
    except:"""
    print('Iterating Surface 1...')
    d1 = calcdistMetrics_MemErrAux(c1, edge2, pixelspacing)
    print('Iterating Surface 2...')
    d2 = calcdistMetrics_MemErrAux(c2, edge1, pixelspacing)

    if max(d1) > max(d2):
        ind = np.argmax(d1)
        coord = c1[ind]
    else:
        ind = np.argmax(d2)
        coord = c2[ind]

    return max(max(d1), max(d2)), sum([sum(d1) / len(d1), sum(d2) / len(d2)]) / 2, coord


def calcdistMet(reader, ref, fpath, get_maxindex=False):
    print('entrou calcdist')
    struct = np.tile(ndimage.generate_binary_structure(2, 1), (3, 1, 1))
    struct[0, :, :] = False
    struct[0, 1, 1] = True
    struct[2, :, :] = False
    struct[2, 1, 1] = True

    dicomSource = os.path.join(fpath, 'DICOM')
    print(dicomSource)
    refFlag, readerFlag = False, False
    for fsegm in os.listdir(fpath):
        if f'_{reader}.nrrd' in fsegm:
            readerFlag = True
        if f'_{ref}.nrrd' in fsegm:
            refFlag = True
    print(refFlag,readerFlag)
    if refFlag and readerFlag:
        print(dicomSource)
        for fsegm in os.listdir(fpath):
            if f'_{reader}.nrrd' in fsegm:
                print('reader')
                pdMask, pdEdge, pixelspacing = load_segm(os.path.join(fpath, fsegm), struct)
            if f'_{ref}.nrrd' in fsegm:
                print('ref')
                gtMask, gtEdge, pixelspacing = load_segm(os.path.join(fpath, fsegm), struct)

        [dice, jaccard] = calcDiceJaccard(pdMask, gtMask) # Não está dice e jaccard trocado?
        [hd, mad, coord] = calcdistMetrics(pdEdge, gtEdge, pixelspacing)
        l = [os.path.split(fpath)[-1], dice, jaccard, hd, mad]
        print(l)
    else:
        l = None
    print(l)
    if get_maxindex:
        return l, coord * pixelspacing
    else:
        return l


def load_segm(fpath, struct, isotropic=True):
    mask, segmheader = nrrd.read(fpath)
    mask = np.transpose(mask, (2, 1, 0))
    mask[mask > 0] = 1.
    pixelspacing = np.array((segmheader['space directions'][2][2], segmheader['space directions'][1][1],
                             segmheader['space directions'][0][0]))
    #print(pixelspacing)
    if isotropic:
        pxspacing = np.min(pixelspacing)
        mask = ndimage.zoom(mask.astype(np.float64), pixelspacing / pxspacing, order=1)
        pixelspacing = (pxspacing, pxspacing, pxspacing)
        mask[mask >= .5] = 1.
        mask[mask < .5] = 0.
        mask = mask.astype(np.int16)

    if struct is None:
        return mask, None, pixelspacing

    erode = ndimage.binary_erosion(mask, struct)
    edge = mask ^ erode

    return mask, edge, pixelspacing

from functools import partial

def main(fpath, ref, reader, par_proc=True):
    csv_path=fpath+f'_distmet_{ref}_{reader}.csv'
    if os.path.isfile(csv_path):
        header, lines = utils_csv.readCsv(csv_path)
    else:
        header = ['patient', 'jaccard','dice', 'hd', 'mad']
        lines = []
    flist = [l[0] for l in lines]

    if par_proc:
        listdir = [os.path.join(fpath, f) for f in os.listdir(fpath) if not f in flist]
        print(f'Missing {len(listdir)} patients...')
        pool = multiprocessing.Pool(processes=20)
        print('entrou')
        print(listdir)
        nlines = pool.map(partial(calcdistMet, reader, ref), listdir)
        print('entrou2')
        pool.close()
        pool.join()
        lines.extend([l for l in nlines if not l is None])
    else:
        for f in os.listdir(fpath):
          
            if f in flist:
                continue
            
            l = calcdistMet(reader, ref, os.path.join(fpath, f))
            if not l is None:
                lines.append(l)
            print(l)   

    utils_csv.writeCsv(csv_path, lines, header=header)


if __name__ == "__main__":
    fpath = 'D:\\RubenSilva\\eat_segm\\intra_inter\\NRRD'
    ref = 'EATmanual'

    #for reader in ['CarolinaInter', 'Fabio', 'Siemens']:
    for reader in ['EATcarol','EATfabio']:#,'3D2DNet+pp+conv','3D2DNet+pp+fill2d']:
        main(fpath, ref, reader, par_proc=True)