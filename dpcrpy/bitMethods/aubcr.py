##########################################################################
# Copyright 2023 ******
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################


import numpy as np
from dpcrpy.utils.bitOps import lowbit, lb, kLAN, kLRN
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares
from dpcrpy.utils.optimization import golden
import os
from dpcrpy.data import dataPath
from dpcrpy.framework.dpcrMech import DpcrMech
from threading import Lock
lock = Lock()


def replace(x0, ind, xNew):
    x1 = x0.copy();
    x1[ind] = xNew;
    return x1;


def calcMse(w, z, bkn, lastDbn, usedT):
    m = len(w);
    y1 = 1;
    y2 = 1;
    for i in range(m):
        if w[i] == 0:
            t1 = bkn[m - i - 1];
            t2 = lastDbn[m - i - 1] + 1;
        else:
            t1 = bkn[m - i - 1] * (1 - z) / (1 - z - w[i]);
            t2 = (lastDbn[m - i - 1] + 1) * (1 - z) / (1 - z + lastDbn[m - i - 1] * w[i]);
        t3 = 2 ** (m - i) * t2;
        y1 = y1 + t1 + t3;
        y2 = y2 + t2;
    if z < 1e-8:
        lastErr = y2 + 1;
    else:
        lastErr = (1 - (1 - z) / (y2 * z + 1)) / z;
    blkErr = y1 / (1 - z);
    y = blkErr + lastErr * usedT;
    lastDb = y2 / (1 - z);
    g = np.zeros(m);
    if z == 1:
        for i in range(m):
            g[i] = (bkn[m - i - 1] - ((2 ** (m - i)) * (1 / lastDbn[m - i - 1] + 1))) / (w[i] ** 2);
    else:
        for i in range(m):
            g[i] = bkn[m - i - 1] / (w[i] + z - 1) ** 2 - (2 ** (m - i) + usedT * (1 - z) ** 2 / (y2 * z + 1) ** 2) * (
                    1 + 1 / lastDbn[m - i - 1]) / (w[i] + (1 - z) / lastDbn[m - i - 1]) ** 2;
    return y, g, lastErr, blkErr, lastDb


def getValOfCalcMse(w, z, bkn, lastDbn, usedT):
    m = len(w);
    y1 = 1;
    y2 = 1;
    for i in range(m):
        if w[i] == 0:
            t1 = bkn[m - i - 1];
            t2 = lastDbn[m - i - 1] + 1;
        else:
            t1 = bkn[m - i - 1] * (1 - z) / (1 - z - w[i]);
            t2 = (lastDbn[m - i - 1] + 1) * (1 - z) / (1 - z + lastDbn[m - i - 1] * w[i]);
        t3 = 2 ** (m - i) * t2;
        y1 = y1 + t1 + t3;
        y2 = y2 + t2;
    if z < 1e-8:
        lastErr = y2 + 1;
    else:
        lastErr = (1 - (1 - z) / (y2 * z + 1)) / z;
    blkErr = y1 / (1 - z);
    y = blkErr + lastErr * usedT;
    return y;


def gradOfCalcMse(w, z, bkn, lastDbn, usedT):
    m = len(w);
    y2 = 1;
    for i in range(m):
        if w[i] == 0:
            t2 = lastDbn[m - i - 1] + 1;
        else:
            t2 = (lastDbn[m - i - 1] + 1) * (1 - z) / (1 - z + lastDbn[m - i - 1] * w[i]);
        y2 = y2 + t2;
    g = np.zeros(m);
    if z == 1:
        for i in range(m):
            g[i] = (bkn[m - i - 1] - ((2 ** (m - i)) * (1 / lastDbn[m - i - 1] + 1))) / (w[i] ** 2);
    else:
        for i in range(m):
            g[i] = bkn[m - i - 1] / (w[i] + z - 1) ** 2 - (2 ** (m - i) + usedT * (1 - z) ** 2 / (y2 * z + 1) ** 2) * (
                    1 + 1 / lastDbn[m - i - 1]) / (w[i] + (1 - z) / lastDbn[m - i - 1]) ** 2;
    return g / 100


def calcMseWithZ(z, bkn, lastDbn, usedT, refinedInd=None, x0=None):
    m = len(bkn);
    if m == 0:
        w1 = [];
        fval = calcMse(w1, z, bkn, lastDbn, usedT);
    else:
        if z >= 1:
            w1 = x0;
            fval = float('inf');
        else:
            lz = 1 - z;
            if x0 is None:
                x0 = np.ones(m) * 0.5 * lz;
            if refinedInd is None:
                func1 = lambda x: getValOfCalcMse(x, z, bkn, lastDbn, usedT);
                func2 = lambda x: gradOfCalcMse(x, z, bkn, lastDbn, usedT);
            else:
                xOrg = x0;
                func1 = lambda x: getValOfCalcMse(replace(xOrg, refinedInd, x), z, bkn, lastDbn, usedT);
                func2 = lambda x: gradOfCalcMse(replace(xOrg, refinedInd, x), z, bkn, lastDbn, usedT)[refinedInd];
                x0 = x0[refinedInd];
            res = least_squares(func1, x0, method='trf', bounds=(0, lz), jac=func2, ftol=3e-16, xtol=3e-16, gtol=3e-16)
            w1 = res.x;
            if not refinedInd is None:
                w1 = replace(xOrg, refinedInd, res.x);
            fval = res.fun;
    return fval, w1;


class cofGenerator:
    def __init__(self, m, laterUserd=0):
        data = loadmat(dataPath + os.path.sep + 'metaCofInfo.mat');
        mMax = data['m'] + 1;
        if m > mMax:
            print('阶为%i的规模太大请将阶m设置在%i以下\n' % (m, mMax));
            return;
        self.zn = np.array(data['zn'])
        self.zn = np.reshape(self.zn,(-1,))
        self.WMat = np.array(data['WMat'])
        self.m = m;
        self.laterUserd = laterUserd;
        if self.laterUserd != 0 and self.m > 0:
            fileName = dataPath + os.path.sep + 'aubcrCofs' + os.path.sep + f'cofs_{self.m}_{self.laterUserd}.mat'
            lock.acquire()
            if os.path.exists(fileName):
                # print('Start ==> Loading')
                buffData = loadmat(fileName);
                self.z1 = np.array(buffData['z1'])
                self.z1 = self.z1[0,0]
                self.w1 = np.array(buffData['w1'])
                self.w1 = np.reshape(self.w1,(-1,))
                # print('End <== Loading')
            else:
                # print('Start ==> Doing')
                bkn = np.array((data['blkErr']));
                bkn = np.reshape(bkn, (-1,))[0:(m - 1)]
                lastDbn = np.array((data['lastErrInBlk']));
                lastDbn = np.reshape(lastDbn, (-1,))[0:(m - 1)]
                func1 = lambda z: calcMseWithZ(z, bkn, lastDbn, laterUserd);
                z1 = golden(func1, 0, 1, (1, 0));
                [fval2, w1] = func1(z1);
                for i in range(5, m - 1):
                    [fval2, w1] = calcMseWithZ(z1, bkn, lastDbn, laterUserd, refinedInd=list(range(i, m - 1)), x0=w1);
                self.z1 = z1;
                self.w1 = w1;
                savemat(fileName, {'z1': self.z1,'w1': self.w1})
                # print('End <== Doing')
            lock.release()

    def getCof(self, i):
        if i > self.size() or i <= 0:
            print(self.m)
            print('下标1范围应当在1至%i之间，当前的下标%i不符合范围\n' % (self.size(), i));
            return None;
        if self.laterUserd != 0 and self.m > 0:
            if i == self.size():
                return self.z1;
            else:
                p = 0;
                m = self.m;
                for j in range(2, m + 1):
                    p = p + 2 ** (m - j + 1);
                    if i < p:
                        t = self.__getCof(i, m);
                        y = t * (1 - self.z1 - self.w1[j - 2]) / (1 - self.zn[m - j]);
                        return y;
                    elif i == p:
                        y = self.w1[j - 2];
                        return y;
                y = 1 - self.z1;
                return y;
        if i == self.size():
            if self.m == 0:
                return 1;
            else:
                return 0;
        return self.__getCof(i, self.m);

    def getLeftCof(self, i):
        if i > self.size() or i <= 0:
            print('下标2范围应当在1至%i之间，当前的下标%i不符合范围\n' % (self.size(), i));
            return None;
        if i % 2 == 1:
            return 0;
        return self.getCof(i - 1);

    def getL1Sens(self):
        if self.m == 0:
            return 1;
        s = 1;
        bn = [];
        for i in range(self.m - 1):
            z = self.zn[i];
            s1 = 1;
            for j in range(i):
                wj = self.WMat[i, j] / (1 - z);
                s1 = max(s1, bn[i - 1 - j] * np.sqrt(1 - wj) + np.sqrt(wj));
            s = max(s, np.sqrt(1 - z) * s1 + np.sqrt(z));
            bn.append(s1);
        if self.laterUserd != 0 and self.m > 0:
            s1 = 1;
            for j in range(self.m - 1):
                wj = self.w1[j] / (1 - self.z1);
                s1 = max(s1, bn[self.m - 2 - j] * np.sqrt(1 - wj) + np.sqrt(wj));
            s = np.sqrt(1 - self.z1) * s1 + np.sqrt(self.z1);
        return s;

    def getL2Sens(self):
        return 1;

    def __getCof(self, i, m):
        if m == 1:
            return 1;
        mid = 2 ** (m - 1);
        if i == mid:
            y = self.zn[m - 2];
        elif i < mid:
            p = 0;
            for j in range(2, m):
                p = p + 2 ** (m - j);
                if i < p:
                    t = self.__getCof(i, m - 1);
                    y = t * (1 - self.zn[m - 2] - self.WMat[m - 2, j - 2]) / (1 - self.zn[m - j - 1]);
                    return y;
                elif i == p:
                    y = self.WMat[m - 2, j - 2];
                    return y;
            y = 1 - self.zn[m - 2];
        else:
            y = self.__getCof(i - mid, m - 1);
        return y;

    def size(self):
        return 2 ** self.m;


class AuBCR(DpcrMech): # 原名OCRHB
    def __init__(self, kOrder=0, laterUsed=0, abandonLast=False, noiMech=None, isInit=True):
        self.T = 2 ** kOrder;
        self.abandonLast = abandonLast;
        if abandonLast:
            if kOrder == 0:
                print('阶为0时不可遗弃最后一个发布！')
                return;
            self.T = self.T - 1;
            if laterUsed > 0:
                print('选择遗弃最后一个发布时，建议将laterUserd设为0！')
        self.kOrder = kOrder;
        self.gen = cofGenerator(kOrder, laterUsed)
        self.laterUserd = laterUsed;
        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def init(self):
        self.t = 0;
        self.stk = [];
        self.mse = 0;
        self.sNoi = 0;
        self.buff = [None] * (self.kOrder + 1);
        return self

    def getL1Sens(self):
        return self.gen.getL1Sens();

    def getL2Sens(self):
        return self.gen.getL2Sens();

    def dpRelease(self, x):
        self.t += 1;
        lp = lb(lowbit(self.t))
        cof2 = self.gen.getCof(self.t);
        cof = np.sqrt(cof2);
        leftCof2 = self.gen.getLeftCof(self.t);
        if leftCof2 > 0:
            leftCof = np.sqrt(leftCof2);
            xErr2 = x + self.noiMech.genNoise() / leftCof;
        if self.t % 2 == 0:
            cErr = cof * (self.buff[lp] + x);
            self.buff[lp] = None;
        else:
            cErr = cof * x + self.noiMech.genNoise();
        tmp1 = kLAN(self.t);
        while tmp1 <= self.T:
            j = lb(lowbit(tmp1))
            if self.buff[j] is None:
                cofTmp1 = np.sqrt(self.gen.getCof(tmp1));
                cofTmp1 = max(cofTmp1, 1e-16)
                self.buff[j] = self.noiMech.genNoise() / cofTmp1;
            self.buff[j] += x;
            tmp1 = kLAN(tmp1)
        while len(self.stk) > 0:
            if self.stk[-1][0] <= lp:
                self.stk.pop();
            else:
                break;
        if leftCof2 > 0:
            v2 = xErr2 + self.sNoi - (self.stk[-1][2] if len(self.stk) > 0 else 0);
            D2 = self.mse + 1 / leftCof2 - (self.stk[-1][1] if len(self.stk) > 0 else 0);
            if cof2 > 0:
                v1 = cErr / cof;
                D1 = 1 / cof2;
                alpha = D2 / (D1 + D2);
                self.mse = D1 * D2 / (D1 + D2) + (self.stk[-1][1] if len(self.stk) > 0 else 0);
                self.sNoi = alpha * v1 + (1 - alpha) * v2 + (self.stk[-1][2] if len(self.stk) > 0 else 0);
            else:
                self.mse = D2 + (self.stk[-1][1] if len(self.stk) > 0 else 0);
                self.sNoi = v2 + (self.stk[-1][2] if len(self.stk) > 0 else 0);
        else:
            self.mse = 1 / cof2 + (self.stk[-1][1] if len(self.stk) > 0 else 0);
            self.sNoi = cErr / cof + (self.stk[-1][2] if len(self.stk) > 0 else 0);
        self.stk.append((lp, self.mse, self.sNoi))
        mse = self.noiMech.getMse() * self.mse;
        return (self.sNoi, mse)

class AuBCRComp(DpcrMech):
    def getParamList(self, T, addtionT=0):
        blkList = [];
        bn = bin(T).replace('0b', '')
        bn = list(bn)
        m = len(bn)
        leftT = T + addtionT;
        for i in range(m):
            if bn[i] == '1':
                if addtionT == 0 and kLRN(leftT + 1) == 0:
                    blkList.append((m - i, 0))
                    break;
                q = 2 ** (m - i - 1)
                leftT -= q;
                blkList.append((m - i - 1, leftT + 1))
        return blkList

    def __init__(self, T=1, addtionT=0, noiMech=None, isInit=True):
        self.T=T
        self.addtionT=addtionT
        self.ocrhbBlk=None
        self.blkList = self.getParamList(self.T, self.addtionT)
        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def init(self):
        self.t=0;
        self.blkId=0;
        self.ocrhbBlk=None
        self.lastRs=0
        self.lastMse=0;
        self.cumSum=0;
        self.cumMse=0;
        return self

    def setNoiMech(self, noiMech):
        self.noiMech=noiMech;
        return self;

    def getL1Sens(self):
        if self.blk==None:
            return None;
        return self.blk.getL1Sens();

    def getL2Sens(self):
        if self.blk==None:
            return None;
        return self.blk.getL2Sens();

    def dpRelease(self, x):
        if self.ocrhbBlk is None or self.ocrhbBlk.size()==self.j:
            self.ocrhbBlk=AuBCR(kOrder=self.blkList[self.blkId][0], laterUsed=self.blkList[self.blkId][1], abandonLast=(self.blkList[self.blkId][1] == 0));
            self.ocrhbBlk.setNoiMech(self.noiMech)
            self.j=0;
            self.cumSum+=self.lastRs;
            self.cumMse+=self.lastMse;
            self.blkId+=1;
        self.ocrhbBlk.setNoiMech(self.noiMech)
        (self.lastRs,self.lastMse)=self.ocrhbBlk.dpRelease(x);
        res=self.cumSum+self.lastRs
        mse=self.cumMse+self.lastMse;
        self.t+=1;
        self.j+=1;
        return (res,mse)



