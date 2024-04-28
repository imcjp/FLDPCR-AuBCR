##########################################################################
# Copyright 2024 Cai Jianping
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
from dpcrpy.utils.bitOps import lowbit, lb
import math
from dpcrpy.framework.dpcrMech import DpcrMech

class Honaker(DpcrMech):
    def __init__(self, kOrder=1, noiMech=None, isInit=True):
        self.T = 2 ** kOrder;
        self.kOrder = kOrder;
        (self.wn, _) = self.genWn(self.kOrder)
        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def init(self):
        self.t = 0;
        self.stk = [];
        self.buff = [0] * (self.kOrder + 1);
        return self

    def getL1Sens(self):
        return self.kOrder + 1;

    def getL2Sens(self):
        return np.sqrt(self.kOrder + 1);

    def genWn(self, k):
        m = k + 1;
        sigman = np.ones((m,))
        wn = np.ones((m,))
        for i in range(1, m):
            wn[i] = (sigman[i] ** -2) / (sigman[i] ** -2 + 1.0 / (2 * sigman[i - 1] ** 2))
            sigman[i] = sigman[i] * math.sqrt(wn[i])
        return (wn, sigman)

    def dpRelease(self, x):
        self.t += 1;
        lp = lb(lowbit(self.t))
        for i in range(len(self.buff)):
            self.buff[i] += x;
        noiX = self.buff[0] + self.noiMech.genNoise();
        mseX = self.noiMech.getMse();
        self.stk.append((noiX, mseX))
        self.buff[0] = 0;
        for i in range(1, 1 + lp):
            v1, mse1 = self.stk[-1]
            self.stk.pop()
            v2, mse2 = self.stk[-1]
            self.stk.pop()
            noiV = self.buff[i] + self.noiMech.genNoise()
            v3 = (1 - self.wn[i]) * (v1 + v2) + self.wn[i] * noiV;
            mse3 = (1 - self.wn[i]) ** 2 * (mse1 + mse2) + self.wn[i] ** 2 * self.noiMech.getMse();
            self.stk.append((v3, mse3))
            self.buff[i] = 0
        sNoi = 0
        mse = 0
        for i in range(len(self.stk)):
            sNoi += self.stk[i][0];
            mse += self.stk[i][1];
        return (sNoi, mse)
