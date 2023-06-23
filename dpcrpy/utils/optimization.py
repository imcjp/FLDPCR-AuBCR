import numpy as np

def golden(f,a,b,isCloseInv=(1,1)):
    a0=a;
    b0=b;
    P2=(np.sqrt(5)-1)/2;
    P1=1-P2;
    eps2=(b-a)*np.spacing(1);
    t1=a+P1*(b-a);
    t2=a+P2*(b-a);
    y1=f(t1);
    isTuple=False;
    if isinstance(y1,tuple):
        y1=y1[0];
        isTuple=True;
    if isTuple:
        y2=f(t2)[0];
        while True:
            if y1<=y2:
                b=t2;
                if b-a<=eps2:
                    z=(b+a)/2;
                    if isCloseInv[0]:
                        y3=f(a0)[0];
                        if y3<=y1:
                            z=a0;
                            y1=y3;
                    if isCloseInv[1] and f(b0)[0]<=y1:
                        z=b0;
                    return z;
                else:
                    t2=t1;
                    y2=y1;
                    t1=a+P1*(b-a);
                    y1=f(t1)[0];
            else:
                a=t1;
                if b-a<=eps2:
                    z=(b+a)/2;
                    if isCloseInv[0]:
                        y3=f(a0)[0];
                        if y3<=y2:
                            z=a0;
                            y2=y3;
                    if isCloseInv[1] and f(b0)[0]<=y2:
                        z=b0;
                    return z;
                else:
                    t1=t2;
                    y1=y2;
                    t2=a+P2*(b-a);
                    y2=f(t2)[0];
    else:
        y2=f(t2);
        while True:
            if y1<=y2:
                b=t2;
                if b-a<=eps2:
                    z=(b+a)/2;
                    y3=f(a0);
                    if y3<=y1:
                        z=a0;
                        y1=y3;
                    if f(b0)<=y1:
                        z=b0;
                    return z;
                else:
                    t2=t1;
                    y2=y1;
                    t1=a+P1*(b-a);
                    y1=f(t1);
            else:
                a=t1;
                if b-a<=eps2:
                    z=(b+a)/2;
                    y3=f(a0);
                    if y3<=y2:
                        z=a0;
                        y2=y3;
                    if f(b0)<=y2:
                        z=b0;
                    return z;
                else:
                    t1=t2;
                    y1=y2;
                    t2=a+P2*(b-a);
                    y2=f(t2);
