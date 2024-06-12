import math
import numpy as np

def array_response(a1,a2,N,antenna_array):

    y = np.zeros((N,1),dtype = complex)
    if antenna_array =='USPA':
        for m in range(int(math.sqrt(N))):
            for n in range(int(math.sqrt(N))):
                y[m*(int(math.sqrt(N)))+n] = np.exp( 1j* math.pi* ( m*math.sin(a1)*math.cos(a2) + n*math.cos(a2) ) )
    elif antenna_array == 'ULA':
        for n in range(N):
            y[n] = np.exp( 1j* math.pi* ( n*math.sin(a1) ) )
   
    y = y/math.sqrt(N)
    return y

if __name__ == '__main__':
    y = array_response(2,3,4,'USPA')
    print(y) 