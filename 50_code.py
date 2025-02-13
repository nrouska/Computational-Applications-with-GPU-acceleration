import threading
import tkinter as tk
import numpy as np
from numba import vectorize, cuda
# to measure exec time
from timeit import default_timer as timer
import time
from threading import Thread

def Matrix(d):#ΝΧΝ με ομοιόμορφα κατανεμημένα στοιχεία απο 50 εως 100 
  matrix =np.ndarray(shape=(d,d), dtype=np.float32)
  for i in range (d):
    for j in range (d):
      matrix[i][j] = np.random.uniform(50,100)
  return matrix 

def resetclock():
  tinac.configure(text ='')
  tiCPU.configure(text ='')
  tiGPU.configure(text ='')

def thread():
  n1 = threading.Thread(target=main)
  n1.start()


def main():
  resetclock()
  N = int(NNget.get())
  global M
  M = int(MMget.get())
  
  
  A = Matrix(N)
  print("PINAKAS A")
  print(A)

  
  B = Matrix(N)
  print("PINAKAS B")
  print(B)  


  def ArrayMul_CPU(A,B):
    F = A*B
    for i in range (M-1):
      F = F + A * B
    return F
  
  @vectorize(['float32(float32, float32)'],target = 'cpu') 
  def VectorArrayMul_CPU(A,B):
    F = A*B
    for i in range (M-1):
      F = F + A * B 
    return F
    
# Simple python Mul arrays on CPU
  start = timer()
  C = ArrayMul_CPU(A, B)
  mul_array_cpu_time = timer() - start
  print(C)
  
  print("Multiplicatipon of array took %f seconds in CPU" % mul_array_cpu_time)
  print("C[5][5] = " + str(C[5,5]))
  tinac.configure(text ='{:.8f}'.format (mul_array_cpu_time))

# Vectorized Mul arrays on CPU
  start = timer()
  D = VectorArrayMul_CPU(A, B)
  vector_array_mul_cpu_time = timer() - start
  print(D)
  print("Vectorized Multipplication of array took %f seconds in CPU" % vector_array_mul_cpu_time)
  print("D[5][5] = " + str(D[5,5]))
  tiCPU.configure(text ='{:.8f}'.format (vector_array_mul_cpu_time))

 
  @vectorize(['float32(float32, float32)'], target = 'cuda')
  def VectorArrayMul_GPU(A,B):
    F = A*B
    for i in range (M-1):
      F = F + A * B
    return F

#Vectorized Mul arrays on GPU
  start = timer()
  E = VectorArrayMul_GPU(A, B)
  vector_array_mul_gpu_time = timer() - start
  print("Vectorized Multiplication of array took %f seconds in GPU" % vector_array_mul_gpu_time)
  print("E[5][5] = " + str(E[5,5]))
  tiGPU.configure(text ='{:.8f}'.format (vector_array_mul_gpu_time))
    
  
root = tk.Tk()
root.geometry("600x300")
root.resizable(False, False)
root.title("Yπολογιστικές εφαρμογές με επιτάχυνση GPU με Python")
root.configure(bg='#B4C6A6')

NN = tk.IntVar()
MM = tk.IntVar()

tk.Label(root, text="Διάσταση πίνακα",bg='#B4C6A6',fg='#30475E').grid(row=0,column=0)
tk.Label(root, text="Αριθμός πολλαπλασιασμών των στοιχείων",bg='#B4C6A6',fg='#30475E').grid(row=1,column=0)

NNget =tk.Entry(root,textvariable=NN,bg='#B4C6A6',fg='#222831')
NNget.grid(row=0,column=1)

MMget= tk.Entry(root,textvariable=MM,bg='#B4C6A6',fg='#222831')
MMget.grid(row =1,column=1)

start = tk.Button(root,text = 'start',command=thread ,bg ='#A3DA8D')
start.grid(row=2,column=2)

tnac= tk.Label(root,text='time without acceleration:',bg='#B4C6A6',fg='#30475E').grid(row=3)

tinac=tk.Label(root, text ='',bg='#B4C6A6',fg='#222831')
tinac.grid(row=3,column=1)


tCPU =tk.Label(root,text = 'time with cpu acceleration:',bg='#B4C6A6',fg='#30475E').grid(row=4)
tiCPU = tk.Label(root,text='',bg='#B4C6A6',fg='#222831' )
tiCPU.grid(row=4,column=1)

tGPU = tk.Label(root,text = 'time with GPU acceleration:',bg='#B4C6A6',fg='#30475E').grid(row=5)
tiGPU = tk.Label(root,text = '',bg='#B4C6A6',fg='#222831')
tiGPU.grid(row=5,column=1)


root.mainloop()




