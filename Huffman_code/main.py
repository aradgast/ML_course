import numpy as np

pmf = [1/2, 1/8, 1/8, 1/16, 1/16, 1/16, 1/16]
X = [i for i in range(7)]
seq = np.random.choice(X, 10000, p=pmf)
cnt = [100*(np.count_nonzero(seq == alpha))/len(seq) for alpha in X]
cnt_uncomp = {key:str(val)+'%' for key,val in enumerate(cnt)}
x_bin = {num:bin(num)[2:] for num in X}
x_bin[0] = '00' + x_bin[0]
x_bin[1] = '00' + x_bin[1]
x_bin[2] = '0' + x_bin[2]
x_bin[3] = '0' + x_bin[3]

seq_bin = "".join([x_bin[num] for num in seq])
cnt_bin = [100*(seq_bin.count(b))/len(seq_bin) for b in ['0','1']]
x_huff = dict()
x_huff[0] = '1'
x_huff[1] = '011'
x_huff[2] = '010'
x_huff[3] = '0011'
x_huff[4] = '0010'
x_huff[5] = '0001'
x_huff[6] = '0000'
seq_huff = "".join([x_huff[num] for num in seq])
cnt_huff = [100*(seq_huff.count(b)/len(seq_huff)) for b in ['0', '1']]

print('Huffman encoding into |X| =2 is, \n', x_huff)
print( 'X = ', X)
print('seq = ', seq)
print('count = ', cnt_uncomp)


print('len of bin representation = ', len(seq_bin))
print('% of 0 in the binary seq = ', str(cnt_bin[0])+' %' , '\n% of 1 in the binary seq = ', str(cnt_bin[1])+' %')
print('compress file length is ', len(seq_huff))
print('% of 0 in the huff seq = ', str(cnt_huff[0])+' %', '\n% of 1 in the huff seq = ', str(cnt_huff[1])+' %')
