import numpy as np


def read_data(filename):
    n_row = 0
    with open(filename, 'r') as f:
        fake = 0
        for line in f:
            if len(line.strip()) == 0:
                fake += 1
            n_row += 1
        f.seek(0)
        data = np.zeros((n_row - fake, 39972), dtype=np.int32)
        id_row = 0
        for line in f:
            try:
                data[id_row, [int(i) for i in line.replace('\n', '').rsplit(' ')]] = 1.
            except ValueError:
                id_row -= 1
                assert len(line.strip()) == 0
            id_row += 1
            print('\r\x1b[6;30;42mProgress:\x1b[0;1m [{:.0f}]%\x1b[0m'.format(id_row/n_row * 100), end='')
        print("\rDone read data from: '{}'".format(filename))
    return data


if __name__ == '__main__':
    data = read_data('./on_bits/A2A_act_onbits')
    print(data)
