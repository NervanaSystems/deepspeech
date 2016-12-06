import numpy as np
import random

class DataLoader:
    def __init__(self, mbsz=128, min_len=20, max_len=30, num_classes=29):
        self.mbsz = mbsz
        self.min_len = min_len
        self.max_len = max_len
        self.num_classes = num_classes

    def sample(self):
        inputs = []
        input_lens = []
        outputs = []
        output_lens = []
        for i in xrange(self.mbsz):
            length = random.randint(self.min_len, self.max_len)
            input_lens.append(length)
            input = [random.randint(1, self.num_classes-1) for j in xrange(length)]
            #output = input[:] # identity output
            output = input[::4] # every 4th input is output
            """
            # for acronym output
            output = []
            flag = True
            for j in xrange(len(input)):
                if input[j] == 1:
                    flag = True
                elif flag == True:
                    flag = False
                    output.append(input[j])
            """
            output_lens.append(len(output))
            inputs.append(input)
            outputs.append(output)

        input_arr = np.zeros((self.mbsz, self.max_len, self.num_classes))
        for i in xrange(self.mbsz):
            for j in xrange(len(inputs[i])):
                input_arr[i, j, inputs[i][j]] = 1.0
        label_arr = np.zeros((sum(output_lens)), dtype=np.int32)
        pos = 0
        for i in xrange(self.mbsz):
            label_arr[pos:pos+output_lens[i]] = outputs[i]
            pos += output_lens[i]

        return input_arr, np.array(input_lens, dtype=np.int32), label_arr, np.array(output_lens, dtype=np.int32)


if __name__ == '__main__':
    dl = DataLoader()
    ret = dl.sample()
    print ret[0].shape
