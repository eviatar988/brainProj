
import os.path as op
from mne.datasets import sample
from patients_matrix import PatientsMatrix

bids_path = r"C:\Users\eyala\Documents\GitHub\brainProj\ds003688"
dataset = "ds003688"
subject = "07"
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
run = '1'



def main():
    print(sample.data_path())

    bids_root = op.join(op.dirname(sample.data_path()), dataset)

    #p1 = PatientsMatrix(bids_root)
    #p1.save_matrix_to_file()


    """loaded_data = np.load('coherence_matrixs.npz', allow_pickle=True)
    rest_matrixs = loaded_data['arr_film']
    plt.imshow(rest_matrixs[1], cmap='viridis')
    plt.colorbar()
    plt.show()"""


if __name__ == '__main__':
    main()

