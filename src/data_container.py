import csv
import numpy as np

class DataContainer:
    def __init__(self, datafile, train_ratio=0.8):
        
        self.t_ratio = train_ratio
        self.X = None
        self.Y = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        
        self.N = None # number of samples
        self.train_index = 0  # index of next unused samples
        self.test_index = 0  # index of next unused samples
        
        # Load input/output data from file
        self.load_data(datafile)

        # Generate train/test set split
        self.train_test_split()

    def reset(self):
        """Reset the counter so we can start reusing data"""
        self.train_index = 0
        self.test_index = 0

    def load_data(self, filename):
        """
        Train/test data is stored in CSV files with the following format:
            - The first few lines starting with HDR are headers
            - Subsequent lines consist of [x,y,xdot,ydot] measurements

        Return input-output data in two numpy arrays of the following format:
            Input:  [[xdot0, ydot0], ...]
            Output: [[x1-x0, y1-y0, xdot1, ydot1], ...]
        """
        raw_data = []
        myinput = []
        myoutput = []

        # load raw data from CSV
        with open(filename,'r') as incsv:
            reader = csv.reader(incsv)

            for row in reader:
                if row[0][0] == "H":
                    # ignore headers
                    pass
                else:
                    raw_data.append(row)

        # Create a input/output lists
        for i in range(1, len(raw_data)-1):   # traverse in order, from 1 to N-1
            xl = float(raw_data[i-1][0])  # last positions
            yl = float(raw_data[i-1][1])

            x0 = float(raw_data[i][0])
            y0 = float(raw_data[i][1])
            xdot0 = float(raw_data[i][2])
            ydot0 = float(raw_data[i][3])
            
            x1 = float(raw_data[i+1][0])
            y1 = float(raw_data[i+1][1])
            xdot1 = float(raw_data[i+1][2])
            ydot1 = float(raw_data[i+1][3])

            myinput.append([x0-xl, y0-yl, xdot0, ydot0])
            myoutput.append([x1-x0, y1-y0, xdot1, ydot1])

        # Convert to np arrays and store
        self.X = np.asarray(myinput)
        self.Y = np.asarray(myoutput)
        self.N = len(self.X)

    def train_test_split(self):
        """
        Divide the given dataset into a training set and a testing
        set, with the given ratio. I.e. by default 80 percent goes
        to training and 20 percent to testing. 
        """
        
        self.N = len(self.X)
        cutoff = int(self.N*self.t_ratio)

        self.train_X = self.X[0:cutoff]
        self.train_Y = self.Y[0:cutoff]

        self.test_X = self.X[cutoff:]
        self.test_Y = self.Y[cutoff:]
    
    def get_next_batch(self, num_steps, batch_size, dataset="Train", add_noise=False):
        """
        Give a set of training data and a starting index, generate a batch as follows:

        x (input): np.array
            current velocity
        y (output): np.array
            Subsequent change in position and subsequent velocity
        """
        x = np.zeros((num_steps, batch_size, 4))
        y = np.zeros((num_steps, batch_size, 4))

        # Determine which dataset to use
        if dataset == "Train":
            input_data = self.train_X
            output_data = self.train_Y
            index = self.train_index
        elif dataset == "Test":
            input_data = self.test_X
            output_data = self.test_Y
            index = self.test_index
        else:
            raise AssertionError("invalid dataset '%s'" % dataset)
     
        # Ensure there is enough data
        max_index = index + batch_size*num_steps
        if max_index > len(input_data):
            raise AssertionError("Not enough data! You asked for %s new data points, but I only have %s left" % (batch_size*num_steps, len(input_data)-index))

        for i in range(batch_size):
            deltax0 = input_data[index:index+num_steps][:,0]
            deltay0 = input_data[index:index+num_steps][:,1]
            xdot0 = input_data[index:index+num_steps][:,2]
            ydot0 = input_data[index:index+num_steps][:,3]

            deltax = output_data[index:index+num_steps][:,0]
            deltay = output_data[index:index+num_steps][:,1]
            xdot1 = output_data[index:index+num_steps][:,2]
            ydot1 = output_data[index:index+num_steps][:,3]

            if add_noise:
                # add a bit of Gaussian noise to the input and output
                sigma_position = 0.01
                sigma_velocity = 0.01

                x[:, i, 0] = deltax0 + np.random.normal(0,sigma_position, num_steps)
                x[:, i, 1] = deltay0 + np.random.normal(0,sigma_position, num_steps)
                x[:, i, 2] = xdot0 + np.random.normal(0,sigma_velocity, num_steps)
                x[:, i, 3] = ydot0 + np.random.normal(0,sigma_velocity, num_steps)

                y[:, i, 0] = deltax# + np.random.normal(0,sigma_position, num_steps)
                y[:, i, 1] = deltay# + np.random.normal(0,sigma_position, num_steps)
                y[:, i, 2] = xdot1# + np.random.normal(0,sigma_velocity, num_steps)
                y[:, i, 3] = ydot1# + np.random.normal(0,sigma_velocity, num_steps)
            else:
                x[:, i, 0] = deltax0
                x[:, i, 1] = deltay0
                x[:, i, 2] = xdot0
                x[:, i, 3] = ydot0

                y[:, i, 0] = deltax
                y[:, i, 1] = deltay
                y[:, i, 2] = xdot1
                y[:, i, 3] = ydot1

            index += num_steps

        # Update the index for next time
        if dataset == "Train":
            self.train_index = index
        else:
            self.test_index = index

        return x, y
