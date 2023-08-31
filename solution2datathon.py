import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras import Model

# Load data
df = pd.read_csv("flow_simulation_data.csv")

data_cols = list(list(df)[i] for i in [8,9,10])

# Formulate the Input and Output
well_locs = pd.read_csv("well_loc.txt", sep="\t", header=None)
well_locs.columns = ['x','y']
well_locs = well_locs.to_numpy()

def Euc_dist(loc1,loc2):
    dist = np.sqrt((loc1[0]-loc2[0])**2+(loc1[1]-loc2[1])**2)
    return dist

def find_the_nearby(well_target,well_locs):
    # Euclidean distance
    well_digit = []
    loc1 = well_target
    for i in range(well_locs.shape[0]):
        loc2 = well_locs[i,:]
        dist = Euc_dist(loc1,loc2)
        well_digit.append([i+1,dist])
    well_digit = np.array(well_digit)
    well_digit = well_digit[well_digit[:, 1].argsort()]
    well_digit_selectd = well_digit[0:5,0]
    return well_digit_selectd


def find_dynamic_input(well_num, well_locs, df_model, t_steps, well_amt):
    well_target = well_locs[well_num, :]
    well_digit = find_the_nearby(well_target, well_locs)
    s_inputs = []
    d_inputs = []
    outputs = []

    for i in range(well_amt):
        df_well = df_model.iloc[i * 365:i * 365 + 365]
        df_well = df_well.reset_index()
        # for the output and static data
        if df_well['Well'][1] == int(well_digit[0]):
            d_output = df_well['Production Rate'].to_numpy()
            data_cols = list(list(df)[ii] for ii in [8, 9, 10])
            s_input = df_well[data_cols].to_numpy()
            outputs.append(d_output)
            s_inputs.append(s_input)
        # fot the input surrounding 4 injectors
        if df_well['Well'][1] == int(well_digit[1]) or df_well['Well'][1] == int(well_digit[2]) or df_well['Well'][
            1] == int(well_digit[3]) or df_well['Well'][1] == int(well_digit[4]):
            d_input = df_well['Bottom Hole Pressure'].to_numpy()
            d_inputs.append(d_input)

    outputs = np.array(outputs)
    outputs = outputs.reshape(-1, 1)
    d_inputs = np.array(d_inputs)
    d_inputs = d_inputs.reshape(-1, 1)
    d_inputs = d_inputs.reshape(t_steps, 4)
    s_inputs = np.array(s_inputs)
    s_inputs = s_inputs.reshape(t_steps, 3)
    inputs = np.concatenate([s_inputs, d_inputs], axis=1)

    return inputs, outputs

t_steps = 365
inputs = []
labels = []
model_num = 9
low_bun = 36
up_bun1 = 61
well_test = [54,55,59,60]

for i in range(model_num+1):
    for j in range(low_bun,up_bun1):
        df_model = df.iloc[i*t_steps*up_bun1:i*t_steps*up_bun1+t_steps*up_bun1]
        df_model = df_model.reset_index()
        x_input, y_output = find_dynamic_input(j,well_locs,df_model,t_steps,up_bun1)
        inputs.append([x_input])
        labels.append([y_output])

inputs2 = np.array(inputs)
inputs3 = inputs2.reshape(inputs2.shape[0]*inputs2.shape[2],inputs2.shape[3])
labels2 = np.array(labels)
labels3 = labels2.reshape(labels2.shape[0]*labels2.shape[2],labels2.shape[3])

# Normalization
# input
norm_inputs = inputs3
scaler = MinMaxScaler()
scaler.fit(inputs3)
norm_inputs = scaler.transform(norm_inputs)
norm_inputs = norm_inputs.reshape(inputs2.shape[0],inputs2.shape[2],inputs2.shape[3])
print(norm_inputs.shape)

# output
norm_labels = labels3
scaler = MinMaxScaler()
scaler.fit(labels3)
norm_labels = scaler.transform(norm_labels )
norm_labels = norm_labels.reshape(labels2.shape[0],labels2.shape[2],labels2.shape[3])
print(norm_labels.shape)

# separate to training and testing
well4test_inputs = [norm_inputs[-1,:,:],norm_inputs[-2,:,:],norm_inputs[-6,:,:],norm_inputs[-7,:,:]]
well4test_labels = [norm_labels[-1,:,:],norm_labels[-2,:,:],norm_labels[-6,:,:],norm_labels[-7,:,:]]
test_inputs = np.concatenate(well4test_inputs,axis=0)
test_labels = np.concatenate(well4test_labels,axis=0)
test_inputs = test_inputs.reshape(4,365,7)
test_labels = test_labels.reshape(4,365,1)

train_inputs = np.delete(norm_inputs, [-1,-2,-6,-7], axis=0)
train_labels = np.delete(norm_labels, [-1,-2,-6,-7], axis=0)
print('Train Input shape == {}'.format(train_inputs.shape))
print('Train Label shape == {}'.format(train_labels.shape))

print('Test Input shape == {}'.format(test_inputs.shape))
print('Test Label shape == {}'.format(test_labels.shape))

# Define model
# Define input layer
LSTM_input = Input(shape=(train_inputs.shape[1],train_inputs.shape[2]),name="Dynamic_Inputs")
print(LSTM_input)

# layer 1
rec_layer_one = LSTM(64,return_sequences=True,name ="LSTM_layer1")(LSTM_input)

rec_layer_one = Dropout(0.2,name ="LSTM_ouput_layer1")(rec_layer_one)

# layer 2
rec_layer_two = LSTM(64,return_sequences=True,name ="LSTM_layer2")(rec_layer_one)

rec_layer_two = Dropout(0.2,name ="LSTM_ouput_layer2")(rec_layer_two)

# layer 3
dense_two = Dense(128, activation='tanh',name="Combined_Dense_layer1")(rec_layer_two)
output = Dense(train_labels.shape[2],activation='tanh',name="Output_layer")(dense_two)

# Compile ModeL
# We first train using binary cross entropy loss and then train the same model using focal loss.
model = Model(inputs=LSTM_input,outputs=output)
# binary cross entropy loss
model.compile(loss='mse', optimizer='adam')
model.summary()

# change float 64 to float 32 to accelerate the trainning process
start_time = time.time() # record the time training the model
history =  model.fit(train_inputs,
                     train_labels, epochs=1000, batch_size= 30,
                     verbose=1, validation_split=0.1)
print("--- %s seconds ---" % (time.time() - start_time))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Save the weight
filename="weight1000ep"
filepath = filename+".hdf5"
model.save_weights(filepath)

# Testing
# plots qw
fig, ax = plt.subplots(2, 2, figsize=(15, 13))
n = 0
well_name = [19, 20, 24, 25]
for i in range(1, 3):
    for j in range(1, 3):
        print(i)
        predict_num = n
        testX = test_inputs[predict_num, :, :]
        testX = testX.reshape(1, testX.shape[0], testX.shape[1])
        testPredict_label = test_labels[predict_num, :, :]
        testPredict = model.predict(testX).reshape(-1, 1)

        # Inverse Normalization
        scaler = MinMaxScaler()
        scaler.fit(labels3)
        testPredict_label_org = scaler.inverse_transform(testPredict_label)
        testPredict_org = scaler.inverse_transform(testPredict)

        testScore = mean_squared_error(testPredict_label_org, testPredict_org, squared=False)
        print('Test Score: %.2f RMSE' % (testScore))

        # Plot
        ax[i - 1, j - 1].plot(range(t_steps), testPredict_label_org, 'ro-', mfc='none', label='Original', linewidth=1)
        ax[i - 1, j - 1].plot(range(t_steps), testPredict_org, 'b-+', label='Predicted', linewidth=1)
        ax[i - 1, j - 1].legend()
        ax[i - 1, j - 1].set_title('Well={}'.format(well_name[n]) + ' RMSE: %.2f' % (testScore))
        ax[i - 1, j - 1].set_xlabel('Time Step')
        ax[i - 1, j - 1].set_ylabel('Water Production Rate (m3/day)')

        n += 1
