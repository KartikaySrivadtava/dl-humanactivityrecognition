import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FACTOR = 16


def load_data():
    df = pd.read_csv(os.path.dirname(__file__) + '/data/rwhar_data.csv', sep=',')
    return df


def downsampling(df, n):
    df_down = df[df.index % n == 0]
    print("Downsampled file created successfully")
    return df_down


def upsampling(df, df_down, n):
    list_1 = df['sub'].to_numpy()
    list_2 = df_down['x'].to_numpy()
    list_3 = df_down['y'].to_numpy()
    list_4 = df_down['z'].to_numpy()
    list_5 = df['act'].to_numpy()

    list_2_new = []

    for i in range(list_2.size - 1):
        list_2_new.append((np.linspace(list_2[i], list_2[i + 1], n)))

    np_list_2_new = (np.array(list_2_new)).flatten()

    list_3_new = []

    for i in range(list_3.size - 1):
        list_3_new.append((np.linspace(list_3[i], list_3[i + 1], n)))

    np_list_3_new = (np.array(list_3_new)).flatten()

    list_4_new = []

    for i in range(list_4.size - 1):
        list_4_new.append((np.linspace(list_4[i], list_4[i + 1], n)))

    np_list_4_new = (np.array(list_4_new)).flatten()
    np.resize(list_1, (list_1.size - list_2.size))
    np.resize(list_5, (list_1.size - list_2.size))

    df_up = pd.DataFrame(list(zip(list_1, np_list_2_new, np_list_3_new, np_list_4_new, list_5)),
                         columns=['sub', 'x', 'y', 'z', 'act'])
    print("Up-sampled file created successfully")
    return df_up


def resampling(data, n):
    print("Inside resampling")
    print("Resampling factor is", n)

    freq = 50 / n

    # Down-sampling
    df_down = downsampling(data, n)
    df_down.to_csv(os.path.dirname(__file__) + '/data/rwhar_data_down_' + str(freq) + '.csv', index=False)

    # Up-sampling
    df_up = upsampling(data, df_down, n)
    df_up.to_csv(os.path.dirname(__file__) + '/data/rwhar_data_up_' + str(freq) + '.csv', index=False)


def evaluation():
    print("Inside evaluation")

    # Load csv file as dataframe containing values of all activities using predict function of the original program

    eval_df = pd.read_csv(os.path.dirname(__file__) + '/predictions/f1_final.csv', sep=',',
                          index_col=None, header=None)

    # Create dataframes containing values of a particular activity at all frequencies

    df_climb_up = eval_df.loc[eval_df[1] == 'climbingup']
    df_climb_down = eval_df.loc[eval_df[1] == 'climbingdown']
    df_jumping = eval_df.loc[eval_df[1] == 'jumping']
    df_lying = eval_df.loc[eval_df[1] == 'lying']
    df_running = eval_df.loc[eval_df[1] == 'running']
    df_sitting = eval_df.loc[eval_df[1] == 'sitting']
    df_standing = eval_df.loc[eval_df[1] == 'standing']
    df_waling = eval_df.loc[eval_df[1] == 'walking']

    # Create list of values for each activity starting from 50 Hz to 3.125 Hz

    list_climb_up = (np.array(df_climb_up.iloc[:, 0])).astype(float)
    list_climb_down = (np.array(df_climb_down.iloc[:, 0])).astype(float)
    list_jumping = (np.array(df_jumping.iloc[:, 0])).astype(float)
    list_lying = (np.array(df_lying.iloc[:, 0])).astype(float)
    list_running = (np.array(df_running.iloc[:, 0])).astype(float)
    list_sitting = (np.array(df_sitting.iloc[:, 0])).astype(float)
    list_standing = (np.array(df_standing.iloc[:, 0])).astype(float)
    list_walking = (np.array(df_waling.iloc[:, 0])).astype(float)

    # Generic list of frequencies at which these F1 scores have been recorded

    list_freq = ['50Hz', '25Hz', '12.5', '6.25Hz', '3.125Hz']

    # Plotting for all activities F1 score begins from here

    # Climbing up
    x = list_freq
    y = list_climb_up
    ax1 = plt.subplot(2, 4, 1)
    ax1.set_ylim([0, 1])
    ax1.bar(x, y)
    ax1.set_title('F1 climbing up')

    # Climbing down
    x = list_freq
    y = list_climb_down
    ax2 = plt.subplot(2, 4, 2)
    ax2.bar(x, y)
    ax2.set_yticklabels([])
    ax2.set_ylim([0, 1])
    ax2.set_title('F1 climbing down')

    # Jumping
    x = list_freq
    y = list_jumping
    ax3 = plt.subplot(2, 4, 3)
    ax3.bar(x, y)
    ax3.set_yticklabels([])
    ax3.set_ylim([0, 1])
    ax3.set_title('F1 jumping')

    # Lying
    x = list_freq
    y = list_lying
    ax4 = plt.subplot(2, 4, 4)
    ax4.bar(x, y)
    ax4.set_yticklabels([])
    ax4.set_ylim([0, 1])
    ax4.set_title('F1 lying')

    # Running
    x = list_freq
    y = list_running
    ax5 = plt.subplot(2, 4, 5)
    ax5.bar(x, y)
    ax5.set_yticklabels([])
    ax5.set_ylim([0, 1])
    ax5.set_title('F1 running')

    # Sitting
    x = list_freq
    y = list_sitting
    ax6 = plt.subplot(2, 4, 6)
    ax6.bar(x, y)
    ax6.set_yticklabels([])
    ax6.set_ylim([0, 1])
    ax6.set_title('F1 Sitting')

    # Standing
    x = list_freq
    y = list_standing
    ax7 = plt.subplot(2, 4, 7)
    ax7.bar(x, y)
    ax7.set_yticklabels([])
    ax7.set_ylim([0, 1])
    ax7.set_title('F1 Standing')

    # Walking
    x = list_freq
    y = list_walking
    ax8 = plt.subplot(2, 4, 8)
    ax8.bar(x, y)
    ax8.set_yticklabels([])
    ax8.set_ylim([0, 1])
    ax8.set_title('F1 Walking')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.show()

    # Plotting for overall best results

    acc = np.array(
        [np.max(list_climb_up), np.max(list_climb_down), np.max(list_jumping), np.max(list_lying), np.max(list_running),
         np.max(list_sitting), np.max(list_standing), np.max(list_walking)])
    print("acc is ", acc)
    act = np.array(["Climbing-up 50Hz", "Climbing-down 25Hz ", "Jumping 50Hz ", "Lying 25Hz ", "Running 50Hz ",
                    "Sitting 3.125Hz ", "Standing 25Hz ", "Walking 25Hz "])
    c = ['red', 'green', 'red', 'green', 'red', 'blue', 'green', 'green']
    plt.bar(act, height=acc, color=c)
    plt.title("Best F1 scores for all 8 activities for RWHAR dataset")
    plt.xlabel("Activity")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot for overall savings that can be deducted by using best F1 score for each activity from the previous plots

    act = np.array(
        ["Climbing-up", "Climbing-down", "Jumping", "Lying", "Running", "Sitting", "Standing", "Walking", "Total"])
    savings = np.array([0.0, 50.0, 0.0, 50.0, 0.0, 62.5, 50.0, 50.0, 262.5])
    c = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'green']
    plt.bar(act, savings, color=c)
    plt.title("Total saving by picking best accuracy")
    plt.xlabel("Activity")
    plt.ylabel("Savings as percentage")
    plt.show()


def plots_wtih_sensor():
    data_orig = pd.read_csv(os.path.dirname(__file__) + '/predictions/data_15_person/rwhar_data_15_6.25.csv')
    acc_x = data_orig[data_orig.columns[1]]
    acc_y = data_orig[data_orig.columns[2]]
    acc_z = data_orig[data_orig.columns[3]]

    # plot 1:
    plt.subplot(2, 1, 1)
    plt.yticks([])
    plt.xticks([])
    plt.plot(acc_x, color='blue')
    plt.plot(acc_y, color='green')
    plt.plot(acc_z, color='red')
    plt.ylabel("Acceleration values for Ground Truth")
    plt.legend(["acc x", "acc y", "acc z"])
    plt.margins(0)

    # plot 2:
    plt.subplot(2, 1, 2)
    data = pd.read_csv(os.path.dirname(__file__) + '/predictions/prediction_results/results_6.25hz.csv')
    pred = data[data.columns[0]].tolist()
    gt = data[data.columns[1]].tolist()
    idx_climbdown = gt.index(1)
    idx_climbup = gt.index(7)
    idx_jumping = gt.index(5)
    idx_lying = gt.index(4)
    idx_running = gt.index(6)
    idx_sitting = gt.index(2)
    idx_standing = gt.index(0)
    idx_walking = gt.index(3)
    plt.yticks([])
    plt.ylabel("Ground Truth vs Predicted Values")
    plt.pcolor(np.array([gt, pred]), cmap='Set1')
    plt.text(idx_climbdown, 0.05, 'Climbing-up', color='black')
    plt.text(idx_climbup, 0.05, 'Walking', fontdict=None, color='black')
    plt.text(idx_jumping, 0.05, 'Sitting', fontdict=None, color='black')
    plt.text(idx_lying, 0.05, 'Running', fontdict=None, color='black')
    plt.text(idx_running, 0.05, 'Standing', fontdict=None, color='black')
    plt.text(idx_sitting, 0.05, 'Jump', fontdict=None, color='black')
    plt.text(idx_standing, 0.05, 'Climbing-down', fontdict=None, color='black')
    plt.text(idx_walking, 0.05, 'Lying', fontdict=None, color='black')
    plt.pcolor(np.array([gt, pred]), cmap='Set2')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle("RWHAR 3.125Hz")
    plt.show()


if __name__ == '__main__':
    data = load_data()
    resampling(data, FACTOR)
    evaluation()
    plots_wtih_sensor()
