from psychopy import visual, core, event
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from psychopy.hardware import keyboard
import pandas as pd
from nltk.corpus import wordnet as wn
import pickle
from scipy import stats
import string
from hanziconv import HanziConv
max_words = 500
WINDOW_SHAPE = (1366, 768)
global SUBJECT_ID
OUTPUT_FOLDER = "output"
SUBJECT_ID_FILE = os.path.join(OUTPUT_FOLDER, "subject_id_list.txt")
RAW_DUMP_FOLDER = os.path.join(OUTPUT_FOLDER, "raw_dump")
global points


def take_trial_input():
    global SUBJECT_ID

    subject_name = input("\nPlease enter subject name (No spaces)\n")
    f = open(SUBJECT_ID_FILE, "r", encoding="utf-8")
    last_line = f.readlines()[-1]
    ID = int(last_line.split(" ")[0]) + 2
    SUBJECT_ID = ID
    print("Subject ID: ", ID)
    f.close()

    return subject_name, ID


# lets creata  class button that inherits from visual.shape
class Button():

    def __init__(self, win, shape, text):
        self.shape = shape
        self.shape.color = "black"
        self.text = visual.TextStim(win, text=text, pos=self.shape.pos, height=self.shape.width * 0.5, color='white')

        self.win = win
        self.clicked = -1

        self.unClickedColor = 'black'
        self.unseenColor = 'red'
        self.seenColor = 'green'
        self.set_text(text)

    def update_state(self, Pressed):
        clicked_button = -1
        if Pressed[0] > 0:
            clicked_button = 0

        elif Pressed[2] > 0:
            clicked_button = 2
        self.clicked = clicked_button

        if clicked_button == 0:
            self.shape.color = self.seenColor
        elif clicked_button == 2:
            self.shape.color = self.unseenColor

    def set_text(self, num):
        self.text.text = word_list[num]

    def hover_on(self):
        self.shape.height = self.shape.height * 1.1
        self.shape.width = self.shape.width * 1.1

    def hover_off(self):
        self.shape.height = self.shape.height / 1.1
        self.shape.width = self.shape.width / 1.1


def get_button_grid(win, text, n_items=9, *, button_dims=np.array([0.1, 0.1]), borders=np.array([0.05, 0.1]),
                    inter_button_space=np.array([0.5, 0.5])):
    button_grid = []
    row_cols = np.array([np.ceil(np.sqrt(n_items)), np.ceil(np.sqrt(n_items))], dtype=int)
    button_dims = (1 / (row_cols + 1))

    position = np.zeros(2)
    index = 0

    for i in range(row_cols[0]):
        for j in range(row_cols[1]):
            position[0] = ((((i + 1) * button_dims[0])) * 2 - 1) * (
                    1 - borders[0])
            position[1] = ((((j + 1) * button_dims[1])) * 2 - 1) * (
                    1 - borders[1])
            button_grid.append(Button(win, visual.Rect(win, pos=position, width=button_dims[0], height=button_dims[1]),
                                      text=text[index]))
            index += 1
            if index >= n_items:
                break
        if index >= n_items:
            break

    return button_grid


def toggle_button_grid(button_grid, state=0):
    if state == 0:
        for button in button_grid:
            button.shape.autoDraw = False
            button.text.autoDraw = False
    elif state == 1:
        for button in button_grid:
            button.shape.autoDraw = True
            button.text.autoDraw = True


def random_array(n_items, mini=0, maxi=max_words):
    return np.array(np.random.choice(range(mini, maxi), n_items, replace=False))


def recall_array(orignal, perc_recall, shuffle=False, min=0, max=max_words):
    new_array = np.copy(orignal)
    indices = np.random.choice(range(len(new_array)), int(len(new_array) * perc_recall), replace=False)
    replace_indices = np.setdiff1d(range(len(new_array)), indices)
    new_number_choices = np.setdiff1d(range(min, max), new_array)
    new_array[replace_indices] = np.random.choice(new_number_choices, len(replace_indices), replace=False)
    if shuffle:
        np.random.shuffle(new_array)
    for i in range(len(indices)):
        for j in range(len(new_array)):
            if orignal[indices[i]] == new_array[j]:
                indices[i] = j
                break
    return new_array, indices


def check_press(mouse, button_grid):
    for button in button_grid:
        if mouse.isPressedIn(button.shape):
            button.update_state(mouse.getPressed())
            mouse.clickReset()


def get_responses(button_grid):
    responses = np.array([button.clicked for button in button_grid])

    return responses


def analyse_responses(responses, present_indices):
    Hitrate = np.sum(responses[present_indices] == 0)
    Missrate = np.sum(responses[present_indices] == 2)
    not_present_indices = np.setdiff1d(range(len(responses)), present_indices)
    FalseAlarmrate = np.sum(responses[not_present_indices] == 0)
    CorrectRejectionrate = np.sum(responses[not_present_indices] == 2)
    print()
    print(f"HR: {Hitrate / len(present_indices):.2f}", f"FA: {FalseAlarmrate / len(not_present_indices):.2f}", sep="\t")
    print(f"Mi: {Missrate / len(present_indices):.2f}", f"CR: {CorrectRejectionrate / len(not_present_indices):.2f}",
          sep="\t")

    return Hitrate, Missrate, FalseAlarmrate, CorrectRejectionrate


def task_paradigm(win, mouse, n_items, perc_recall, shuffle=False, *, instructions_time=5, memory_time=20,
                  forget_time=5, response_time=20,
                  extra_text="You have to recall these numbers. \nLeftclick on the numbers you remember and right click on the numbers you don't remember seeing "):
    global points
    clock_text = visual.TextStim(win, text="memory_time", pos=(0.8, 0.8), height=0.07, color='yellow')

    numbers = random_array(n_items)

    button_grid = get_button_grid(win, numbers, n_items=n_items)

    Text_stim = visual.TextStim(win, text="Memorise the Words!", pos=(0, 0), height=0.1)
    Text_stim.draw()
    win.flip()
    core.wait(instructions_time)
    toggle_button_grid(button_grid, state=1)
    win.flip()

    timer = core.Clock()
    clock_text.autoDraw = True
    while timer.getTime() < memory_time:
        clock_text.text = "Time Left: " + str(int(memory_time - timer.getTime()))
        win.flip()
    toggle_button_grid(button_grid, state=0)
    clock_text.autoDraw = False

    text = extra_text + "\n Recall starts in "
    Text_stim.text = text + str(forget_time)
    Text_stim.autoDraw = True
    timer.reset()
    while (timer.getTime() < forget_time):
        win.flip()
        Text_stim.text = text + str(int(forget_time - timer.getTime()))
    Text_stim.autoDraw = False

    new_numbers, repeated_indices = recall_array(numbers, perc_recall, shuffle=shuffle)
    new_numbers = new_numbers

    toggle_button_grid(button_grid, state=1)
    for ii, button in enumerate(button_grid):
        button.text.text = word_list[new_numbers[ii]]

    timer.reset()
    clock_text.autoDraw = True
    while timer.getTime() < response_time:
        # check_mouse_hover(mouse, button_grid)
        check_press(mouse, button_grid)
        clock_text.text = "Time Left: " + str(int(response_time - timer.getTime()))
        win.flip()
    clock_text.autoDraw = False

    toggle_button_grid(button_grid, state=0)
    win.flip()
    respones = get_responses(button_grid)
    Hitrate, Missrate, FalseAlarmrate, CorrectRejectionrate = analyse_responses(respones, repeated_indices)
    this_round_points = Hitrate -FalseAlarmrate
    if perc_recall <= 0.3:
        this_round_points = int((Hitrate - 2 * FalseAlarmrate) * 2 / 3)
    elif perc_recall >= 0.7:
        this_round_points = int((2 * Hitrate - FalseAlarmrate) * 2 / 3)
    Text_stim.text = f"You got {this_round_points}  this round"
    Text_stim.autoDraw = True
    win.flip()
    core.wait(1.5)
    Text_stim.autoDraw = False
    points += this_round_points

    return np.array([Hitrate, Missrate, FalseAlarmrate, CorrectRejectionrate])


def block(win, mouse, kb, TEXT_STIM, *, n_items=None, perc_recall=None, shuffle=None, texts=None, Points_stim=None):
    results = np.zeros((len(n_items), 7), dtype=int)
    if type(perc_recall) == float or type(perc_recall) == int:
        temp = perc_recall
        perc_recall = []
        if temp == -1:
            perc_recall = np.random.choice([0.4, 0.5, 0.6], len(n_items), replace=True, p=[0.2, 0.6, 0.2])
        else:
            perc_recall = [temp] * len(n_items)

    if type(shuffle) == bool:
        temp = shuffle
        shuffle = [temp] * len(n_items)
    if type(texts) == str:
        temp = texts
        texts = [temp] * len(n_items)
    for i in range(len(n_items)):
        print('\n', n_items[i])
        TEXT_STIM.text = " Click to start\n Trial " + str( i + 1) + " /" + str(len(n_items)) + "\n"
        TEXT_STIM.draw()
        win.flip()
        while True:
            if mouse.getPressed()[0] or mouse.getPressed()[2]:
                break
        win.flip()
        x = task_paradigm(win, mouse, n_items[i], perc_recall[i], shuffle=shuffle[i], instructions_time=1,
                          memory_time= 2*n_items[i], forget_time=5, response_time=1 + n_items[i],
                          extra_text=texts[i])
        Points_stim.text = f"Points: {points}"
        results[i][:4] = x
        results[i][4] = n_items[i]
        results[i][5] = int(perc_recall[i] * n_items[i])
        results[i][6] = shuffle[i]
    # results are arranged as [Hitrate, Missrate, FalseAlarmrate, CorrectRejectionrate, n_items, perc_recall, shuffle]
    return results


def post_process(main_results, shuffle_results, crit_results):
    # each results  array is a list  with elements
    # [n_items, perc_recall, shuffle, texts, Hits, Misses, FalseAlarms, CorrectRejections, word_list[numbers], word_list[new_numbers], repeated_indices]
    file_name = os.path.join(RAW_DUMP_FOLDER, "p_" + str(SUBJECT_ID))
    np.savez_compressed(file_name, main_results=main_results, shuffle_results=shuffle_results,
                        crit_results=crit_results, allow_pickle=True)
    '''
    # now process the data
    # first get the d_prime and criterion for each block
    main_d_prime, main_criterion = block_d_prime_critera(main_results)
    shuffle_d_prime, shuffle_criterion = block_d_prime_critera(shuffle_results)
    crit_d_prime, crit_criterion = block_d_prime_critera(crit_results)
    #  we now count the different stimuli in main task

    # first main results
    '''

def find_14_(results):
    # [Hit, Miss, FalseAlarms, CorrectRejection, n_items, perc_recall, shuffle]
    # from the above  we find the lowest number for which the person has a HR+CR<1.4

    speeds=np.unique(results[:,4])
    # sort speeeds
    speeds.sort()
    counts=np.zeros(len(speeds))
    PR=np.zeros(len(speeds))

    for i in range(len(results)):
        for j in range(len(speeds)):
            if results[i,4]==speeds[j]:
                counts[j]+=results[i,4]
                PR[j]+=results[i,0]+results[i,3]

    PR/=counts
    for i in range(len(speeds)):
        if PR[i]<=0.7:
            return speeds[i]
    return speeds[-1]

def main():
    global points
    points = 0
    subject_name, _= take_trial_input()
    win = visual.Window(WINDOW_SHAPE, color="gray", fullscr=False)
    win.flip()

    # let's create a rectangular button
    mouse = event.Mouse(win=win)
    POINTS_STIM = visual.TextStim(win, text="points: " + str(points), pos=(-0.9, -0.9), height=0.05)
    POINTS_STIM.autoDraw = True

    TEXT_STIM = visual.TextStim(win, text="", pos=(0, 0), height=0.1)

    # create keyboard object
    kb = keyboard.Keyboard()
    kb.keys = ['space']

    # training phase
    temp = block(win, mouse, kb, TEXT_STIM, n_items=[16,8,9,12], perc_recall=-1.0, shuffle=[False, True, True, False], texts=["Training Phase"]*4, Points_stim=POINTS_STIM)
    print(temp)
    print(find_14_(temp))
    points = 0
    # main_task
    TEXT_STIM.text = " Now the main task begins. You will be rewarded with +1 point for each correct recall and -1 point for each false alarm"
    TEXT_STIM.draw()
    win.flip()
    core.wait(5)
    n_items = [ 6, 6, 6, 9,9,9,12,12,12,16,16,16]

    main_task = block(win, mouse, kb, TEXT_STIM, n_items=n_items, perc_recall=-1.0, shuffle=True, texts="",
                      Points_stim=POINTS_STIM)
    ideal_speed=find_14_(main_task)

    # shuffle vs lack of shuffle effect
    # from the above  we find the lowest number for which the person has a HR+CR<1.4

    n_items = [ideal_speed] * 2

    shuffle = [False,False]
    shuffle = np.random.permutation(shuffle)
    TEXT_STIM.text = " The rewards remains +1 for correct recall and -1 for false alarm"
    TEXT_STIM.draw()
    win.flip()
    core.wait(5)
    shuffle_effect = block(win, mouse, kb, TEXT_STIM, n_items=n_items, perc_recall=-1.0, shuffle=shuffle, texts="",
                           Points_stim=POINTS_STIM)

    # changing the criterion
    n_items = [ideal_speed] * 6
    perc_recall = [0.2,0.2,0.2,0.8,0.8,0.8]
    perc_recall = np.random.permutation(perc_recall)
    extratrext = []
    TEXT_STIM.text = " In this round the rewards can change. \n On some trials most items will repeat. And the reward is higher for correct recall. \n On other trials, most items will be new and the penalty for false alarm is higher"
    TEXT_STIM.draw()
    win.flip()
    core.wait(10)
    for i in range(len(n_items)):
        if perc_recall[i] < 0.5:
            extratrext.append(
                "Very Few items will be repeated,\n So please be very stringent. You will be penalised for too many false alarms.\n\n +1 for correct recall and -2 for false alarm\n")
        else:
            extratrext.append(
                "Almost all items will be repeated,\n So you can be liberal with your responses. You will be penalised for too many misses.\n\n +2 for correct recall and -1 for false alarm\n")

    dif_crit = block(win, mouse, kb, TEXT_STIM, n_items=n_items, perc_recall=perc_recall, shuffle=True,
                     texts=extratrext, Points_stim=POINTS_STIM)
    win.flip()

    # post process the data
    post_process(main_task, shuffle_effect, dif_crit)
    save_status(SUBJECT_ID,subject_name,1)
    print("success")
    win.close()


def save_status(ID, Name, success=1):
    if success == -1:
        return
    f = open(SUBJECT_ID_FILE, "a", encoding="utf-8")
    f.write("\n" + str(ID) + " " + Name + " " + str(success))
    f.close()


def extract_data(inp, out):
    data = pd.read_csv(inp)
    nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
    # lets take the column called 'word' and make a numpy array
    words = data['word'].to_numpy()
    words = words[:10_000]
    # we select the top 100 word with  5 or 6  letters

    word_count = 0

    with open(out, 'w') as f:
        for index, word in enumerate(words):
            if (len(word) == 5 or len(word) == 6 or len(word) == 7) and word in nouns:
                f.write(word + '\n')
                word_count += 1
                if word_count >= max_words:
                    break


def load_word_list(file):
    with open(file, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    return np.array(words)


word_list = None


def number_list(min, max, Output):
    # we add all numbers in the range with no repeating digits
    numbers = []
    # we take all combinations of numbers and letters  26*10
    for t in range(1, 510):
        numbers.append(HanziConv.toSimplified(chr(0x4E00 + t)) )

    number = np.array(numbers)
    np.savez(Output, number=number, allow_pickle=True)


if __name__ == '__main__':
    '''EXTRACT_INPUT = 'unigram_freq.csv'
    EXTRACT_OUTPUT = 'best_5_letter.csv'
    if extract:
        extract_data(EXTRACT_INPUT, EXTRACT_OUTPUT)
        print("Extracted")
    word_list = load_word_list(EXTRACT_OUTPUT)
    print(" words loaded: folder exists  "+str(os.path.isdir(RAW_DUMP_FOLDER)))
    main()'''
    OUTPUT = "3_digit_numbers.npz"
    Extract = True
    if Extract:
        number_list(100, 1000, OUTPUT)
        print("done")
    word_list = np.load(OUTPUT)["number"]
    max_words = len(word_list)

    print(" words loaded: folder exists  " + str(os.path.isdir(RAW_DUMP_FOLDER)))
    main()

''' ToDo:
- Make buttons change color: DOne
- Make buttons change size when hover: maybe
- get a grid of buttons
- get a status bar at the top that shows time
'''

# initially, Points= (HR-FA)*n_items
# depending on conditions  it can become (1.5*HR-FA)*n_items or (HR-FA*1.5)*n_items
