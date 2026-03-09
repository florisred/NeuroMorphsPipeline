import numpy as np

def choose_transitions(labels):
    unique_trans = np.unique(labels["pair_key"].dropna())
    chosen_transitions = []
    print("Possible transisitions:")
    for trans in unique_trans:
        print(trans)
    while True:
        print(f"Current list: {chosen_transitions}")
        chosen_trans = input("Add or remove transitions, type q to stop:")
        if chosen_trans == "q":
            break
        elif chosen_trans not in chosen_transitions:
            if chosen_trans not in unique_trans:
                print("please choose a valid transition")
            chosen_transitions.append(chosen_trans)
        elif chosen_trans in chosen_transitions:
            chosen_transitions.remove(chosen_trans)
        else:
            print("how did you get here?? I'm gonna crash because this shouldn't be possible. congrats!")
            raise AssertionError("what the hell")
    return chosen_transitions