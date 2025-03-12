import pandas as pd
import numpy as py
def p_max_deceleration(max_decel):
    '''
    output the probability of 3 classes given max deceleration during the trip
    '''
    if max_decel <= -0.7:
        p_crash = 488/4184
        p_near_crash = 3355/4184
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.7 < max_decel <= -0.6:
        p_crash = 166/2148
        p_near_crash = 1470/2148
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.6 < max_decel <= -0.5:
        p_crash = 182/2590
        p_near_crash = 1742/2590
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.5 < max_decel <= -0.4:
        p_crash = 311/7855
        p_near_crash = 688/7855
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.4 < max_decel <= -0.3:
        p_crash = 477/15927
        p_near_crash = 368/15927
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.3 < max_decel <= -0.2:
        p_crash = 147/6787
        p_near_crash = 53/6787
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.2 < max_decel <= -0.1:
        p_crash = 15/253
        p_near_crash = 5/253
        p_no_crash = 1-p_crash-p_near_crash
    elif -0.1 < max_decel <= 0:
        p_crash = 1/22
        p_near_crash = 2/22
        p_no_crash = 1-p_crash-p_near_crash
    else:
        p_crash = 1/9
        p_near_crash = 0.1/9
        p_no_crash = 1-p_crash-p_near_crash
    
    return p_crash,p_near_crash,p_no_crash

def p_max_turn_rate(max_turn_rate):
    '''
    output the probability of 3 classes given max deceleration during the trip
    '''
    if max_turn_rate <= 0:
        p_crash = 0.01/7
        p_near_crash = 3/7
        p_no_crash = 1-p_crash-p_near_crash
    elif 0 < max_turn_rate <= 5:
        p_crash = 14/122
        p_near_crash = 12/122
        p_no_crash = 1-p_crash-p_near_crash
    elif 5 < max_turn_rate <= 10:
        p_crash = 17/203
        p_near_crash = 26/203
        p_no_crash = 1-p_crash-p_near_crash
    elif 10< max_turn_rate <= 15:
        p_crash = 30/388
        p_near_crash = 47/388
        p_no_crash = 1-p_crash-p_near_crash
    elif 15< max_turn_rate <= 20:
        p_crash = 62/1334
        p_near_crash = 150/1334
        p_no_crash = 1-p_crash-p_near_crash
    elif 20 < max_turn_rate <= 25:
        p_crash = 263/7154
        p_near_crash = 801/7154
        p_no_crash = 1-p_crash-p_near_crash
    elif 25 < max_turn_rate <= 30:
        p_crash = 599/15259
        p_near_crash = 2297/15259
        p_no_crash = 1-p_crash-p_near_crash
    elif 30 < max_turn_rate <= 35:
        p_crash = 418/9820
        p_near_crash = 1843/9520
        p_no_crash = 1-p_crash-p_near_crash
    else:
        p_crash = 330/4502
        p_near_crash = 1128/4502
        p_no_crash = 1-p_crash-p_near_crash
    
    return p_crash,p_near_crash,p_no_crash

def p_cellphone_flag(cellphone_flag):
    """
    flag 0 = no records
        1 = no cell phone usage indicated
        2 = cell phone usage likely
    """
    if cellphone_flag == 0:
        p_crash = 1626/35235
        p_near_crash = 5841/35235
        p_no_crash = 1-p_crash-p_near_crash
    elif cellphone_flag == 1:
        p_crash = 122/3431
        p_near_crash = 535/3431
        p_no_crash = 1-p_crash-p_near_crash
    else:
        p_crash = 76/1964
        p_near_crash = 336/1964
        p_no_crash = 1-p_crash-p_near_crash
    return p_crash,p_near_crash,p_no_crash

