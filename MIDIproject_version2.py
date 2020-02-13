#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import  pandas as pd
import pretty_midi
from music21 import *
import os
import mathExtraction
import random


# ### Functions

# In[33]:


def extract_notes(midi_part):
    parent_element = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, note.Note):
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                nt_in_chord=note.Note(nameWithOctave=pitch.nameWithOctave, duration=nt.duration)
                parent_element.append(nt_in_chord)
    
    return parent_element

def extract_nonagg_chords(midi_part):
    parent_element = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, chord.Chord):
            parent_element.append(nt)
    
    return parent_element

def simplify_roman_name(roman_numeral):
    # Chords can get nasty names as "bII#86#6#5",
    # in this method we try to simplify names, even if it ends in
    # a different chord to reduce the chord vocabulary and display
    # chord function clearer.
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()
    
    # Checking valid inversions.
    if ((roman_numeral.isTriad() and inversion < 3) or
            (inversion < 4 and
                 (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()
        
    if (inversion_name is not None):
        ret = ret + str(inversion_name)
        
    elif (roman_numeral.isDominantSeventh()): ret = ret + "M7"
    elif (roman_numeral.isDiminishedSeventh()): ret = ret + "o7"
    return ret


def note_count(measure, count_dict):
    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        # All notes have the same length of its chord parent.
        note_length = chord.quarterLength
        for note in chord.pitches:          
            # If note is "C5", note.name is "C". We use "C5"
            # style to be able to detect more precise inversions.
            note_name = str(note) 
            if (bass_note is None or bass_note.ps > note.ps):
                bass_note = note
                
            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length
        
    return bass_note



def harmonic_reduction(midi_file):
    ret = []
    temp_midi = stream.Score()
    temp_midi_chords = midi_file.chordify()
    temp_midi.insert(0, temp_midi_chords)    
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 7   
    for m in temp_midi_chords.measures(0, None): # None = get all measures.
        if (type(m) != stream.Measure):
            continue
        
        # Here we count all notes length in each measure,
        # get the most frequent ones and try to create a chord with them.
        count_dict = dict()
        bass_note = note_count(m, count_dict)
        if (len(count_dict) < 1):
            ret.append("-") # Empty measure
            continue
        
        sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)
        
        # Convert the chord to the functional roman representation
        # to make its information independent of the music key.
        roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
        ret.append(simplify_roman_name(roman_numeral))
        #ret.append(roman_numeral)
    return ret


# ### loop within the current folder for the midi file parsing

# In[3]:


path = 'C:\\pythonMidi\\'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.mid' in file:
            files.append(os.path.join(r, file))
            

random.SystemRandom().shuffle(files)


# In[208]:


fnow=files[0]
f_index=0
dict_all={}


# In[14]:


# f_index = 0
# for f in files:
#     try:


# ### import file into both pretty midi and music21 libraries

# In[15]:


pm_data=pretty_midi.PrettyMIDI(fnow)
mf = midi.MidiFile()
mf.open(fnow)
mf.read()
mf.close()
m21_data=midi.translate.midiFileToStream(mf)


# ### correct accidentals

# In[17]:


ks = m21_data.analyze('key')
for n in m21_data.recurse().notes:  # we need to recurse because the notes are in measures...
    if isinstance(n, note.Note):
        nStep = n.pitch.step
        rightAccidental = ks.accidentalByStep(nStep)
        n.pitch.accidental = rightAccidental
#m21_data.show('text')


# ### Features that can be extracted from entire midi data:
# ##### Score overall
# 
# - n_intru(int): number of instrument tracks
# - n_beatrack(int): number of precussion tracks
# - n_voice(int): total track number
# 
# ##### rhythm
# 
# - e_tempo(float): estimated tempo/speed
# - time_sig_numer(int) time signature numerator
# - time_sig_denom(int): time signature denominator
# - time_sig_divclass (str): classification of the time signature: simple triple
# - tol_measure(int): total measure number
# - tol_time(float): end time in sec
# - note_duration_occ (dict): note_legth (str) vs. occurance (int)
# - time_sig_change_cnt (int): number of time signature changes
# - tempo_change_cnt (int): number of time tempo changes
# 
# ##### molody
# 
# - pitch_class_occ(dict): pitch class (12 str-key) with : occurance (int)
# - key_sig(dict): key signature (str-key) estimate with: confidence (float)
# 
# ##### chord
# 
# - chord_occ(dict): chord (str-key) with: occurance (int)

# In[36]:


n_intru = 0
n_beatrack = 0
for i in pm_data.instruments:
    if not i.is_drum:
        n_intru += 1
    else:
        n_beatrack += 1
n_voice = len(m21_data.recurse().voices)



e_tempo=pm_data.estimate_tempo()

time_sig_numer=m21_data.getTimeSignatures()[0].numerator

time_sig_denom=m21_data.getTimeSignatures()[0].denominator

time_sig_divclass=m21_data.getTimeSignatures()[0].classification

end_time_in_quater=m21_data.flat.highestOffset
end_time_in_beats=math.ceil(end_time_in_quater / 4) * time_sig_denom
tol_measure = end_time_in_beats / time_sig_numer

tol_time = end_time_in_beats / e_tempo * 60

all_notes=extract_notes(m21_data.flat)
all_durations = [d.duration.type for d in all_notes]
note_duration_occ=dict(pd.Series(all_durations).value_counts())

time_sig_change_cnt=len(pm_data.time_signature_changes) - 1
measure_number, tempo_now = pm_data.get_tempo_changes()
tempo_change_cnt = len(tempo_now)

all_pitch_class = [d.pitch.name for d in all_notes]
pitch_class_occ = dict(pd.Series(all_pitch_class).value_counts())

analysis_key = m21_data.analyze('key')
key_proba_dict = {analysis_key.name : analysis_key.correlationCoefficient}
for key in analysis_key.alternateInterpretations:
    key_proba_dict.update({key.name : key.correlationCoefficient})

reduced_chords = harmonic_reduction(m21_data)
chord_occ= dict(pd.Series(reduced_chords).value_counts())


# ### n-gram analysis
# ##### rhythm
# 
# - note duration pattern
# 
# ##### melody
# 
# - interval : 2-gram
# 
# ##### harmony
# 
# - chord progression

# In[198]:


from sklearn.feature_extraction.text import CountVectorizer


# In[199]:


vectorizer = CountVectorizer(stop_words='english', ngram_range=(time_sig_numer,time_sig_numer))
all_durations_in_one= ' '.join(all_durations)
X = vectorizer.fit_transform([all_durations_in_one])
notepattern_df=pd.DataFrame(X.toarray(), columns=['pat: '+x for x in vectorizer.get_feature_names()])
note_pattern=dict(notepattern_df.loc[0].sort_values(ascending=False).head(5))
note_pattern


# In[200]:


all_next_notes=all_notes.copy()
all_next_notes.pop(0)
i=0
intervallist=[]
consonantlist=[]
for n in all_next_notes:
    intervallist.append(interval.Interval(noteStart=all_notes[i], noteEnd=n).simpleName)
    consonantlist.append(interval.Interval(noteStart=all_notes[i], noteEnd=n).isConsonant())

interval_occ = dict(pd.Series(intervallist).value_counts())
consonant_percent=sum(consonantlist)/len(consonantlist) * 100


# In[201]:


vectorizer = CountVectorizer(stop_words='english', ngram_range=(time_sig_numer-1,time_sig_numer-1))
all_interval_in_one = ' '.join(intervallist)
X = vectorizer.fit_transform([all_interval_in_one])
intervalpattern_df=pd.DataFrame(X.toarray(), columns=['pat: '+x for x in vectorizer.get_feature_names()])
interval_pattern=dict(intervalpattern_df.loc[0].sort_values(ascending=False).head(5))
interval_pattern


# In[202]:


import re
reduced_chords_name=[]
for c in reduced_chords:
    reduced_chords_name.append(re.split('(\d+)', c)[0])


# In[203]:


vectorizer = CountVectorizer(stop_words='english', ngram_range=(3,3))
all_chords_in_one = ' '.join(reduced_chords_name)
X = vectorizer.fit_transform([all_chords_in_one])
chordpattern_df=pd.DataFrame(X.toarray(), columns=['pat: '+x for x in vectorizer.get_feature_names()])
chord_pattern =dict(chordpattern_df.loc[0].sort_values(ascending=False).head(5))
chord_pattern


# ### write data into DataFrame

# In[204]:


column_names = ['n_intru', 'n_beatrack','n_voice', 'e_tempo','time_sig_numer','time_sig_denom','time_sig_divclass'
                ,'tol_measure','tol_time','time_sig_change_cnt', 'tempo_change_cnt','consonant_percent']
row_sig_value_data=[n_intru, n_beatrack, n_voice,  e_tempo, time_sig_numer,  time_sig_denom, time_sig_divclass
                 ,tol_measure ,tol_time, time_sig_change_cnt,  tempo_change_cnt, consonant_percent]

pd_row = pd.DataFrame(dict(zip(column_names,row_sig_value_data)), index=[f_index])


pd_duration = pd.DataFrame(note_duration_occ, index=[f_index])
pd_pclass = pd.DataFrame(pitch_class_occ, index= [f_index])
pd_sig = pd.DataFrame(key_proba_dict, index= [f_index])
pd_chord = pd.DataFrame(chord_occ, index= [f_index])

pd_note_pattern = pd.DataFrame(note_pattern, index= [f_index])
pd_interval_occ = pd.DataFrame(interval_occ, index= [f_index])
pd_interval_pattern = pd.DataFrame(interval_pattern, index= [f_index])
chord_pattern = pd.DataFrame(chord_pattern, index= [f_index])

df=pd.concat([pd_row, pd_duration, pd_pclass, pd_sig, pd_chord, pd_note_pattern, pd_interval_occ, pd_interval_pattern, chord_pattern], axis=1, sort=False)


# In[205]:


pd.set_option('max_columns', None)


# In[206]:


df


# In[209]:


ser_now=df.iloc[0]
dict_all.update({f_index : ser_now})

df=pd.DataFrame(dict_all).T
f_index+=1


# In[210]:


df


# In[ ]:




