import numpy as np
import os,sys,inspect, abjad
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios

points = np.array((
[0, 0, 0],
[-1, 0, 0],
[-1, 0, 1],
[0, 0, -1],
[0, 1, 0],
[1, 1, 0]
))

primes = np.array((3.0, 5.0, 7.0))
octaves = np.array(((0, 0, 0), (-1, -1, -1)))

shifts = ['A', 'B']
for s, shift in enumerate(shifts):

    ratios = get_ratios(points, primes, octaves[s], string = False)
    cents = [hz_to_cents(i, 1) for i in ratios]
    pitch_steps = np.array([round(i/100) for i in cents])
    print(pitch_steps)
    octs = [ps // 12 for ps in pitch_steps]
    pitch_class_steps = [pitch_steps[i] - 12 * octs[i] for i in range(len(octs))]
    devs = np.array([round(cents[i] - pitch_steps[i] * 100) for i in range(len(cents))])

    order = np.argsort(pitch_steps)
    zero_index = [i for i in pitch_steps[order]].index(0)

    notes = [abjad.Note(i, abjad.Duration(1, 4)) for i in pitch_steps]
    for i, item in enumerate(pitch_steps):
        if item < -22:
            abjad.attach(abjad.Ottava(n=-1), notes[i])
            abjad.attach(abjad.Ottava(n=0, format_slot="after"), notes[i])
        elif item > 35:
            abjad.attach(abjad.Ottava(n=1), notes[i])
            abjad.attach(abjad.Ottava(n=0, format_slot="after"), notes[i])

    # notes = [abjad.Note(i, abjad.Duration(1, 4)) for i in pitch_steps[order]]
    # for i, item in enumerate(pitch_steps[order]):
    #     if item < -22:
    #         abjad.attach(abjad.Ottava(n=-1), notes[i])
    #         abjad.attach(abjad.Ottava(n=0, format_slot="after"), notes[i])
    #     elif item > 35:
    #         abjad.attach(abjad.Ottava(n=1), notes[i])
    #         abjad.attach(abjad.Ottava(n=0, format_slot="after"), notes[i])
    lower_staff = abjad.Staff([])
    upper_staff = abjad.Staff([])
    for i, note in enumerate(notes):
        if pitch_steps[i] < 0:
            lower_staff.append(note)

            upper_staff.append(abjad.Rest((1,4)))
        else:
            upper_staff.append(note)
            lower_staff.append(abjad.Rest((1, 4)))
        
    devs = devs[order]
    abjad.attach(abjad.Clef('bass'), lower_staff[0])
    piano_staff = abjad.StaffGroup([], lilypond_type='PianoStaff')
    piano_staff.append(upper_staff)
    piano_staff.append(lower_staff)
    time_signature = abjad.TimeSignature((6, 4))

    abjad.attach(time_signature, upper_staff[0] )
    upper_staff.remove_commands.append('Time_signature_engraver')
    lower_staff.remove_commands.append('Time_signature_engraver')
    path = currentdir + '/shift_' + shift + '_notation'
    abjad.persist(piano_staff).as_pdf(pdf_file_path=path)
    abjad.persist(piano_staff).as_ly(path+'.ly')
    make_plot(points, primes, currentdir + '/shift_' + shift, octaves[s])
