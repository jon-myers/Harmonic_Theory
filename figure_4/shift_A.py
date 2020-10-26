import numpy as np
import os,sys,inspect, abjad
import more_itertools
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
octaves = np.array(((0, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0), (-1, -1, -1), (-1, -2, -2)))

shifts = ['A', 'B', 'C', 'D', 'E', 'F']
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
    colors_ = ['black', 'SaddleBrown', 'MediumSeaGreen', 'red', 'DarkOrchid', 'RoyalBlue']
    colors = [abjad.SchemeColor(i) for i in colors_]
    for i, note in enumerate(notes):
        abjad.override(note).note_head.color = colors[i]
        
    for i, item in enumerate(pitch_steps):
        if item < -22:
            abjad.attach(abjad.Ottava(n=-1), notes[i])
            abjad.attach(abjad.Ottava(n=0, format_slot="after"), notes[i])
        elif item > 31:
            abjad.attach(abjad.Ottava(n=1), notes[i])
            abjad.attach(abjad.Ottava(n=0, format_slot="after"), notes[i])

    lower_staff = abjad.Staff([])
    upper_staff = abjad.Staff([])
    for i, note in enumerate(notes):
        if pitch_steps[i] < 0:
            lower_staff.append(note)

            upper_staff.append(abjad.Skip((1,4)))
        else:
            upper_staff.append(note)
            lower_staff.append(abjad.Skip((1, 4)))
        
    devs = [i for i in devs if i != 0]
    devs = [abjad.Markup(str(i)).tiny() if i < 1 else abjad.Markup('+'+str(i)).tiny() for i in devs]
    spacer = abjad.Markup.hspace(1.7)
    markup_list = more_itertools.intersperse(spacer, devs)
    abjad.attach(abjad.Markup.concat(markup_list, direction=abjad.Up).raise_(1.5), notes[1])
    
    abjad.attach(abjad.Clef('bass'), lower_staff[0])
    piano_staff = abjad.StaffGroup([], lilypond_type='PianoStaff')
    piano_staff.append(upper_staff)
    piano_staff.append(lower_staff)
    time_signature = abjad.TimeSignature((6, 4))

    abjad.attach(time_signature, upper_staff[0] )
    upper_staff.remove_commands.append('Time_signature_engraver')
    lower_staff.remove_commands.append('Time_signature_engraver')
    path = currentdir + '/shift_' + shift + '_notation'
    
    layout_block = abjad.Block(name='layout')
    context = abjad.ContextBlock(source_lilypond_type='Score')
    abjad.override(context).spacing_spanner.base_shortest_duration = abjad.SchemeMoment((1, 12))
    layout_block.items.append(context)

    lilypond_file = abjad.LilyPondFile.new(piano_staff)
    lilypond_file.items.append(layout_block)
    
    abjad.persist(lilypond_file).as_pdf(pdf_file_path=path)
    abjad.persist(lilypond_file).as_ly(path+'.ly')
    make_plot(points, primes, currentdir + '/shift_' + shift , octaves[s], dot_size=4, 
          colors=colors_, ratios=False)
