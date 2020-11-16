import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
print(parentdir)
sys.path.insert(0,parentdir)
# from utils import make_plot
import abjad
notes = ["f,4", "c4", "g4", "d'4", "a'4"]
notes = [abjad.Note(i) for i in notes]

time_signature = abjad.TimeSignature((5, 4))

two = abjad.Markup('+2').tiny()
four = abjad.Markup('+4').tiny()
six = abjad.Markup('+6').tiny()
eight = abjad.Markup('+8').tiny()
spacer = abjad.Markup.hspace(0.9)
markup_list = [two, spacer, four, spacer, six, spacer, eight]
abjad.attach(abjad.Markup.concat(markup_list, direction=abjad.Up).raise_(1.5), notes[1])

lower_staff = abjad.Staff(notes[:3] + [abjad.Skip('s2')])
lower_staff.remove_commands.append('Time_signature_engraver')
abjad.attach(abjad.Clef('bass'), lower_staff[0])

upper_staff = abjad.Staff([abjad.Skip('s2.')] + notes[3:])
upper_staff.remove_commands.append('Time_signature_engraver')
abjad.attach(time_signature, upper_staff[0])

piano_staff = abjad.StaffGroup([], lilypond_type='PianoStaff')
piano_staff.append(upper_staff)
piano_staff.append(lower_staff)

path = currentdir + '/chord_B/chord_B'
abjad.persist(piano_staff).as_pdf(pdf_file_path=path)
abjad.persist(piano_staff).as_ly(path+'.ly')

points = np.array((
[0, 0, 0],
[-1, 1, 0],
[-3, 1, 1],
[-2, 2, 0],
[-3, 3, 0]
))

draw_points = np.array((
[0, 1, 0],
[-1, 0, 0],
[-1, 2, 0],
[-2, 1, 0],
[-2, 3, 0],
[-3, 2, 0],
[-3, 1, 0],
[-2, 1, 0],
[-1, 1, 1],
[-2, 1, 1]
))
primes = np.array([2.0, 3.0, 5.0])


points = np.roll(points, 2, axis=1)
draw_points = np.roll(draw_points, 2, axis=1)
primes = np.roll(primes, 2)

make_plot(points, primes, 'figure_2/chord_B/chord_B_plot', draw_points = draw_points)
