import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0,parentdir)
from utils import make_plot
import abjad

notes = ["c,4", "e'", "bf'"]
notes = [abjad.Note(i) for i in notes]


# chord = abjad.Chord("<e' bf'>4")
# note = abjad.Note("c,4")
time_signature = abjad.TimeSignature((3, 4))
abjad.attach(abjad.Markup('-14', direction=abjad.Down).tiny().center_align(), notes[1])
abjad.attach(abjad.Markup('-31', direction=abjad.Down).tiny().center_align(), notes[2])
piano_staff = abjad.StaffGroup([], lilypond_type='PianoStaff')
upper_staff = abjad.Staff([abjad.Skip('s4')] + notes[1:] )

abjad.override(upper_staff[2]).stem.direction = abjad.Up
abjad.attach(time_signature, upper_staff[0])
bass_clef = abjad.Clef('bass')
lower_staff = abjad.Staff(notes[:1] + [abjad.Skip('s2')])
print(lower_staff)
abjad.attach(bass_clef, lower_staff[0])
piano_staff.append(upper_staff)
piano_staff.append(lower_staff)
upper_staff.remove_commands.append('Time_signature_engraver')
lower_staff.remove_commands.append('Time_signature_engraver')


path = currentdir + '/chord_A'
abjad.persist(piano_staff).as_pdf(pdf_file_path=path)
abjad.persist(piano_staff).as_ly(path+'.ly')

points = np.array((
[0, 0, 0],
[0, 1, 0],
[0, 0, 1]
))

primes = np.array([3.0, 5.0, 7.0])
make_plot(points, primes, 'figure_2/chord_A_plot')
