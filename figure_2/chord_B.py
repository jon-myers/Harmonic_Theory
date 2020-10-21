import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot
import abjad

chord_1 = abjad.Chord("<f, c g>4")
chord_2 = abjad.Chord("<d' a'>4")
time_signature = abjad.TimeSignature((1, 4), hide=True)

abjad.attach(abjad.Markup('+4' + u'\xa2', direction=abjad.Up).small(), chord_1)
abjad.attach(abjad.Markup('+2' + u'\xa2', direction=abjad.Up).small(), chord_1)

abjad.attach(abjad.Markup('+6' + u'\xa2', direction=abjad.Down).small(), chord_2)
abjad.attach(abjad.Markup('+8' + u'\xa2', direction=abjad.Up).small(), chord_2)

piano_staff = abjad.StaffGroup([], lilypond_type='PianoStaff')
upper_staff = abjad.Staff([chord_2])
bass_clef = abjad.Clef('bass')
lower_staff = abjad.Staff([chord_1])
abjad.attach(bass_clef, lower_staff[0])
piano_staff.append(upper_staff)
piano_staff.append(lower_staff)
upper_staff.remove_commands.append('Time_signature_engraver')
lower_staff.remove_commands.append('Time_signature_engraver')


path = currentdir + '/chord_B'
abjad.persist(piano_staff).as_pdf(pdf_file_path=path)
abjad.persist(piano_staff).as_ly(path+'.ly')

points = np.array((
[0, 0, 0],
[0, 1, 0],
[0, 0, 1]
))

primes = np.array([3.0, 5.0, 7.0])
make_plot(points, primes, 'figure_2/chord_B_plot')
