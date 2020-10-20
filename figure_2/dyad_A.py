import abjad
c = abjad.Note(0, abjad.Duration(1, 4))
chord = abjad.Chord("<e' bf'>4")
time_signature = abjad.TimeSignature((1, 4), hide=True)

abjad.attach(abjad.Markup('-31'+u'\xa2', direction=abjad.Up).small(), chord)
abjad.attach(abjad.Markup('-14'+u'\xa2', direction=abjad.Down).small(), chord)

staff = abjad.Staff([chord])
abjad.attach(time_signature, staff[0])
path = '~/documents/2020/harmony-figures/figure_2/chord_A.py'
abjad.persist(staff).as_pdf(pdf_file_path=path)
