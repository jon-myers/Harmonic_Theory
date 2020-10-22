\version "2.20.0"   %! abjad.LilyPondFile._get_format_pieces()
\language "english" %! abjad.LilyPondFile._get_format_pieces()

\header { %! abjad.LilyPondFile._get_formatted_blocks()
    tagline = ##f
} %! abjad.LilyPondFile._get_formatted_blocks()

\layout {}

\paper {}

\score { %! abjad.LilyPondFile._get_formatted_blocks()
    \new PianoStaff
    <<
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \time 6/4
            c'4
            r4
            ef''4
            r4
            e'''4
            \ottava 1
            b''''4
            \ottava 0
        }
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \clef "bass"
            r4
            f,4
            r4
            \ottava -1
            d,,4
            \ottava 0
            r4
            r4
        }
    >>
} %! abjad.LilyPondFile._get_formatted_blocks()