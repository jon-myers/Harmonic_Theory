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
            \time 3/4
            s4
            s4
            c'4
            ef''4
            e'''4
            b''''4
        }
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            d,,4
            f,4
            s4
            s4
            s4
            s4
        }
    >>
} %! abjad.LilyPondFile._get_formatted_blocks()