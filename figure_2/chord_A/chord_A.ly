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
            e'4
            _ \markup {
                \center-align
                    \tiny
                        -14
                }
            \once \override Stem.direction = #up
            bf'4
            _ \markup {
                \center-align
                    \tiny
                        -31
                }
        }
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \clef "bass"
            c,4
            s2
        }
    >>
} %! abjad.LilyPondFile._get_formatted_blocks()