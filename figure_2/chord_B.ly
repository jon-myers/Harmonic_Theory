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
            <d' a'>4
            ^ \markup {
                \small
                    +8¢
                }
            _ \markup {
                \small
                    +6¢
                }
        }
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \clef "bass"
            <f, c g>4
            ^ \markup {
                \small
                    +4¢
                }
            ^ \markup {
                \small
                    +2¢
                }
        }
    >>
} %! abjad.LilyPondFile._get_formatted_blocks()