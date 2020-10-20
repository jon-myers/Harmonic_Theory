\version "2.20.0"   %! abjad.LilyPondFile._get_format_pieces()
\language "english" %! abjad.LilyPondFile._get_format_pieces()

\header { %! abjad.LilyPondFile._get_formatted_blocks()
    tagline = ##f
} %! abjad.LilyPondFile._get_formatted_blocks()

\layout {}

\paper {}

\score { %! abjad.LilyPondFile._get_formatted_blocks()
    \new Staff
    {
        <e' bf'>4
        ^ \markup {
            \small
                -31¢
            }
        _ \markup {
            \small
                -14¢
            }
    }
} %! abjad.LilyPondFile._get_formatted_blocks()