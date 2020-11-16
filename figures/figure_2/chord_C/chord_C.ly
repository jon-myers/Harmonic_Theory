\version "2.20.0"   %! abjad.LilyPondFile._get_format_pieces()
\language "english" %! abjad.LilyPondFile._get_format_pieces()

\header { %! abjad.LilyPondFile._get_formatted_blocks()
    tagline = ##f
} %! abjad.LilyPondFile._get_formatted_blocks()

\layout {}

\paper {}

\score { %! abjad.LilyPondFile._get_formatted_blocks()
    \new Staff
    \with
    {
        \remove Time_signature_engraver
    }
    {
        \time 5/4
        c'4
        f'4
        _ \markup {
            \raise
                #1.5
                \concat
                    {
                        \tiny
                            -2
                        \hspace
                            #0.9
                        \tiny
                            -49
                        \hspace
                            #0.9
                        \tiny
                            -47
                    }
            }
        fs'4
        cs''4
    }
} %! abjad.LilyPondFile._get_formatted_blocks()