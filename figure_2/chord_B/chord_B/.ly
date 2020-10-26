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
            \time 5/4
            s2.
            d'4
            a'4
        }
        \new Staff
        \with
        {
            \remove Time_signature_engraver
        }
        {
            \clef "bass"
            f,4
            c4
            ^ \markup {
                \raise
                    #1.5
                    \concat
                        {
                            \tiny
                                +2
                            \hspace
                                #0.9
                            \tiny
                                +4
                            \hspace
                                #0.9
                            \tiny
                                +6
                            \hspace
                                #0.9
                            \tiny
                                +8
                        }
                }
            g4
            s2
        }
    >>
} %! abjad.LilyPondFile._get_formatted_blocks()